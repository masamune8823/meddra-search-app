import openai
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# 外部からインポート（configで統一管理）
from config import faiss_index, meddra_terms, synonym_df

# モデルの読み込み（MiniLMベース）
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# クエリをベクトル化
def encode_query(query):
    return model.encode([query])[0]

# GPTベースでSOCカテゴリを予測
def predict_soc_keywords_with_gpt(query, client, model="gpt-3.5-turbo"):
    prompt = f"""
以下の症状は、MedDRAで定義されるどのSOCカテゴリに最も関連していますか？

症状: 「{query}」

可能性のあるSOCカテゴリ（日本語で1～3個程度）を、簡潔なキーワードで出力してください。
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    keywords = response.choices[0].message.content.strip().splitlines()
    return [kw.replace("-", "").strip("・:：●- ") for kw in keywords if kw]

# SOCやHLGTの階層でフィルタ
def filter_by_predicted_soc(results_df, soc_keywords):
    if not soc_keywords:
        return results_df
    mask = results_df[["SOC", "HLGT", "HLT"]].apply(
        lambda cols: cols.astype(str).str.contains("|".join(soc_keywords), case=False).any(axis=1),
        axis=1
    )
    return results_df[mask].copy()

# MedDRA検索（FAISS＋synonym_df）
def search_meddra(query, top_k=10):
    query_vector = encode_query(query)
    D, I = faiss_index.search(np.array([query_vector]), top_k)
    results = [(meddra_terms[i], float(D[0][idx]), "faiss") for idx, i in enumerate(I[0])]

    # synonym_dfを用いた拡張
    if query in synonym_df.index:
        synonyms = synonym_df.loc[query]["synonyms"]
        for syn in synonyms:
            syn_vec = encode_query(syn)
            D_syn, I_syn = faiss_index.search(np.array([syn_vec]), top_k)
            for idx, i in enumerate(I_syn[0]):
                results.append((meddra_terms[i], float(D_syn[0][idx]), "synonym_faiss"))
    return results

# GPTによる再スコアリング（Top10件）
def rerank_results_v13(query, results, top_n=10, cache_path="score_cache.pkl"):
    import os
    import pickle

    key = (query, tuple(results))
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        if key in cache:
            return cache[key]

    top_terms = [term for term, *_ in results[:top_n]]
    scores = []
    for term in top_terms:
        prompt = f"{query} に関連する症状名として、 {term} はどの程度妥当ですか？パーセンテージで答えてください（例: 80%）"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        content = response["choices"][0]["message"]["content"]
        percent = "".join(c for c in content if c.isdigit())
        score = int(percent) if percent else 50
        scores.append(score)

    df = pd.DataFrame({
        "term": top_terms,
        "score": scores
    })
    df["確からしさ（％）"] = df["score"].round(1)
    result_df = df.merge(pd.DataFrame(results, columns=["term", "raw_score", "source"]), on="term", how="left")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}
    cache[key] = result_df
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    return result_df

# MedDRAの階層情報を付加
def add_hierarchy_info(results_df, term_master_df):
    return results_df.merge(term_master_df, how="left", left_on="term", right_on="PT_Japanese")

# term_master_dfの読み込み
def load_term_master_df(path):
    return pd.read_pickle(path)