import os
import pickle
import faiss
import numpy as np
import pandas as pd
import openai

# クエリ拡張（GPT）
def expand_query_gpt(query):
    cache_path = "query_expansion_cache.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    if query in cache:
        return cache[query]

    prompt = f"以下は医療症状に関する言葉です。「{query}」から連想される代表的な症状名を日本語で3つ、コンマ区切りで出力してください。"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    keywords = response.choices[0].message.content.strip()
    expanded = [kw.strip() for kw in keywords.split(",") if kw.strip()]

    cache[query] = expanded
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    return expanded

# ベクトル化（MiniLM）
def encode_query(query, model=None):
    from sentence_transformers import SentenceTransformer
    if model is None:
        model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return model.encode([query])[0]

# MedDRA検索（synonym_df + FAISS）
def search_meddra(query, faiss_index, meddra_terms, synonym_df, top_k=10):
    query_vec = encode_query(query)
    _, indices = faiss_index.search(np.array([query_vec]), top_k)

    term_candidates = list(meddra_terms[indices[0]])

    if query in synonym_df.index:
        synonyms = synonym_df.loc[query]["synonyms"]
        for syn in synonyms:
            syn_vec = encode_query(syn)
            _, syn_indices = faiss_index.search(np.array([syn_vec]), top_k)
            term_candidates.extend(list(meddra_terms[syn_indices[0]]))

    unique_terms = list(set(term_candidates))
    results_df = pd.DataFrame(unique_terms, columns=["term"])
    results_df["score"] = 0.0
    results_df["source"] = "FAISS+synonym"
    return results_df

# 再スコアリング（GPT）
def rerank_results_v13(query, results_df):
    cache_path = "score_cache.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            score_cache = pickle.load(f)
    else:
        score_cache = {}

    top_terms = results_df["term"].tolist()[:10]
    scored = []

    for term in top_terms:
        key = (query, term)
        else:
            prompt = f"以下の症状にどれだけ関連があるかを100点満点で評価してください。\n症状: {query}\n候補用語: {term}\nスコア:"
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            score = float(response.choices[0].message.content.strip().split("\n")[0])
            score_cache[key] = score

        scored.append((term, score))

    with open(cache_path, "wb") as f:
        pickle.dump(score_cache, f)

    reranked = pd.DataFrame(scored, columns=["term", "score"])
    reranked["source"] = "reranked"
    return reranked

# 階層情報を付加
def add_hierarchy_info(results_df, term_master_df):
    merged = results_df.merge(
        term_master_df[["PT_Japanese", "HLT_Japanese", "HLGT_Japanese", "SOC_Japanese"]],
        how="left",
        left_on="term",
        right_on="PT_Japanese"
    )
    merged = merged.drop(columns=["PT_Japanese"])
    merged = merged.rename(columns={
        "HLT_Japanese": "HLT",
        "HLGT_Japanese": "HLGT",
        "SOC_Japanese": "SOC"
    })
    return merged

# スコア再スケーリング
def rescale_scores(df, col="score"):
    min_val = df[col].min()
    max_val = df[col].max()
    if max_val > min_val:
        df["確からしさ（％）"] = ((df[col] - min_val) / (max_val - min_val)) * 100
    else:
        df["確からしさ（％）"] = 100
    df["確からしさ（％）"] = df["確からしさ（％）"].round(1)
    return df

# LLMでSOCカテゴリ予測
def predict_soc_keywords_with_gpt(query):
    cache_path = "soc_predict_cache.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    if query in cache:
        return cache[query]

    prompt = f"以下の医療症状に関連しそうなMedDRAのSOCカテゴリのキーワードを3つ、日本語で挙げてください（例：神経、皮膚、消化）。\n\n症状: {query}\nキーワード:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    keywords = response.choices[0].message.content.strip()
    predicted = [kw.strip() for kw in keywords.split("、") if kw.strip()]
    cache[query] = predicted
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    return predicted

# SOCキーワードでフィルタ
def filter_by_predicted_soc(df, keywords):
    mask = df[["HLT", "HLGT", "SOC"]].apply(
        lambda col: col.astype(str).str.contains("|".join(keywords)), axis=1
    )
    return df[mask].copy()