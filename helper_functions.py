
import numpy as np
import pandas as pd
import faiss
import openai
import os
import pickle
from sentence_transformers import SentenceTransformer

# 環境変数からOpenAI APIキーを取得
openai.api_key = os.getenv("OPENAI_API_KEY")

# ベクトル化モデル
encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# FAISSとデータの読み込み
faiss_index = faiss.read_index("faiss_index.index")
meddra_terms = np.load("meddra_terms.npy", allow_pickle=True)
synonym_df = pd.read_pickle("synonym_df_cat1.pkl")

# term_master_df は app.py 側で読み込み

def encode_query(text):
    return encoder.encode(text)

def expand_query_gpt(query):
    cache_path = "query_expansion_cache.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    if query in cache:
        return cache[query]

    messages = [
        {"role": "system", "content": "あなたは医療用語の検索支援AIです。"},
        {"role": "user", "content": f"「{query}」という症状に関連する代表的な医学用語を3〜5個、日本語で挙げてください。"}
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        terms = response.choices[0].message.content.strip().split("、")
        terms = [term.strip(" ・
") for term in terms if term.strip()]
    except Exception:
        terms = []

    cache[query] = terms
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    return terms

def rerank_results_v13(query, candidates):
    cache_path = "score_cache.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    scored = []
    for term, source in candidates:
        key = (query, term)
        if key in cache:
            score = cache[key]
        else:
            prompt = f"以下の症状にどれだけ関連があるかを100点満点で評価してください。

症状: {query}
用語: {term}"
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                score = float(response.choices[0].message.content.strip().split()[0])
            except Exception:
                score = 0.0
            cache[key] = score
        scored.append((term, score, source))

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    return sorted(scored, key=lambda x: x[1], reverse=True)

def predict_soc_keywords_with_gpt(query):
    messages = [
        {"role": "system", "content": "あなたは医療用語の検索支援AIです。"},
        {"role": "user", "content": f"「{query}」という症状に関連しそうなSOCカテゴリ名を2〜4語、日本語で挙げてください（例：神経系障害、皮膚および皮下組織障害など）。"}
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return [kw.strip() for kw in response.choices[0].message.content.strip().split("、")]
    except Exception:
        return []

def filter_by_predicted_soc(df, soc_keywords):
    if not soc_keywords:
        return df
    mask = df[["SOC", "HLGT", "HLT"]].apply(
        lambda cols: cols.astype(str).str.contains("|".join(soc_keywords)).any(), axis=1
    )
    return df[mask].copy()

def rescale_scores(df, col="score"):
    min_val, max_val = df[col].min(), df[col].max()
    if max_val > min_val:
        df["確からしさ（％）"] = ((df[col] - min_val) / (max_val - min_val)) * 100
    else:
        df["確からしさ（％）"] = 100
    df["確からしさ（％）"] = df["確からしさ（％）"].round(1)
    return df

def add_hierarchy_info(results, term_master_df):
    df = pd.DataFrame(results, columns=["term", "score", "source"])
    df = df.merge(term_master_df, left_on="term", right_on="PT_English", how="left")
    return df

def load_term_master_df(path):
    return pd.read_pickle(path)

def search_meddra(query, top_k=10):
    results = []
    expanded = expand_query_gpt(query)
    all_terms = [query] + synonym_df.get(query, {}).get("synonyms", []) + expanded
    seen = set()

    for term in all_terms:
        if term not in seen:
            seen.add(term)
            vec = encode_query(term)
            D, I = faiss_index.search(np.array([vec]), top_k)
            for idx in I[0]:
                results.append((meddra_terms[idx], "faiss"))
    return results
