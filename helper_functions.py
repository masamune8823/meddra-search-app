import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import openai
from config import (
    faiss_index,
    meddra_embeddings,
    meddra_terms,
    query_expansion_cache_path,
    score_cache_path,
    openai_api_key
)

openai.api_key = openai_api_key

# クエリ拡張（GPT利用）
def expand_query_gpt(user_query):
    if os.path.exists(query_expansion_cache_path):
        with open(query_expansion_cache_path, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    if user_query in cache:
        return cache[user_query]

    prompt = f"以下は医療症状に関する言葉です。
ユーザーの訴え：{user_query}
関連するキーワードを5つ挙げてください。"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
    )

    keywords = response.choices[0].message.content.strip().split("、")
    keywords = [kw.strip() for kw in keywords if kw.strip()]
    cache[user_query] = keywords

    with open(query_expansion_cache_path, "wb") as f:
        pickle.dump(cache, f)

    return keywords

# クエリベクトル化
def encode_query(query, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(query)

# FAISS検索
def search_faiss(query_vector, index, terms, top_k=10):
    D, I = index.search(np.array([query_vector]), top_k)
    return [(terms[i], float(D[0][idx])) for idx, i in enumerate(I[0])]

# MedDRA検索（拡張語すべてに対して検索し、重複を除く）
def search_meddra(query, top_k_per_method=5):
    expanded_keywords = expand_query_gpt(query)
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    all_results = []
    seen_terms = set()

    for keyword in expanded_keywords:
        vector = model.encode(keyword)
        results = search_faiss(vector, faiss_index, meddra_terms, top_k=top_k_per_method)

        for term, score in results:
            if term not in seen_terms:
                all_results.append((term, score))
                seen_terms.add(term)

    return all_results

# 再スコアリング（キャッシュ対応）
def rerank_results_v13(results, query, top_n=10, cache_path=score_cache_path):
    key = (query, tuple(results))
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    if key in cache:
        return cache[key]

    # GPT再スコアリング（疑似）
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)[:top_n]
    cache[key] = sorted_results

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    return sorted_results

# 階層情報追加（PTに一致するterm_master_dfから追加）
def add_hierarchy_info(results, term_master_df):
    enriched_results = []
    for term, score in results:
        row = term_master_df[term_master_df["PT_Japanese"] == term]
        if not row.empty:
            row = row.iloc[0]
            enriched_results.append([
                term,
                score,
                row.get("HLT_Japanese", ""),
                row.get("HLGT_Japanese", ""),
                row.get("SOC_Japanese", ""),
                "FAISS + GPT"
            ])
        else:
            enriched_results.append([term, score, "", "", "", "FAISS + GPT"])
    return enriched_results

# term_master_df 読み込み（Streamlit側で使用）
def load_term_master_df(path):
    return pd.read_pickle(path)