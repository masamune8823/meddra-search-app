
import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from config import faiss_index, meddra_terms

# エンコーダ（MiniLMなど）
encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# クエリベクトル化
def encode_query(query):
    return encoder.encode(query)

# GPTによるクエリ拡張（キャッシュ付き）
def expand_query_gpt(query, cache_path="query_expansion_cache.pkl"):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    if query in cache:
        return cache[query]

    import openai
    prompt = (
        f"以下は医療症状に関する言葉です。
"
        f"ユーザーの訴え：{query}
"
        "この訴えに関連するキーワードを日本語で5つ挙げてください。"
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "あなたは医療に詳しいAIです。"},
            {"role": "user", "content": prompt},
        ],
    )
    keywords = response["choices"][0]["message"]["content"].strip().splitlines()
    keywords = [kw.strip("・- 0123456789.") for kw in keywords if kw.strip()]
    cache[query] = keywords

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    return keywords

# FAISS検索（単体）
def search_faiss_single(query_vector, index, terms, top_k=10):
    D, I = index.search(np.array([query_vector]), top_k)
    return [(terms[i], float(D[0][idx])) for idx, i in enumerate(I[0])]

# FAISS検索（拡張語すべて）
def search_meddra(query, top_k_per_method=5):
    query_vector = encode_query(query)
    original_results = search_faiss_single(query_vector, faiss_index, meddra_terms, top_k=top_k_per_method)

    keywords = expand_query_gpt(query)
    expanded_results = []
    for kw in keywords:
        vec = encode_query(kw)
        results = search_faiss_single(vec, faiss_index, meddra_terms, top_k=top_k_per_method)
        expanded_results.extend(results)

    all_results = original_results + expanded_results
    df = pd.DataFrame(all_results, columns=["term", "score"]).drop_duplicates(subset="term")
    return df.sort_values("score", ascending=False)

# 再スコアリング（CrossEncoder＋キャッシュ）
def rerank_results_v13(results, query, top_n=10, cache_path="score_cache.pkl"):
    key = (query, tuple(results["term"]))
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    if key in cache:
        return cache[key]

    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(query, t) for t in results["term"][:top_n]]
    scores = model.predict(pairs)
    reranked = results.copy().head(top_n)
    reranked["score"] = [round(float(s) * 10, 2) for s in scores]
    reranked = reranked.sort_values("score", ascending=False)

    cache[key] = reranked
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    return reranked

# 階層情報追加（ダミー）
def add_hierarchy_info(results, term_master_df):
    return results

# term_master_df 読み込み
def load_term_master_df(path):
    return pd.read_pickle(path)
