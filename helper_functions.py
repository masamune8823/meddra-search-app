
import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder

# モデル読み込み
encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# クエリをベクトル化
def encode_query(query):
    return encoder.encode(query)

# クエリ拡張（OpenAI APIによる）
def expand_query_gpt(query, cache_path="query_expansion_cache.pkl"):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    if query in cache:
        return cache[query]

    prompt = (
        f"以下は医療症状に関する言葉です。\n"
        f"ユーザーの訴え：{query}\n"
        f"上記の内容から、検索に適したキーワードを5つ提案してください。"
    )

    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    expanded = response.choices[0].message.content.strip().split("\n")
    keywords = [kw.strip("・ ") for kw in expanded if kw.strip()]
    cache[query] = keywords
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    return keywords

# FAISS検索をラップ（複数拡張語に対応）
def search_meddra(query, top_k_per_method=5):
    from config import faiss_index, meddra_terms
    keywords = expand_query_gpt(query)
    all_results = []
    for kw in keywords:
        q_vec = encode_query(kw)
        D, I = faiss_index.search(np.array([q_vec]), top_k_per_method)
        for idx, i in enumerate(I[0]):
            all_results.append((meddra_terms[i], float(D[0][idx])))
    return all_results

# 再スコアリング（キャッシュ対応）
def rerank_results_v13(results, query, top_n=10, cache_path="score_cache.pkl"):
    key = (query, tuple(results))
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    if key in cache:
        return cache[key]

    texts = [r[0] for r in results]
    pairs = [(query, t) for t in texts]
    scores = cross_encoder.predict(pairs)
    reranked = sorted(zip(texts, scores), key=lambda x: x[1], reverse=True)[:top_n]

    # スコアをパーセンテージに変換
    max_score = max(scores) if scores else 1
    final = [(term, round(score / max_score * 100, 1)) for term, score in reranked]
    cache[key] = final
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    return final

# 階層情報追加（ダミー：後ほど必要に応じて定義）
def add_hierarchy_info(results, term_master_df):
    return results

# term_master_df 読み込み（Streamlit側で使用）
def load_term_master_df(path):
    return pd.read_pickle(path)
