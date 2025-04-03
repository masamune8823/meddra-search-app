import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import openai
import os

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# モデルの読み込み（再利用）
encoder_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# ベクトル化
def encode_query(text):
    return encoder_model.encode(text)

# 再ランキング用のキャッシュ読み込み・保存
def load_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return {}

def save_cache(cache, cache_path):
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

# クエリ拡張（GPT使用）
def expand_query_gpt(query):
    prompt = (
        f"以下は医療症状に関する言葉です。\n"
        f"ユーザーの訴え：{query}\n"
        f"これに関連する代表的なキーワードを3つ、短く簡潔に日本語で挙げてください。\n"
        f"例：\n"
        f"訴え：皮膚がかゆい → かゆみ, 発疹, アレルギー\n"
        f"訴え：頭が痛い → 頭痛, スキズキ, 偏頭痛\n"
        f"訴え：{query} →"
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "あなたは医療用語の抽出アシスタントです。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=100,
    )

    output = response["choices"][0]["message"]["content"].strip()
    return [kw.strip() for kw in output.replace("→", "").split(",") if kw.strip()]

# FAISS検索本体
def search_meddra_core(query_vector, index, terms, top_k=10):
    D, I = index.search(np.array([query_vector]), top_k)
    return [(terms[i], float(D[0][idx])) for idx, i in enumerate(I[0])]

# Streamlit UI用ラッパー関数
def search_meddra(query, top_k_per_method=5):
    # 必要な変数はここで読み込む（Streamlit側でセット済の前提）
    from config import faiss_index, meddra_terms
    query_vector = encode_query(query)
    return search_meddra_core(query_vector, faiss_index, meddra_terms, top_k=top_k_per_method)

# 階層情報追加（ダミー：後ほど必要に応じて定義）
def add_hierarchy_info(results, term_master_df):
    return results  # 今はそのまま返す

# term_master_df 読み込み（Streamlit側で使用）
def load_term_master_df(path):
    import pandas as pd
    return pd.read_pickle(path)

# 再ランキング（省略・本体が別定義）
def rerank_results_v13(results, query, top_n=10, cache_path="score_cache.pkl"):
    key = (query, tuple(results))
    cache = load_cache(cache_path)
    if key in cache:
        return cache[key]

    # GPTスコア付け（仮）
    reranked = []
    for term, score in results:
        reranked.append((term, score + 1))  # 仮のスコア補正

    reranked = sorted(reranked, key=lambda x: -x[1])[:top_n]
    cache[key] = reranked
    save_cache(cache, cache_path)
    return reranked
