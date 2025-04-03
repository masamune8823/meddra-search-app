
import os
import numpy as np
import pandas as pd
import openai
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# クエリをエンコード（MiniLMモデル使用）
def encode_query(query, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(query)

# GPTでクエリ拡張（OpenAI API v1対応）
def expand_query_gpt(query, model="gpt-3.5-turbo"):
    client = openai.OpenAI()
    prompt = (
        f"以下は医療症状に関する言葉です。\n"
        f"ユーザーの訴え：{query}\n"
        f"これに関連する代表的なキーワードを3つ、短く簡潔に日本語で挙げてください。\n"
        f"例：\n"
        f"訴え：皮膚がかゆい → かゆみ, 発疹, アレルギー\n"
        f"訴え：頭が痛い → 頭痛, ズキズキ, 偏頭痛\n"
        f"訴え：{query} →"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "あなたは優秀な医療用語アシスタントです。"},
            {"role": "user", "content": prompt}
        ]
    )
    output = response.choices[0].message.content.strip()
    return [kw.strip() for kw in output.replace("→", "").split(",") if kw.strip()]

# FAISS検索本体
def search_meddra_core(query_vector, index, terms, top_k=10):
    D, I = index.search(np.array([query_vector]), top_k)
    return [(terms[i], float(D[0][idx])) for idx, i in enumerate(I[0])]

# 検索ラッパー関数（キーワードごとにベクトル化＋検索＋集約）
def search_meddra(query, top_k_per_method=5):
    from config import faiss_index, meddra_terms
    keywords = expand_query_gpt(query)
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    results = []
    for kw in keywords:
        vec = model.encode(kw)
        hits = search_meddra_core(vec, faiss_index, meddra_terms, top_k=top_k_per_method)
        for term, score in hits:
            results.append((term, score, kw))

    return results

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

    # 簡易スコアリング（スコアを0-100でスケーリング）
    df = pd.DataFrame(results, columns=["term", "score", "source"])
    df["確からしさ（％）"] = (df["score"] - df["score"].min()) / (df["score"].max() - df["score"].min()) * 100
    df["確からしさ（％）"] = df["確からしさ（％）"].round(1)
    deduped = df.sort_values("score", ascending=False).drop_duplicates("term").head(top_n)

    output = [
        (row["term"], row["確からしさ（％）"], "", "", "", row["source"])
        for _, row in deduped.iterrows()
    ]

    cache[key] = output
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    return output

# ✅ 階層情報を term_master_df からマージして追加する関数（正式版）
def add_hierarchy_info(results, term_master_df):
    df = pd.DataFrame(results, columns=["term", "score", "source"])
    merged_df = df.merge(term_master_df, how="left", left_on="term", right_on="PT_Japanese")
    output = [
        (
            row["term"],
            row["score"],
            row.get("HLT_Japanese", ""),
            row.get("HLGT_Japanese", ""),
            row.get("SOC_Japanese", ""),
            row["source"]
        )
        for _, row in merged_df.iterrows()
    ]
    return output

# term_master_df 読み込み
def load_term_master_df(path):
    return pd.read_pickle(path)
