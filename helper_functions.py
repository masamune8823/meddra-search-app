
import openai
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import pandas as pd
import os

# クエリ拡張（GPTによる）
def expand_query_gpt(query):
    prompt = (
        f"以下は医療症状に関する言葉です。\n"
        f"ユーザーの訴え：{query}\n"
        f"これに関連する代表的なキーワードを3つ、短く簡潔に日本語で挙げてください。\n"
        f"例：\n"
        f"訴え：皮膚がかゆい → かゆみ, 発疹, アレルギー\n"
        f"訴え：頭が痛い → 頭痛, ズキズキ, 偏頭痛\n"
        f"訴え：{query} →"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "医療用語に詳しいアシスタントです。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=100,
        )
        keywords = response["choices"][0]["message"]["content"]
        return keywords
    except Exception as e:
        return f"[ERROR] {e}"

# クエリのベクトル化
def encode_query(query, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    encoder = SentenceTransformer(model_name)
    return encoder.encode([query])[0]

# FAISS検索
def search_meddra(query_vector, index, terms, top_k=10):
    D, I = index.search(np.array([query_vector]), top_k)
    return [(terms[i], float(D[0][idx])) for idx, i in enumerate(I[0])]

# MedDRA階層の付加（仮の簡易版）
def add_hierarchy_info(result_terms, term_master_df):
    rows = []
    for term, score in result_terms:
        match = term_master_df[term_master_df["PT_English"] == term]
        if not match.empty:
            row = match.iloc[0].to_dict()
            row["Score"] = score
            rows.append(row)
        else:
            rows.append({"PT_English": term, "Score": score})
    return pd.DataFrame(rows)

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

    prompt = (
        f"以下は医療症状に関する言葉です。\n"
        f"ユーザーの訴え：{query}\n"
        f"候補語：{[x[0] for x in results]}\n"
        f"この中から、訴えとの関連性が高いものを優先して、スコア付きで上位{top_n}件を挙げてください。\n"
        f"形式：候補語：スコア（0～100）"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "医療用語のスコアリングを行うアシスタントです。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=300,
        )
        result = response["choices"][0]["message"]["content"]
        cache[key] = result
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        return result
    except Exception as e:
        return f"[ERROR] {e}"
