# helper_functions.py

import openai
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os

# ✅ OpenAI APIキーを環境変数から取得
openai.api_key = os.environ.get("OPENAI_API_KEY")

# ✅ ベクトル用モデル（FAISS互換）
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# ✅ クエリ拡張（OpenAI GPT）
def expand_query_gpt(user_query):
    try:
        prompt = f"以下の医療用語に関連する別の表現を3つ挙げてください：{user_query}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        expansions = response["choices"][0]["message"]["content"].strip().split("\n")
        expansions = [e.replace("\u30fb", "").strip(" -0123456789.") for e in expansions if e]
        return [user_query] + expansions
    except Exception as e:
        return [user_query]

# ✅ クエリのベクトル化（次元384を維持）
def encode_query(query):
    embedding = model.encode(query)
    return np.array(embedding, dtype=np.float32)

# ✅ スコアによる再ランキング（Top10）
def rerank_results_v13(df):
    if "score" not in df.columns:
        return df
    return df.sort_values(by="score", ascending=False).head(10).reset_index(drop=True)

# ✅ シノニム辞書とのマッチング
def match_synonyms(user_query_list, synonym_df):
    matched_rows = synonym_df[synonym_df["synonym"].isin(user_query_list)].copy()
    matched_rows["score"] = 100.0
    matched_rows["source"] = "Synonym"
    return matched_rows

# ✅ FAISS結果とシノニム結果の統合
def merge_faiss_and_synonym_results(faiss_df, synonym_df):
    if faiss_df is None or faiss_df.empty:
        return synonym_df
    if synonym_df is None or synonym_df.empty:
        return faiss_df
    merged_df = pd.concat([faiss_df, synonym_df], ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=["PT_Japanese", "HLT_Japanese"], keep="first")
    return merged_df.reset_index(drop=True)
