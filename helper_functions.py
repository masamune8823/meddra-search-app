import os
import re
import pickle
import numpy as np
import pandas as pd
import faiss
import openai
import torch

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler

# モデルの初期化（MiniLM）
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# OpenAI APIキー（環境変数から）
openai.api_key = os.getenv("OPENAI_API_KEY", "sk-xxx")  # 本番環境ではSecrets使用

# クエリのエンコード（ベクトル化）
def encode_query(text):
    return model.encode([text])[0]

# クエリ拡張（OpenAI GPT）
def expand_query_gpt(query):
    prompt = f"以下の日本語の症状から、英語の医学的キーワードを3つ予測してください（カンマ区切り）:\n「{query}」"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    keywords = response["choices"][0]["message"]["content"].strip()
    return [kw.strip() for kw in keywords.split(",")]

# 検索関数
def search_meddra(query, faiss_index, meddra_terms, synonym_df, top_k=20):
    query_vec = encode_query(query).astype("float32")
    _, indices = faiss_index.search(np.array([query_vec]), top_k)
    results = pd.DataFrame(meddra_terms[indices[0]], columns=["term"])
    results["query"] = query
    return results

# スコア再ランキング（GPTを用いた確からしさ予測）
def rerank_results_v13(query, candidates, max_items=10):
    top_terms = candidates["term"].tolist()[:max_items]
    messages = [
        {"role": "system", "content": "医療用語の関連性をスコアで評価してください（0〜100）"},
        {"role": "user", "content": f"症状: {query}\n候補: {', '.join(top_terms)}"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.3
    )
    content = response["choices"][0]["message"]["content"]
    scores = [int(s) for s in re.findall(r"\d+", content)][:len(top_terms)]
    candidates = candidates.copy()
    candidates["score"] = scores + [0] * (len(candidates) - len(scores))
    return candidates

# スコアを0〜100に正規化
def rescale_scores(df, score_col="score"):
    scaler = MinMaxScaler(feature_range=(0, 100))
    df[score_col] = scaler.fit_transform(df[[score_col]])
    return df

# 階層情報の追加（term_master_dfからPT名を基にマージ）
def add_hierarchy_info(df, term_master_df):
    return pd.merge(df, term_master_df, how="left", left_on="term", right_on="PT_English")

# SOCカテゴリの推定（OpenAI API）
def predict_soc_category(query):
    prompt = f"この症状「{query}」に最も関連するSOCカテゴリ（MedDRAの大分類）を日本語で1つだけ返答してください。"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response["choices"][0]["message"]["content"].strip()
# UPdate