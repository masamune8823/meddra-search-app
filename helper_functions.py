
import numpy as np
import pandas as pd
import faiss
import openai
import torch
from sentence_transformers import SentenceTransformer

import os
import re
import pickle

# OpenAI APIキー（Streamlit CloudではSecretsで管理推奨）
openai.api_key = os.getenv("OPENAI_API_KEY", "sk-xxx")

# モデルの読み込み（MiniLM）
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# クエリのベクトル化
def encode_query(text):
    return model.encode([text])[0]

# FAISS検索処理（synonym_df対応）
def search_meddra(query, faiss_index, meddra_terms, synonym_df, top_k=20):
    query_vector = encode_query(query).astype("float32")
    scores, indices = faiss_index.search(np.array([query_vector]), top_k)
    terms = [meddra_terms[idx] for idx in indices[0]]
    df = pd.DataFrame({"term": terms, "score": scores[0]})
    
    # synonym_dfが存在すれば一致するsynonymも追加
    if synonym_df is not None and query in synonym_df["keyword"].values:
        matched_terms = synonym_df[synonym_df["keyword"] == query]["term"].tolist()
        synonym_df_rows = pd.DataFrame({"term": matched_terms, "score": [0.85] * len(matched_terms)})
        df = pd.concat([df, synonym_df_rows], ignore_index=True)
    
    df = df.drop_duplicates(subset=["term"]).reset_index(drop=True)
    return df

# GPT再スコアリング（Top N件に限定）
def rerank_results_v13(df, query, top_n=10, cache=None):
    if df.empty:
        return df

    df = df.head(top_n).copy()
    results = []

    for _, row in df.iterrows():
        term = row["term"]
        key = (query, term)
        if cache and key in cache:
            score = cache[key]
        else:
            prompt = f"ユーザーが「{query}」と訴えています。これが「{term}」という医学用語とどれくらい一致するか、100点満点で数値化してください。"
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "あなたは医療用語に精通したアシスタントです。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            score_text = response.choices[0].message.content.strip()
            try:
                score = float(re.findall(r"\d+", score_text)[0])
                score = min(max(score, 0), 100)
            except:
                score = 50.0  # fallback
            if cache is not None:
                cache[key] = score
        results.append(score)

    df["確からしさ（％）"] = results
    df = df.sort_values(by="確からしさ（％）", ascending=False).reset_index(drop=True)
    return df

# MedDRA階層情報の付加（term_master_df を使用）
def add_hierarchy_info(df, term_master_df):
    if df.empty:
        return df

    df["term_normalized"] = df["term"].str.lower().str.strip()
    term_master_df["term_normalized"] = term_master_df["PT_English"].str.lower().str.strip()
    return df.merge(term_master_df, on="term_normalized", how="left")

# GPTによるSOCカテゴリ推定
def predict_soc_category(query):
    prompt = f"次の症状に最も関連の深いMedDRAのSOC（System Organ Class）を一つだけ日本語で答えてください：「{query}」"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "あなたはMedDRA分類に詳しい医療専門家です。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()
