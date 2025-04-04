
import os
import re
import pickle
import numpy as np
import pandas as pd
import openai
import torch
from sentence_transformers import SentenceTransformer, util

# クエリをエンコードする
def encode_query(query, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(query, convert_to_tensor=True)

# クエリ拡張（OpenAI GPT-3.5使用）
def expand_query_gpt(text, api_key=None):
    openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
    prompt = f"以下の日本語の症状から、英語の医学的キーワードを3つ予測してください：{text}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        result = response["choices"][0]["message"]["content"]
        return [kw.strip() for kw in result.split(",") if kw.strip()]
    except Exception as e:
        print("OpenAI API error:", e)
        return []

# FAISS + 類義語辞書による検索
def search_meddra(query, faiss_index, terms, synonym_df, top_k=20):
    synonyms = synonym_df.get(query, [query])
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectors = model.encode(synonyms)
    all_scores = []
    for vector in vectors:
        D, I = faiss_index.search(np.array([vector]), top_k)
        for i, d in zip(I[0], D[0]):
            all_scores.append((i, float(d)))
    all_scores = sorted(all_scores, key=lambda x: x[1])
    seen = set()
    unique_results = []
    for idx, score in all_scores:
        if idx not in seen:
            unique_results.append({"term": terms[idx], "score": score})
            seen.add(idx)
    return pd.DataFrame(unique_results)

# スコア再スケーリング
def rescale_scores(scores):
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [100.0] * len(scores)
    return [100 * (s - min_score) / (max_score - min_score) for s in scores]

# スコア再ランキング（GPTではなくMiniLMで処理する想定）
def rerank_results_v13(query, candidates, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_name)
    pairs = [[query, c] for c in candidates]
    scores = model.predict(pairs)
    return scores

# 階層情報の付与
def add_hierarchy_info(df, hierarchy_df):
    return pd.merge(df, hierarchy_df, how="left", left_on="term", right_on="PT_English")

# 拡張語表示用フォーマット整形
def format_keywords(keywords):
    return ", ".join(keywords)

# SOCカテゴリ予測（仮のダミー実装）
def predict_soc_category(text):
    return "General disorders and administration site conditions"
