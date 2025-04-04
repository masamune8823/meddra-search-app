
import os
import re
import pickle
import openai
import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# モデルロード
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# OpenAI APIキー（環境変数から取得）
openai.api_key = os.getenv("OPENAI_API_KEY")

# クエリをベクトル化
def encode_query(text):
    return model.encode([text])[0]

# 検索処理本体
def search_meddra(query, faiss_index, meddra_terms, synonym_df=None, top_k=10):
    query_vector = encode_query(query).astype(np.float32)
    distances, indices = faiss_index.search(np.array([query_vector]), top_k)
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        if idx < len(meddra_terms):
            term = meddra_terms[idx]
            score = float(distances[0][i])
            results.append({"term": term, "score": score})
    return pd.DataFrame(results)

# 回答からスコア抽出（単純実装）
def extract_score_from_response(response_text):
    for word in ["10", "９", "8", "７", "6", "5", "4", "3", "2", "1", "0"]:
        if word in response_text:
            try:
                return float(word)
            except:
                continue
    return 5.0  # fallback

# スコアの再スケーリング
def rescale_scores(scores):
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [100.0 for _ in scores]
    return [100.0 * (s - min_score) / (max_score - min_score) for s in scores]

# 再ランキング処理（GPT使用）
def rerank_results_v13(query, candidates, score_cache=None):
    if score_cache is None:
        score_cache = {}

    scored = []
    for i, row in candidates.iterrows():
        term = row["term"]
        cache_key = (query, term)
        if cache_key in score_cache:
            score = score_cache[cache_key]
        else:
            messages = [
                {"role": "system", "content": "あなたは医療用語の関連性判定モデルです。"},
                {"role": "user", "content": f"以下の記述は、用語「{term}」とどのくらい意味的に一致しますか？ 記述: {query}"}
            ]
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0,
                )
                score = extract_score_from_response(response["choices"][0]["message"]["content"])
            except Exception as e:
                score = 5.0  # Fallback
            score_cache[cache_key] = score
        scored.append((term, score))

    df = pd.DataFrame(scored, columns=["term", "Relevance"])
    return df.sort_values(by="Relevance", ascending=False)

# GPTでSOCカテゴリを予測
def predict_soc_category(query):
    messages = [
        {"role": "system", "content": "あなたは医療分野に詳しいアシスタントです。"},
        {"role": "user", "content": f"次の症状に最も関連するMedDRAのSOCカテゴリを教えてください:「{query}」"}
    ]
    try:
        # 修正後（OpenAI v1対応の正しい使い方）
        from openai import OpenAI
        client = OpenAI()

        # その後は以下のように使ってOKです
        response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return "エラー: " + str(e)

# クエリ拡張（GPT使用）
def expand_query_gpt(query):
    messages = [
        {"role": "system", "content": "あなたは日本語医療文を英語のキーワードに変換するアシスタントです。"},
        {"role": "user", "content": f"以下の日本語の症状から、英語の医学的キーワードを3つ予測してください。\n\n症状: {query}"}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
        )
        text = response["choices"][0]["message"]["content"]
        return [kw.strip() for kw in text.split(",") if kw.strip()]
    except Exception as e:
        return ["headache", "nausea", "fever"]

# 表示整形（キーワードリスト）
def format_keywords(keywords):
    return "、".join(keywords)

# MedDRA階層情報を付与
def add_hierarchy_info(df, term_master_df):
    merged = pd.merge(df, term_master_df, how="left", left_on="term", right_on="PT_English")
    return merged
# update