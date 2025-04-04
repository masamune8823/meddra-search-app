# ✅ ファイルの先頭に一度だけ書く
from openai import OpenAI
client = OpenAI()

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

# ✅ 再ランキング処理（GPT一括呼び出し版）
def rerank_results_batch(query, candidates, score_cache=None):
    if score_cache is None:
        score_cache = {}

    # Top10件に絞る
    top_candidates = candidates.head(10)

    messages = [{"role": "system", "content": "あなたは医療用語の関連性判定モデルです。"}]
    index_map = {}  # idxとtermの対応を記録

    for i, row in top_candidates.iterrows():
        term = row["term"]
        cache_key = (query, term)

        if cache_key in score_cache:
            continue  # スコア済み

        prompt = f"用語「{term}」は、以下の記述とどれくらい意味的に一致しますか？ 一致度（0～10）を数値で教えてください。\n記述: {query}"
        messages.append({"role": "user", "content": prompt})
        index_map[len(messages) - 2] = term  # systemを除いたindex

    # GPT呼び出し（1回）
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
        )
        
        # ✅ デバッグ用ログ（前後を含めて明示）
        import streamlit as st
        st.subheader("🧾 GPTレスポンス内容（デバッグ用）")
        st.write("🔍 GPTからの生レスポンス（全体構造）")
        st.write(response)  # ← response全体を表示（構造確認）

        st.write("📝 GPTからのメッセージ本文（choices[0].message.content）")
        st.code(response.choices[0].message.content)  # ← 実際の返答テキスト表示

        # 返答（1つ）から全体の内容を取得
        content = response.choices[0].message.content

        # 返答の中から個別に数値を抽出（改行 or カンマ区切り想定）
        lines = [l.strip() for l in content.strip().split("\n") if l.strip()]
        for i, line in enumerate(lines):
            st.write(f"[{i}] {line}")  # ← 🔍 どんな行かを可視化
            if i in index_map:
                term = index_map[i]
                try:
                    score = extract_score_from_response(line)
                    score_cache[(query, term)] = score
                except:
                    score_cache[(query, term)] = 5.0  # fallback
    except Exception as e:
        # 全体失敗時のfallback
        for term in top_candidates["term"]:
            score_cache[(query, term)] = 5.0

    # スコアをまとめて返す
    scored = [(term, score_cache.get((query, term), 5.0)) for term in top_candidates["term"]]
    df = pd.DataFrame(scored, columns=["term", "Relevance"])
    return df.sort_values(by="Relevance", ascending=False)

# GPTでSOCカテゴリを予測
def predict_soc_category(query):
    messages = [
        {"role": "system", "content": "あなたは医療分野に詳しいアシスタントです。"},
        {"role": "user", "content": f"次の症状に最も関連するMedDRAのSOCカテゴリを教えてください:「{query}」"}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content
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
        response_text = response.choices[0].message.content
        return [kw.strip() for kw in response_text.split(",") if kw.strip()]
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