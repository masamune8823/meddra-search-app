import os
import re
import pickle
import numpy as np
import pandas as pd
import faiss
import openai
import torch
from sentence_transformers import SentenceTransformer

# モデルのロード（検索用エンコーダー）
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# OpenAI設定（環境変数 OPENAI_API_KEY を使用）
openai.api_key = os.getenv("OPENAI_API_KEY", "sk-xxx")  # Streamlit CloudではSecretsで管理推奨

# クエリのベクトル化
def encode_query(text):
    return model.encode([text])[0]

# FAISS検索
def search_meddra(query, faiss_index, meddra_terms, synonym_df, top_k=20):
    query_vec = encode_query(query)

    # 類義語マッチ（カテゴリ1のみ）
    match = synonym_df[synonym_df["表現"].str.strip() == query.strip()]
    synonym_terms = match["PT_Name"].unique().tolist() if not match.empty else []

    # FAISS類似検索
    D, I = faiss_index.search(np.array([query_vec]), top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        term = meddra_terms[idx]
        results.append({"term": term, "score": float(score)})

    # 類義語一致分はスコアを上書きして追加
    for term in synonym_terms:
        results.append({"term": term, "score": 999.0})

    return pd.DataFrame(results)

# 再ランキング（GPTスコアリング）
def rerank_results_v13(query, results_df, top_n=10):
    scored_results = []
    for _, row in results_df.head(top_n).iterrows():
        candidate = row["term"]
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "あなたは医療用語の専門家です。"},
                    {"role": "user", "content": f"以下の症状に最も適切な用語を評価してください。
症状: {query}
候補: {candidate}
この候補はどれほど適切ですか？10点満点で数値のみ返してください。"}
                ],
                temperature=0,
                max_tokens=10,
            )
            score_text = response["choices"][0]["message"]["content"].strip()
            score = float(score_text)
        except Exception as e:
            print(f"再スコアリング失敗: {candidate} -> {e}")
            score = 0.0
        scored_results.append({**row, "gpt_score": score})
    return pd.DataFrame(scored_results)

# MedDRA階層の付与
def add_hierarchy_info(df, term_master_df):
    df["term_normalized"] = df["term"].str.lower().str.strip()
    term_master_df["term_normalized"] = term_master_df["PT_English"].str.lower().str.strip()
    return df.merge(term_master_df, on="term_normalized", how="left")

# スコアを0〜100%に変換
def rescale_scores(df):
    if "gpt_score" not in df.columns:
        return df
    min_score = df["gpt_score"].min()
    max_score = df["gpt_score"].max()
    if max_score == min_score:
        df["確からしさ（％）"] = 100
    else:
        df["確からしさ（％）"] = ((df["gpt_score"] - min_score) / (max_score - min_score) * 100).round(1)
    return df

# OpenAIによるSOCカテゴリ予測
def predict_soc_category(query: str) -> list:
    system_prompt = (
        "あなたは医療分野に詳しいAIです。入力された症状の説明から、"
        "該当する可能性の高いMedDRAのSOC（System Organ Class）を、日本語で1〜3個、リスト形式で抽出してください。"
        "抽出の際は、「神経系障害」「胃腸障害」「皮膚および皮下組織障害」などの正式な表記を使用してください。"
    )
    user_prompt = f"次の症状に該当するSOCを日本語で挙げてください：『{query}』"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=100,
        )
        text = response["choices"][0]["message"]["content"]
        categories = [line.strip("・-・●-＊ ").strip() for line in text.strip().splitlines() if line.strip()]
        return categories if categories else ["該当なし"]
    except Exception as e:
        print("⚠️ GPTによるSOCカテゴリ推論エラー:", e)
        return ["該当なし"]