import numpy as np
import pandas as pd
import faiss
import openai
import torch
import os
import re

from sentence_transformers import SentenceTransformer

# モデル定義（MiniLMエンコーダ）
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# OpenAI設定（Streamlit Cloudではsecrets使用推奨）
openai.api_key = os.getenv("OPENAI_API_KEY", "sk-xxx")


# クエリをベクトルに変換
def encode_query(text):
    return model.encode([text])[0]


# スコアを0〜100の範囲に再スケーリング
def rescale_scores(df):
    if "score" not in df.columns:
        return df
    min_score = df["score"].min()
    max_score = df["score"].max()
    if min_score == max_score:
        df["score"] = 100
    else:
        df["score"] = (df["score"] - min_score) / (max_score - min_score) * 100
    return df


# MedDRA検索（ベクトル類似＋シノニム補完）
def search_meddra(query, faiss_index, meddra_terms, synonym_df=None, top_k=20):
    query_vector = encode_query(query)
    D, I = faiss_index.search(np.array([query_vector]), top_k)
    results = []
    for i, score in zip(I[0], D[0]):
        if 0 <= i < len(meddra_terms):
            results.append({
                "term": meddra_terms[i],
                "score": float(score)
            })

    df = pd.DataFrame(results)

    # synonym_df にマッチする語があれば追加（スコアは0とする）
    if synonym_df is not None and query in synonym_df["synonym"].values:
        matched_terms = synonym_df[synonym_df["synonym"] == query]["PT_English"].tolist()
        for term in matched_terms:
            df = pd.concat([df, pd.DataFrame([{"term": term, "score": 0}])], ignore_index=True)

    return df


# GPT再スコアリング
def rerank_results_v13(query, df, top_n=10):
    df_top = df.head(top_n).copy()
    if df_top.empty:
        return df_top
    prompts = [
        f"ユーザーの症状「{query}」は、医療用語「{row['term']}」とどの程度一致しますか？ 0から10の数字で評価してください。"
        for _, row in df_top.iterrows()
    ]
    try:
        responses = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "あなたは医療用語のマッチング評価AIです。"},
                {"role": "user", "content": "\n".join(prompts)}
            ],
            max_tokens=300,
            temperature=0,
        )
        text = responses["choices"][0]["message"]["content"]
        scores = re.findall(r"\d+", text)
        scores = [int(s) for s in scores[:len(df_top)]]
        df_top["score"] = scores
    except Exception as e:
        df_top["score"] = 0
    return df_top


# MedDRA階層（HLT/HLGT/SOC）を付加
def add_hierarchy_info(df, term_master_df=None):
    if term_master_df is None or df.empty:
        df["HLT_Japanese"] = None
        df["HLGT_Japanese"] = None
        df["SOC_Japanese"] = None
        return df

    df["term_normalized"] = df["term"].str.lower().str.strip()
    term_master_df["term_normalized"] = term_master_df["PT_English"].str.lower().str.strip()

    merged = pd.merge(df, term_master_df, on="term_normalized", how="left")

    # 欠損列があれば埋める
    for col in ["HLT_Japanese", "HLGT_Japanese", "SOC_Japanese"]:
        if col not in merged.columns:
            merged[col] = None
    return merged


# GPTによるSOC予測（キーワード生成）
def predict_soc_keywords_with_gpt(query):
    prompt = f"以下は医療症状です：「{query}」。関連しそうなMedDRAのSOCカテゴリ名（日本語）を1～3個、簡潔なキーワードで挙げてください。"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "あなたは医療用語の分類アシスタントです。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.7,
        )
        result = response["choices"][0]["message"]["content"]
        result_clean = re.findall(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]+", result)
        return result_clean
    except Exception as e:
        return []
# updated
