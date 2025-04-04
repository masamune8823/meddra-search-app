import numpy as np
import pandas as pd
import faiss
import openai
import os
import re
import torch
from sentence_transformers import SentenceTransformer

# --- SentenceTransformer モデルロード（日本語対応） ---
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# OpenAI APIキーの取得（環境変数 or secrets）
openai.api_key = os.getenv("OPENAI_API_KEY", "sk-xxx")

# --- クエリのエンコード ---
def encode_query(text):
    return model.encode([text])[0]

# --- 類似検索 ---
def search_meddra(query, faiss_index, meddra_terms, synonym_df=None, top_k=20):
    query_vec = encode_query(query)
    D, I = faiss_index.search(np.array([query_vec]), top_k)
    results = pd.DataFrame({
        "term": [meddra_terms[i] for i in I[0]],
        "score": D[0]
    })

    # synonym_dfが指定されていれば、synonym候補も追加検索
    if synonym_df is not None:
        norm_query = query.strip().lower()
        matched = synonym_df[synonym_df["synonym"].str.lower() == norm_query]
        if not matched.empty:
            for pt in matched["PT"].unique():
                pt_vec = encode_query(pt)
                D_syn, I_syn = faiss_index.search(np.array([pt_vec]), 1)
                results = pd.concat([
                    results,
                    pd.DataFrame({"term": [pt], "score": D_syn[0]})
                ], ignore_index=True)

    results.drop_duplicates(subset="term", inplace=True)
    return results

# --- GPTによる再スコアリング（Top-N） ---
def rerank_results_v13(query, results_df, top_n=10):
    top_results = results_df.head(top_n).copy()
    prompts = [
        f"Q: {query}\nA: {cand}" for cand in top_results["term"]
    ]
    joined_prompt = "\n\n".join(prompts)

    messages = [
        {"role": "system", "content": "以下は医療用語検索の関連性評価です。Qに対してAがどれほど適切か10点満点で評価してください。"},
        {"role": "user", "content": joined_prompt}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=300,
        temperature=0
    )

    scores_text = response["choices"][0]["message"]["content"].strip()
    scores = re.findall(r"(\d+(?:\.\d+)?)", scores_text)

    if len(scores) == len(top_results):
        top_results["score"] = list(map(float, scores))
    else:
        top_results["score"] = [7.0] * len(top_results)  # fallback

    return top_results.sort_values(by="score", ascending=False)

# --- スコアのリスケーリング（0〜100％） ---
def rescale_scores(df):
    min_score = df["score"].min()
    max_score = df["score"].max()
    if max_score == min_score:
        df["score"] = 100.0
    else:
        df["score"] = 100 * (df["score"] - min_score) / (max_score - min_score)
    df["score"] = df["score"].round(1)
    return df

# --- MedDRA階層情報の付加 ---
def add_hierarchy_info(df, term_master_df):
    df = df.copy()
    df["term_normalized"] = df["term"].str.lower().str.strip()
    term_master_df = term_master_df.copy()
    term_master_df["term_normalized"] = term_master_df["PT_English"].str.lower().str.strip()
    merged = pd.merge(df, term_master_df, on="term_normalized", how="left")
    return merged

# --- GPTによるSOCカテゴリ予測（フィルタリング用） ---
def predict_soc_category(query):
    prompt = f"次の症状に関連するSOCカテゴリ（例：Gastrointestinal disorders）を1つだけ出力してください。\n\n症状: {query}\n\nSOCカテゴリ:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=30,
        temperature=0
    )
    return response["choices"][0]["message"]["content"].strip()
