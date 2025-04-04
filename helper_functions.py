import numpy as np
import pandas as pd
import faiss
import openai
import torch
from sentence_transformers import SentenceTransformer

# モデルロード（日本語対応のマルチリンガルMiniLM）
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# OpenAI設定（環境変数 OPENAI_API_KEY を使用）
openai.api_key = os.getenv("OPENAI_API_KEY", "sk-xxx")  # Streamlit CloudではSecretsで管理推奨

# クエリのベクトル化
def encode_query(text):
    return model.encode([text])[0]

# FAISS検索（+日本語synonym_dfによる補完あり）
def search_meddra(query, faiss_index, meddra_terms, synonym_df, top_k=20):
    # synonym展開（正規化）
    query_normalized = query.strip().lower()
    matched = synonym_df[synonym_df["synonym_normalized"] == query_normalized]
    terms = matched["PT_English"].tolist() if not matched.empty else [query]

    # クエリのベクトルを取得し、FAISS検索
    results = []
    for term in terms:
        vec = encode_query(term)
        D, I = faiss_index.search(np.array([vec]), top_k)
        for score, idx in zip(D[0], I[0]):
            results.append({
                "term": meddra_terms[idx],
                "score": float(score)
            })
    return pd.DataFrame(results)

# GPTによるスコア再評価（TopN）
def rerank_results_v13(query, results_df, top_n=10):
    if results_df.empty:
        return results_df
    candidates = results_df.head(top_n).copy()

    # GPT用プロンプト生成
    def make_prompt(term):
        return f"次の医療症状に最も関係する用語かを0〜100点で評価してください。\n\n症状: {query}\n候補: {term}\n\n点数:"

    prompts = [make_prompt(t) for t in candidates["term"]]

    try:
        responses = []
        for prompt in prompts:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=10,
            )
            score_str = response["choices"][0]["message"]["content"]
            score = int("".join(filter(str.isdigit, score_str)))
            responses.append(min(score, 100))
    except Exception as e:
        responses = [0] * len(candidates)
    candidates["score"] = responses
    return candidates

# MedDRA階層マスタの付加（正規化あり）
def add_hierarchy_info(df, term_master_df):
    df["term_normalized"] = df["term"].str.lower().str.strip()
    term_master_df["term_normalized"] = term_master_df["PT_English"].str.lower().str.strip()
    return df.merge(term_master_df, on="term_normalized", how="left")

# スコア正規化（0〜100%）
def rescale_scores(df):
    if "score" in df.columns and not df["score"].isnull().all():
        max_score = df["score"].max()
        min_score = df["score"].min()
        if max_score != min_score:
            df["score"] = (df["score"] - min_score) / (max_score - min_score) * 100
        else:
            df["score"] = 100
    return df

# GPTで症状に関連するSOCカテゴリを予測（例：「神経系障害」など）
def predict_soc_keywords_with_gpt(query):
    prompt = f"""
次の医療症状に関連するMedDRAのSOCカテゴリを日本語で1〜2個、簡潔に教えてください。
出力形式はプレーンテキストのキーワードのみで（例: 神経系障害, 消化器障害）。

症状: 「{query}」
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=50,
        )
        content = response["choices"][0]["message"]["content"]
        keywords = [k.strip() for k in content.replace("、", ",").replace("・", ",").split(",")]
        return [k for k in keywords if k]
    except Exception as e:
        return []
