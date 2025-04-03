import numpy as np
import pandas as pd
import faiss
import openai
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler

# エンコーダーモデル（MiniLMベース）
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# クエリをエンコード（ベクトル化）
def encode_query(text):
    return model.encode([text])[0]

# 類似検索（synonym_dfを利用し、ベクトル検索を強化）
def search_meddra(query, faiss_index, meddra_terms, synonym_df=None, top_k=20):
    # synonym_df にあるならクエリを展開
    expanded_queries = [query]
    if synonym_df is not None and query in synonym_df.index:
        expanded_queries += synonym_df.loc[query]["synonym"].tolist()

    seen_terms = set()
    results = []

    for q in expanded_queries:
        q_vec = encode_query(q).astype("float32")
        D, I = faiss_index.search(np.array([q_vec]), top_k)
        for idx, dist in zip(I[0], D[0]):
            term = meddra_terms[idx]
            if term not in seen_terms:
                seen_terms.add(term)
                results.append({"term": term, "score": 1 - dist})  # scoreは近さを反映（小さいほど近い）

    return pd.DataFrame(results)

# 再スコアリング（GPTベース、top_n件に限定）
def rerank_results_v13(query, df, top_n=10):
    if df.empty:
        return df

    top_df = df.head(top_n).copy()
    system_prompt = "あなたは医療に詳しいAIアシスタントです。"

    def get_score(term):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"以下の症状「{query}」に対して、次の医療用語「{term}」の適合度を100点満点で評価してください。"}
                ],
                temperature=0,
            )
            content = response["choices"][0]["message"]["content"]
            score = int("".join(filter(str.isdigit, content)))
            return max(0, min(score, 100))
        except Exception as e:
            return 0

    top_df["score"] = top_df["term"].apply(get_score)
    return top_df.sort_values("score", ascending=False).reset_index(drop=True)

# MedDRA階層情報の付加（term_master_dfを使用）
def add_hierarchy_info(df, term_master_df):
    if df.empty:
        return df

    df["term_normalized"] = df["term"].str.lower().str.strip()
    term_master_df["term_normalized"] = term_master_df["PT_English"].str.lower().str.strip()

    merged = df.merge(term_master_df, on="term_normalized", how="left")
    return merged

# スコアを0〜100に再スケーリング
def rescale_scores(df):
    if "score" not in df.columns or df.empty:
        return df
    scaler = MinMaxScaler(feature_range=(0, 100))
    df["確からしさ（％）"] = scaler.fit_transform(df[["score"]]).round(0).astype(int)
    return df

# GPTによるSOCキーワード予測（任意）
def predict_soc_keywords_with_gpt(query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "あなたは医療用語に詳しいAIです。症状から関連するSOCカテゴリ（例：神経系障害、胃腸障害など）を1〜3個挙げてください。"},
                {"role": "user", "content": f"症状「{query}」に関連するSOCカテゴリを日本語で答えてください。"}
            ],
            temperature=0,
        )
        content = response["choices"][0]["message"]["content"]
        keywords = [kw.strip(" ・-・。0123456789\n") for kw in content.splitlines() if kw.strip()]
        return keywords
    except Exception:
        return []
