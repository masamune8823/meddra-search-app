import numpy as np
import pandas as pd
import faiss
import openai
import pickle
from sentence_transformers import SentenceTransformer

# クエリベクトル生成（MiniLM）
encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def encode_query(query: str):
    return encoder.encode(query)

# 🔍 MedDRA検索（synonym_df + FAISS）
def search_meddra(query, faiss_index, meddra_terms, synonym_df=None, top_k=20):
    results = []

    # メインクエリで検索
    query_vec = encode_query(query)
    D, I = faiss_index.search(np.array([query_vec]), top_k)
    for i, score in zip(I[0], D[0]):
        results.append({
            "term": meddra_terms[i],
            "score": float(score),
            "source": "main"
        })

    # シノニムがあれば追加検索
    if synonym_df is not None and query in synonym_df.index:
        synonyms = synonym_df.loc[query]["synonyms"]
        for syn in synonyms:
            syn_vec = encode_query(syn)
            D_syn, I_syn = faiss_index.search(np.array([syn_vec]), top_k)
            for i, score in zip(I_syn[0], D_syn[0]):
                results.append({
                    "term": meddra_terms[i],
                    "score": float(score),
                    "source": f"syn:{syn}"
                })

    return pd.DataFrame(results).drop_duplicates(subset="term").reset_index(drop=True)

# 🎯 GPT再スコアリング
def rerank_results_v13(query, df, top_n=10):
    if df.empty:
        return df

    terms = df["term"].tolist()[:top_n]
    prompt = (
        f"以下の症状にどれだけ関連があるかを100点満点で評価してください：{query}\n"
        "各用語ごとに「スコアのみ」を出力してください。"
    )

    messages = [{"role": "user", "content": prompt + "\n" + "\n".join(terms)}]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
        )
        reply = response["choices"][0]["message"]["content"]
        lines = reply.strip().split("\n")
        scores = {}
        for line in lines:
            for term in terms:
                if term in line:
                    num = "".join([c for c in line if c.isdigit()])
                    scores[term] = int(num) if num else 0

        df["score"] = df["term"].map(scores).fillna(0)
        df["score"] = df["score"].astype(int)
        return df.sort_values("score", ascending=False).reset_index(drop=True)

    except Exception as e:
        print(f"Error in reranking: {e}")
        df["score"] = 0
        return df

# 🎯 GPTによるSOCカテゴリ予測
def predict_soc_keywords_with_gpt(query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "あなたは医療分野に詳しいAIです。与えられた症状に関連するMedDRAのSOCカテゴリを3つ、日本語で簡潔に予測してください。"},
            {"role": "user", "content": f"以下の症状に関連するSOCカテゴリを教えてください：\n\n{query}"}
        ],
        temperature=0.3,
    )
    text = response["choices"][0]["message"]["content"]
    keywords = [kw.strip("・ 、。\n") for kw in text.split() if kw.strip()]
    return keywords[:3]

# 🎯 SOCキーワードによるフィルタリング
def filter_by_predicted_soc(df, keywords):
    if not keywords:
        return df
    cols = ["SOC", "HLGT", "HLT"]
    cols = [col for col in cols if col in df.columns]
    if not cols:
        return df
    mask = df[cols].apply(lambda x: x.astype(str).str.contains("|".join(keywords)).any(), axis=1)
    return df[mask].copy()

# 🔧 スコアの再スケーリング（最大値100に）
def rescale_scores(df, col="score"):
    if col in df.columns and df[col].max() > 0:
        df[col] = (df[col] / df[col].max()) * 100
        df[col] = df[col].round(1)
    return df
