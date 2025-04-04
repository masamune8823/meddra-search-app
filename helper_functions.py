
import numpy as np
import pandas as pd
import torch
import openai
import re
import pickle
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def encode_query(query):
    return model.encode(query)

def search_meddra(query, faiss_index, meddra_terms, synonym_df, top_k=20):
    query_vector = encode_query(query).astype("float32")
    D, I = faiss_index.search(np.array([query_vector]), top_k)
    results = [{"term": str(meddra_terms[i]), "score": float(D[0][j])} for j, i in enumerate(I[0])]
    return pd.DataFrame(results)

def rerank_results_v13(original_query, df, score_cache):
    import openai
    results = []
    for _, row in df.iterrows():
        term = row["term"]
        cache_key = (original_query, term)
        if cache_key in score_cache:
            score = score_cache[cache_key]
        else:
            prompt = f"次の日本語の症状に対して、以下の用語がどれくらい適切か10点満点で評価してください：\n症状: {original_query}\n候補用語: {term}\nスコア:"
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            text = response["choices"][0]["message"]["content"]
            score = float(re.search(r"\d+", text).group())
            score_cache[cache_key] = score
        results.append(score)
    df["gpt_score"] = results
    df["score_percent"] = df["gpt_score"].apply(lambda x: rescale_scores(x, 10))
    return df.sort_values(by="gpt_score", ascending=False).reset_index(drop=True)

def rescale_scores(score, max_score=10):
    return round((score / max_score) * 100)

def expand_query_gpt(query):
    prompt = f"以下の日本語の症状から、英語の医学的キーワードを3つ予測してください：{query}\n出力形式: keyword1, keyword2, keyword3"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    text = response["choices"][0]["message"]["content"]
    return [kw.strip() for kw in text.split(",")]

def predict_soc_category(query):
    prompt = f"以下の症状に最も関連の深いSOC名（英語）を1つ予測してください：{query}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"].strip()

def format_keywords(keywords):
    return ", ".join(keywords)

def add_hierarchy_info(df, term_master_df):
    merged = pd.merge(df, term_master_df, left_on="term", right_on="PT_English", how="left")
    return merged[["term", "score", "gpt_score", "score_percent", "HLT", "HLGT", "SOC"]]

def load_score_cache():
    try:
        with open("score_cache.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return {}

def save_score_cache(cache):
    with open("score_cache.pkl", "wb") as f:
        pickle.dump(cache, f)
