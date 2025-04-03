
import numpy as np
import pandas as pd
import faiss
import openai
import pickle
from sentence_transformers import SentenceTransformer

# モデルとキャッシュ
encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def encode_query(query):
    return encoder.encode([query])[0].astype("float32")

def search_meddra(query, faiss_index, meddra_terms, synonym_df=None, top_k=20):
    all_terms = []
    all_scores = []
    all_indexes = []

    queries = [query]
    if synonym_df is not None and query in synonym_df.index:
        queries += synonym_df.loc[query]["synonyms"]

    for q in queries:
        vec = encode_query(q)
        D, I = faiss_index.search(np.array([vec]), top_k)
        all_terms.extend(meddra_terms[I[0]])
        all_scores.extend(D[0])
        all_indexes.extend(I[0])

    df = pd.DataFrame({
        "term": all_terms,
        "score": all_scores,
        "index": all_indexes
    }).drop_duplicates("term").sort_values("score", ascending=True).reset_index(drop=True)
    return df

def rerank_results_v13(query, results_df, top_n=10):
    top_df = results_df.head(top_n).copy()
    scores = []

    for term in top_df["term"]:
        prompt = f"以下の症状にどれだけ関連があるかを100点満点で評価してください：

症状: {query}
候補: {term}

スコア:"
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )
            score_text = response.choices[0].message["content"].strip()
            score = int("".join(filter(str.isdigit, score_text)))
            score = max(0, min(score, 100))
        except Exception as e:
            print("LLMスコア生成失敗:", e)
            score = 0
        scores.append(score)

    top_df["score"] = scores
    return top_df.sort_values("score", ascending=False).reset_index(drop=True)

def predict_soc_keywords_with_gpt(query):
    prompt = f"以下の症状に関連する医学カテゴリ（例：神経、皮膚、消化器など）を3つ程度、日本語で出力してください：

症状: {query}

関連カテゴリ:"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=50,
        )
        text = response.choices[0].message["content"]
        terms = [term.strip(" ・,、。") for term in text.splitlines() if term.strip()]
        return terms
    except Exception as e:
        print("SOC予測失敗:", e)
        return []

def filter_by_predicted_soc(df, keywords):
    if not keywords:
        return df

    available_cols = [col for col in ["SOC", "HLGT", "HLT"] if col in df.columns]
    if not available_cols:
        print("⚠️ 'SOC', 'HLGT', 'HLT' が DataFrame に存在しません。フィルタをスキップします。")
        return df

    pattern = "|".join(keywords)
    mask = df[available_cols].astype(str).apply(lambda x: x.str.contains(pattern, case=False)).any(axis=1)
    return df[mask].copy()

def rescale_scores(df, col="score"):
    if col not in df.columns or df[col].isnull().all():
        return df

    min_score = df[col].min()
    max_score = df[col].max()
    if max_score == min_score:
        df["rescaled_score"] = 100
    else:
        df["rescaled_score"] = ((df[col] - min_score) / (max_score - min_score) * 100).round(1)
    return df
