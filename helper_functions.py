import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import openai

# クエリ拡張（OpenAI GPT 使用）
def expand_query_gpt(q, api_key=None):
    openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
    prompt = f"以下は医療症状に関する言葉です。\nユーザーの訴え: {q}\nこの言葉に関連する症状や疾患名・病名・言い換えを3つ挙げてください。"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7
        )
        text = response.choices[0].message.content.strip()
        return text.splitlines()
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return []

# ベクトル化（MiniLM使用）
def encode_query(query_text, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
    model = SentenceTransformer(model_name)
    return model.encode([query_text])[0]

# FAISS検索
def search_meddra(query_vector, faiss_index, terms, top_k=10):
    distances, indices = faiss_index.search(np.array([query_vector]), top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            'Term': terms[idx],
            'Score': float(distances[0][i]),
            'Index': int(idx)
        })
    return results

# 再ランキング（GPTスコアリング）
def rerank_results_v13(query, candidates, api_key=None):
    openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
    results = []
    for item in candidates:
        term = item["Term"]
        prompt = f"ユーザーの訴え: {query}\n候補用語: {term}\nこの候補は、ユーザーの訴えにどの程度関連しますか？10点満点で評価してください。"
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0
            )
            score_text = response.choices[0].message.content.strip()
            score = int("".join(filter(str.isdigit, score_text)))
            item["Relevance"] = score
        except Exception as e:
            item["Relevance"] = 0
            print(f"Scoring error: {e}")
        results.append(item)
    return sorted(results, key=lambda x: x["Relevance"], reverse=True)

# シノニム対応
def match_synonyms(query, synonym_df):
    matched = synonym_df[synonym_df['synonym'].str.contains(query, na=False)]
    return matched['term'].tolist()

# FAISS検索とシノニムをマージ
def merge_faiss_and_synonym_results(faiss_results, synonym_results):
    faiss_terms = {item['Term']: item for item in faiss_results}
    for term in synonym_results:
        if term not in faiss_terms:
            faiss_results.append({'Term': term, 'Score': None, 'Index': None, 'Relevance': 5})
    return faiss_results