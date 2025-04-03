import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from openai import OpenAIError
from config import faiss_index, meddra_terms
import openai


# OpenAI GPTでクエリ拡張
def expand_query_gpt(user_input, cache_path="query_expansion_cache.pkl"):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    if user_input in cache:
        return cache[user_input]

    prompt = f"以下は医療症状に関する言葉です。
ユーザーの訴え：「{user_input}」
この訴えから適切な医療用語・キーワードを3～5個、日本語で箇条書きで挙げてください。"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "あなたは医療分野に詳しいAIアシスタントです。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        choices = response["choices"][0]["message"]["content"]
        keywords = [line.strip("・-● ").strip() for line in choices.split("
") if line.strip()]
    except OpenAIError:
        keywords = [user_input]

    cache[user_input] = keywords
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    return keywords


# クエリをベクトルに変換
def encode_query(query, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(query)


# FAISS検索本体
def search_faiss(query_vector, index, terms, top_k=10):
    D, I = index.search(np.array([query_vector]), top_k)
    return [(terms[i], float(D[0][idx])) for idx, i in enumerate(I[0])]


# MedDRA検索（拡張語ごとにFAISSし再スコアリング前の候補を返す）
def search_meddra(query, top_k_per_method=5):
    keywords = expand_query_gpt(query)
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    all_results = []

    for kw in keywords:
        q_vec = model.encode(kw)
        res = search_faiss(q_vec, faiss_index, meddra_terms, top_k=top_k_per_method)
        all_results.extend(res)

    return list(set(all_results))


# 再スコアリング（キャッシュ活用）
def rerank_results_v13(results, query, top_n=10, cache_path="score_cache.pkl"):
    key = (query, tuple(results))
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    if key in cache:
        return cache[key]

    try:
        messages = [
            {"role": "system", "content": "あなたは医療テキストを解析するAIです。"},
            {"role": "user", "content": f"以下の訴えに対して、関連する候補語を関連度が高い順に並べ、10点満点で評価してください。

訴え: {query}

候補語:
" + "
".join([f"- {term}" for term, _ in results])}
        ]
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3,
        )
        content = completion["choices"][0]["message"]["content"]
        lines = content.split("
")
        scored = []
        for line in lines:
            for term, _ in results:
                if term in line:
                    digits = [s for s in line if s.isdigit()]
                    if digits:
                        score = int("".join(digits[:2])) if len(digits) > 1 else int(digits[0])
                        scored.append((term, min(score, 10)))
                        break
    except Exception:
        scored = [(term, 5) for term, _ in results]

    scored.sort(key=lambda x: x[1], reverse=True)
    top_results = scored[:top_n]
    cache[key] = top_results

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    return top_results


# 階層情報追加（ダミー：後ほど必要に応じて定義）
def add_hierarchy_info(results, term_master_df):
    return results  # 今はそのまま返す


# term_master_df 読み込み（Streamlit側で使用）
def load_term_master_df(path):
    return pd.read_pickle(path)