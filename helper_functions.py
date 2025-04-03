
import numpy as np
import pandas as pd
import openai
import pickle
import os
import hashlib

# --- Utility: クエリをベクトル化 ---
def encode_query(query, model=None):
    # 実装例（要：外部でベクトル化）
    # ここは仮実装。呼び出し元でベクトル化を済ませておく前提でもOK
    pass

# --- synonym + FAISS統合検索 ---
def search_meddra(query, faiss_index, faiss_index_synonym, synonym_df, meddra_terms, top_k=20):
    import sentence_transformers
    model = sentence_transformers.SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    query_vector = model.encode([query])

    D, I = faiss_index.search(np.array(query_vector).astype("float32"), top_k)
    terms_main = [meddra_terms[i] for i in I[0]]
    results = list(zip(terms_main, D[0], ["faiss"] * len(terms_main)))

    matched = synonym_df[synonym_df["synonym"].str.contains(query, na=False)]
    if not matched.empty:
        for keyword in matched["PT_English"].unique():
            synonym_vector = model.encode([keyword])
            D_syn, I_syn = faiss_index_synonym.search(np.array(synonym_vector).astype("float32"), top_k)
            terms_syn = [meddra_terms[i] for i in I_syn[0]]
            results += list(zip(terms_syn, D_syn[0], ["synonym"] * len(terms_syn)))

    return results

# --- GPT再ランキング（Top N） ---
def rerank_results_v13(results, query, client, top_n=10, cache_path="score_cache.pkl"):
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)

    def _hash(q, t):
        return hashlib.md5(f"{q}_{t}".encode()).hexdigest()

    reranked = []
    top_results = results[:top_n]

    for term, score, source in top_results:
        key = _hash(query, term)
        if key in cache:
            new_score = cache[key]
        else:
            messages = [
                {"role": "system", "content": "あなたは医学に詳しいAIです。"},
                {"role": "user", "content": f"以下の症状にどれだけ関連がありますか？

症状: {query}
用語: {term}

0〜100点で評価してください。"}
            ]
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                )
                content = response.choices[0].message.content.strip()
                new_score = float([s for s in content.split() if s.replace('.', '', 1).isdigit()][0])
            except:
                new_score = 50.0
            cache[key] = new_score

        reranked.append((term, new_score, source))

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked

# --- GPTベースのSOC予測 ---
def predict_soc_keywords_with_gpt(query, client, cache_path="soc_predict_cache.pkl"):
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)

    if query in cache:
        return cache[query]

    messages = [
        {"role": "system", "content": "あなたは医学用語に詳しいAIです。"},
        {"role": "user", "content": f"この症状「{query}」に関連するキーワード（例：神経、皮膚、消化、精神）を1〜3個出力してください。"}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        content = response.choices[0].message.content
        keywords = [kw.strip() for kw in content.split("、") if kw.strip()]
    except:
        keywords = []

    cache[query] = keywords
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    return keywords

# --- フィルタ適用 ---
def filter_by_predicted_soc(df, keywords):
    if not keywords or df.empty:
        return df
    mask_include = df[["HLT", "HLGT", "SOC"]].astype(str).apply(
        lambda row: any(kw in row_str for row_str in row for kw in keywords), axis=1
    )
    return df[mask_include].copy()

# --- スコア再スケーリング ---
def rescale_scores(df, score_col="score"):
    min_val = df[score_col].min()
    max_val = df[score_col].max()
    if max_val > min_val:
        df["確からしさ（％）"] = ((df[score_col] - min_val) / (max_val - min_val)) * 100
    else:
        df["確からしさ（％）"] = 100
    df["確からしさ（％）"] = df["確からしさ（％）"].round(1)
    return df
