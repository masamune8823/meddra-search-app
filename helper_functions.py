import openai
import numpy as np
import pandas as pd
import faiss
import os
import pickle

# OpenAI API キーを環境変数から取得
openai.api_key = os.getenv("OPENAI_API_KEY")

# クエリ拡張（GPTベース）
def expand_query_gpt(query, cache_path="/mnt/data/query_expansion_cache.pkl"):
    cache = load_cache(cache_path)
    if query in cache:
        return cache[query]

    prompt = (
        f"以下は医療症状に関する言葉です。
"
        f"ユーザーの訴え：{query}
"
        f"これに関連する代表的なキーワードを3つ、短く簡潔に日本語で挙げてください。
"
        f"例：
"
        f"訴え：皮膚がかゆい → かゆみ, 発疹, アレルギー
"
        f"訴え：頭が痛い → 頭痛, ズキズキ, 偏頭痛
"
        f"訴え：{query} →"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
        )
        expanded_text = response.choices[0].message["content"]
        expanded_terms = [w.strip() for w in expanded_text.replace("。", "").split(",")]
    except Exception as e:
        expanded_terms = [query]

    cache[query] = expanded_terms
    save_cache(cache, cache_path)
    return expanded_terms

# クエリのベクトル化
def encode_query(query, model, tokenizer):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings.astype("float32")

# FAISS検索
def search_meddra(query_vector, faiss_index, terms, top_k=10):
    D, I = faiss_index.search(np.array([query_vector]), top_k)
    results = []
    for i, idx in enumerate(I[0]):
        result = {
            "term": terms[idx],
            "score": float(D[0][i]),
            "index": int(idx)
        }
        results.append(result)
    return results

# GPTスコアによる再ランキング
def rerank_results_gpt(query, results, cache_path="/mnt/data/score_cache.pkl"):
    cache = load_cache(cache_path)
    reranked = []

    for r in results:
        key = f"{query}__{r['term']}"
        if key in cache:
            score = cache[key]
        else:
            prompt = f"ユーザーの訴え「{query}」に対して、「{r['term']}」はどの程度関連していますか？10点満点で数字のみで評価してください。"
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                )
                score_text = response.choices[0].message["content"].strip()
                score = int(score_text[:2]) if score_text[:2].isdigit() else 5
            except Exception as e:
                score = 5
            cache[key] = score
        reranked.append({**r, "gpt_score": score})

    reranked = sorted(reranked, key=lambda x: x["gpt_score"], reverse=True)
    save_cache(cache, cache_path)
    return reranked

# MedDRA階層情報の追加
def add_hierarchy_info(results_df, term_master_df):
    merged_df = results_df.merge(
        term_master_df,
        how="left",
        left_on="term",
        right_on="PT_Japanese"
    )
    return merged_df

# キャッシュの読み書き
def load_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return {}

def save_cache(cache, cache_path):
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
