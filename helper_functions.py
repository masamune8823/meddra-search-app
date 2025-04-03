
import numpy as np
import pandas as pd
import openai
import os
import pickle
from config import (
    faiss_index,
    faiss_index_synonym,
    meddra_terms,
    synonym_df,
    synonym_vectors,
    encode_query,
)

def search_meddra(query, top_k_per_method=5):
    query_vector = encode_query(query)

    # FAISS検索（メイン）
    D_main, I_main = faiss_index.search(np.array([query_vector]), top_k_per_method)
    results_main = [
        {"term": meddra_terms[i], "score": float(D_main[0][rank]), "source": "faiss"}
        for rank, i in enumerate(I_main[0])
    ]

    # synonym_dfベースの語から追加FAISS検索（synonym_faiss）
    synonym_hits = synonym_df[synonym_df["synonym"].str.contains(query, case=False, na=False)]
    matched_terms = synonym_hits["term"].unique().tolist()

    results_synonym = []
    if len(matched_terms) > 0:
        matched_vecs = [synonym_vectors[matched_terms.index(t)] for t in matched_terms if t in matched_terms]
        if matched_vecs:
            vec_array = np.stack(matched_vecs)
            D_syn, I_syn = faiss_index_synonym.search(vec_array, top_k_per_method)
            for idx, row in enumerate(I_syn):
                for rank, i in enumerate(row):
                    results_synonym.append({
                        "term": meddra_terms[i],
                        "score": float(D_syn[idx][rank]),
                        "source": "synonym_faiss"
                    })

    combined = results_main + results_synonym
    results_df = pd.DataFrame(combined)
    return results_df


def rerank_results_v13(results_df, query, top_n=10, cache_path="score_cache.pkl"):
    results_df = results_df.sort_values("score", ascending=False).drop_duplicates("term")
    top_candidates = results_df.head(top_n)["term"].tolist()
    key = (query, tuple(top_candidates))

    # キャッシュ読み込み
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    if key in cache:
        scores = cache[key]
    else:
        # OpenAI GPT-3.5によるスコアリング
        system_prompt = "あなたは医療用語選定の専門家です。以下の症状に対して、どの用語が最も関連が高いかを評価してください。"
        user_prompt = f"症状：「{query}」\n候補：" + "、".join(top_candidates)

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        scores_text = response.choices[0].message.content.strip().splitlines()
        scores = []
        for line in scores_text:
            for term in top_candidates:
                if term in line:
                    try:
                        score = float("".join(filter(str.isdigit, line)))
                        scores.append((term, score))
                    except:
                        pass
        if len(scores) < len(top_candidates):
            missing = set(top_candidates) - {t for t, _ in scores}
            for m in missing:
                scores.append((m, 50))

        cache[key] = scores
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)

    score_map = dict(scores)
    results_df["rerank_score"] = results_df["term"].map(score_map)
    results_df = results_df.sort_values("rerank_score", ascending=False).copy()
    results_df["確からしさ（％）"] = results_df["rerank_score"].round(1)
    return results_df
