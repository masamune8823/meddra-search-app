
import numpy as np
import openai
import pickle

from sklearn.metrics.pairwise import cosine_similarity

def encode_query(text):
    # ダミーのエンコード処理（本番ではMiniLMなどを使用）
    return np.random.rand(384).astype("float32")

def expand_query_gpt(query):
    return ["疼痛", "神経痛", "ズキズキ"]

def search_meddra(query, faiss_index, meddra_terms, synonym_df, top_k=10):
    results = []
    expanded = expand_query_gpt(query)
    synonyms = synonym_df.loc[query]["synonyms"] if query in synonym_df.index else []
    all_terms = [query] + synonyms + expanded
    seen = set()

    for term in all_terms:
        if term not in seen:
            seen.add(term)
            vec = encode_query(term)
            D, I = faiss_index.search(np.array([vec]), top_k)
            for idx in I[0]:
                results.append((meddra_terms[idx], "faiss"))
    import pandas as pd
    df = pd.DataFrame(results, columns=["PT_Japanese", "source"]).drop_duplicates()
    df["score"] = 50.0
    return df

def rerank_results_v13(query, df, top_n=10):
    df = df.copy()
    df["score"] = 60.0 + np.random.rand(len(df)) * 40
    return df.sort_values("score", ascending=False).head(top_n)

def add_hierarchy_info(df, term_master_df):
    return df.merge(term_master_df, how="left", on="PT_Japanese")

def predict_soc_keywords_with_gpt(query):
    return ["神経", "疼痛", "感覚"]

def filter_by_predicted_soc(df, keywords):
    if not keywords:
        return df
    cols = ["SOC", "HLGT", "HLT"]
    mask = df[cols].apply(lambda x: x.astype(str).str.contains("|".join(keywords)).any(), axis=1)
    return df[mask].copy()

def rescale_scores(df, col="score"):
    min_val, max_val = df[col].min(), df[col].max()
    if max_val > min_val:
        df["確からしさ（％）"] = ((df[col] - min_val) / (max_val - min_val)) * 100
    else:
        df["確からしさ（％）"] = 100
    df["確からしさ（％）"] = df["確からしさ（％）"].round(1)
    return df
