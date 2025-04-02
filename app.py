# app.py

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import faiss
import os

from helper_functions import (
    expand_query_gpt,
    encode_query,
    rerank_results_v13,
    match_synonyms,
    merge_faiss_and_synonym_results
)

# 🔧 データ結合（必要なら）
def restore_faiss_index_from_parts():
    if not os.path.exists("faiss_index.index"):
        with open("faiss_index.index", "wb") as w:
            for part in ["faiss_index_part_a", "faiss_index_part_b"]:
                with open(part, "rb") as r:
                    w.write(r.read())

def restore_meddra_embeddings_from_parts():
    if not os.path.exists("meddra_embeddings.npy"):
        with open("meddra_embeddings.npy", "wb") as w:
            for part in ["meddra_embeddings_part_a", "meddra_embeddings_part_b"]:
                with open(part, "rb") as r:
                    w.write(r.read())

# ✅ キャッシュ付き初期データ読み込み
@st.cache_resource
def load_data():
    restore_faiss_index_from_parts()
    restore_meddra_embeddings_from_parts()
    embeddings = np.load("meddra_embeddings.npy")
    with open("meddra_terms.npy", "rb") as f:
        terms = np.load(f, allow_pickle=True)
    with open("term_master_df.pkl", "rb") as f:
        term_master_df = pickle.load(f)
    with open("synonym_df_cat1.pkl", "rb") as f:
        synonym_df = pickle.load(f)
    return terms, embeddings, term_master_df, synonym_df

# ✅ FAISSインデックスの読み込み
@st.cache_resource
def load_faiss_index():
    restore_faiss_index_from_parts()
    index = faiss.read_index("faiss_index.index")
    return index

# 🔷 Streamlit UI
st.markdown("## 💊 MedDRA検索アプリ")
st.write("症状や記述を入力してください")

user_query = st.text_input("症状入力", "頭痛")

if st.button("検索"):
    if user_query:
        terms, embeddings, term_master_df, synonym_df = load_data()
        index = load_faiss_index()

        # 拡張
        expanded_terms = expand_query_gpt(user_query)

        # 検索
        results = []
        for term in expanded_terms:
            query_vec = encode_query(term)
            D, I = index.search(np.array([query_vec]), k=10)
            for score, idx in zip(D[0], I[0]):
                if idx < len(terms):
                    row = term_master_df[term_master_df["PT_English"] == terms[idx]]
                    if not row.empty:
                        result = row.iloc[0].to_dict()
                        result["score"] = float(score)
                        result["source"] = "FAISS"
                        results.append(result)

        faiss_df = pd.DataFrame(results)
        synonym_df_matched = match_synonyms(expanded_terms, synonym_df)
        merged = merge_faiss_and_synonym_results(faiss_df, synonym_df_matched)
        reranked = rerank_results_v13(merged)

        st.write("### 🔍 検索結果")
        st.dataframe(reranked)
