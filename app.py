# app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import faiss
from helper_functions import (
    expand_query_gpt,
    encode_query,
    rerank_results_v13,
    match_synonyms,
    merge_faiss_and_synonym_results,
)

# 🔧 FAISS・ベクトル・アセットの復元
def restore_faiss_index_from_parts():
    parts = ["faiss_index_part_a", "faiss_index_part_b"]
    if not os.path.exists("faiss_index.index"):
        with open("faiss_index.index", "wb") as f_out:
            for part in parts:
                with open(part, "rb") as f_in:
                    f_out.write(f_in.read())

def restore_meddra_embeddings_from_parts():
    parts = ["meddra_embeddings_part_a", "meddra_embeddings_part_b"]
    if not os.path.exists("meddra_embeddings.npy"):
        with open("meddra_embeddings.npy", "wb") as f_out:
            for part in parts:
                with open(part, "rb") as f_in:
                    f_out.write(f_in.read())

# 🔁 初期データロード
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

@st.cache_resource
def load_faiss_index():
    index = faiss.read_index("faiss_index.index")
    return index

# 💻 Streamlit UI
st.markdown("## 💊 MedDRA検索アプリ")
st.write("症状や記述を入力してください")

user_query = st.text_input("症状入力", "頭痛")

if st.button("検索"):
    if user_query:
        terms, embeddings, term_master_df, synonym_df = load_data()
        index = load_faiss_index()

        # ✅ クエリ拡張
        expanded_terms = expand_query_gpt(user_query)
        st.info(f"🔍 拡張語: {expanded_terms}")

        # ✅ FAISS検索
        results = []
        for term in expanded_terms:
            query_vec = encode_query(term)
            D, I = index.search(np.array([query_vec]), k=10)
            for score, idx in zip(D[0], I[0]):
                if idx != -1:
                    row = {
                        "PT_Japanese": terms[idx]["PT_Japanese"],
                        "PT_English": terms[idx]["PT_English"],
                        "PT_ID": terms[idx]["PT_ID"],
                        "HLT_ID": terms[idx]["HLT_ID"],
                        "HLT_Japanese": terms[idx]["HLT_Japanese"],
                        "score": float(score),
                        "source": "FAISS"
                    }
                    results.append(row)
        faiss_df = pd.DataFrame(results)

        # ✅ シノニム検索
        synonym_df_matched = match_synonyms(expanded_terms, synonym_df)

        # ✅ マージ・再ランキング
        merged_df = merge_faiss_and_synonym_results(faiss_df, synonym_df_matched)
        reranked = rerank_results_v13(merged_df)

        # ✅ 結果表示
        st.write("### 🔎 検索結果")
        st.dataframe(reranked)

        # ✅ キャッシュ情報表示
        if os.path.exists("score_cache.pkl"):
            st.success("✅ score_cache.pkl（再ランキングキャッシュ）は存在します")
        else:
            st.warning("⚠️ score_cache.pkl は存在しません")
