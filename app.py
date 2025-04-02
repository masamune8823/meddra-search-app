# app.py（安定版・search_assets分割対応・シノニム統合）

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import faiss
import os
import zipfile

from helper_functions import (
    expand_query_gpt,
    encode_query,
    rerank_results_v13,
    match_synonyms,
    merge_faiss_and_synonym_results
)

# 🔧 分割ファイルからZIPを復元
def restore_search_assets_from_parts():
    parts = ["search_assets_part_a", "search_assets_part_b", "search_assets_part_c", "search_assets_part_d"]
    output = "search_assets.zip"
    if not os.path.exists(output):
        with open(output, "wb") as f_out:
            for part in parts:
                with open(part, "rb") as f_in:
                    f_out.write(f_in.read())

# 🔄 ZIPからsearch_assets.pklを読み込む
@st.cache_resource
def load_search_assets():
    restore_search_assets_from_parts()
    with zipfile.ZipFile("search_assets.zip", "r") as zip_ref:
        zip_ref.extractall("extracted_assets")
    with open("extracted_assets/search_assets.pkl", "rb") as f:
        return pickle.load(f)

# 🔄 FAISS関係のデータ読み込み
@st.cache_resource
def load_meddra_faiss_assets():
    index = faiss.read_index("faiss_index.index")
    embeddings = np.load("meddra_embeddings.npy")
    with open("term_master_df.pkl", "rb") as f:
        term_master_df = pickle.load(f)
    return index, embeddings, term_master_df

# 🔍 メイン検索ロジック
def search_terms(user_input, index, embeddings, term_master_df, synonym_df):
    expanded_terms = expand_query_gpt(user_input)
    all_results = []

    for term in expanded_terms:
        query_vec = encode_query(term)
        D, I = index.search(np.array([query_vec]), k=10)
        for dist, idx in zip(D[0], I[0]):
            if idx < len(term_master_df):
                row = term_master_df.iloc[idx].to_dict()
                row["score"] = 100 - dist
                row["source"] = f"FAISS:{term}"
                all_results.append(row)

    faiss_df = pd.DataFrame(all_results)
    faiss_df = rerank_results_v13(faiss_df)
    synonym_df_filtered = match_synonyms(user_input, synonym_df)
    final_df = merge_faiss_and_synonym_results(faiss_df, synonym_df_filtered)

    return final_df

# 🖥️ UI本体
st.markdown("## 💊 MedDRA検索アプリ")
st.markdown("症状や記述を入力してください")

user_input = st.text_input("症状入力", value="頭痛")

if st.button("検索"):
    with st.spinner("検索中..."):
        try:
            synonym_df = load_search_assets()
            index, embeddings, term_master_df = load_meddra_faiss_assets()
            results_df = search_terms(user_input, index, embeddings, term_master_df, synonym_df)

            if not results_df.empty:
                st.success("検索結果が見つかりました。")
                st.dataframe(results_df)
            else:
                st.warning("該当する用語が見つかりませんでした。")
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
