# app.py（最新版：分割ZIP対応 + helper_functions使用）

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import faiss
import os
import zipfile
import shutil

from helper_functions import (
    expand_query_gpt,
    encode_query,
    rerank_results_v13,
    match_synonyms,
    merge_faiss_and_synonym_results
)

# ZIPパート結合
def combine_zip_parts(part_prefix, output_zip):
    part_files = sorted([f for f in os.listdir() if f.startswith(part_prefix)])
    with open(output_zip, 'wb') as output:
        for part in part_files:
            with open(part, 'rb') as pf:
                shutil.copyfileobj(pf, output)

# ZIP展開
def unzip_bundle(zip_path, extract_to="."):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# データ読み込み
@st.cache_resource
def load_data():
    if not os.path.exists("streamlit_app_bundle.zip"):
        combine_zip_parts("streamlit_app_bundle.zip", "streamlit_app_bundle.zip")
    unzip_bundle("streamlit_app_bundle.zip", ".")

    index = faiss.read_index("faiss_index.index")
    embeddings = np.load("meddra_embeddings.npy")
    with open("term_master_df.pkl", "rb") as f:
        terms = pickle.load(f)
    with open("synonym_df_cat1.pkl", "rb") as f:
        synonym_df = pickle.load(f)
    return index, embeddings, terms, synonym_df

# 検索
def search_terms(user_input, index, embeddings, terms, synonym_df):
    expanded_terms = expand_query_gpt(user_input)
    all_results = []
    for term in expanded_terms:
        query_vec = encode_query(term)
        D, I = index.search(np.array([query_vec]), k=10)
        for dist, idx in zip(D[0], I[0]):
            if idx < len(terms):
                row = terms.iloc[idx].to_dict()
                row["score"] = 100 - dist
                row["source"] = f"FAISS:{term}"
                all_results.append(row)
    faiss_df = pd.DataFrame(all_results)
    faiss_df = rerank_results_v13(faiss_df)
    synonym_df_filtered = match_synonyms(user_input, synonym_df)
    final_df = merge_faiss_and_synonym_results(faiss_df, synonym_df_filtered)
    return final_df

# UI
st.markdown("## 💊 MedDRA検索アプリ")
st.markdown("症状や記述を入力してください")
user_input = st.text_input("症状入力", value="頭痛")

if st.button("検索"):
    with st.spinner("検索中..."):
        try:
            index, embeddings, terms, synonym_df = load_data()
            results_df = search_terms(user_input, index, embeddings, terms, synonym_df)
            if not results_df.empty:
                st.success("検索結果が見つかりました。")
                st.dataframe(results_df)
            else:
                st.warning("該当する用語が見つかりませんでした。")
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
