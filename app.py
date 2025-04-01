# app.py（シノニム統合版・分割ファイル対応）

import streamlit as st
import pandas as pd
import pickle
import numpy as np
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

# 分割ファイルを結合して保存する関数
def combine_parts(part_names, output_path):
    with open(output_path, "wb") as output_file:
        for part_name in part_names:
            with open(part_name, "rb") as part_file:
                output_file.write(part_file.read())

# ZIPファイルを展開する関数
def unzip_file(zip_path, extract_to="."):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

# データの読み込み関数
@st.cache_resource
def load_data():
    # ZIPファイル展開
    if os.path.exists("streamlit_app_bundle.zip"):
        unzip_file("streamlit_app_bundle.zip", ".")

    # 分割されたFAISSインデックスを結合・復元
    if not os.path.exists("faiss_index.index"):
        combine_parts(["faiss_index_part_a", "faiss_index_part_b"], "faiss_index.index")

    # 分割された埋め込みベクトルを結合・復元
    if not os.path.exists("meddra_embeddings.npy"):
        combine_parts(["meddra_embeddings_part_a", "meddra_embeddings_part_b"], "meddra_embeddings.npy")

    # FAISSインデックスと埋め込みベクトル
    index = faiss.read_index("faiss_index.index")
    embeddings = np.load("meddra_embeddings.npy")

    # その他データ
    with open("term_master_df.pkl", "rb") as f:
        terms = pickle.load(f)
    with open("synonym_df_cat1.pkl", "rb") as f:
        synonym_df = pickle.load(f)

    return index, embeddings, terms, synonym_df

# 検索実行関数
def search_terms(user_input, index, embeddings, terms, synonym_df):
    # クエリ拡張
    expanded_terms = expand_query_gpt(user_input)

    # 各クエリごとにベクトル検索
    all_results = []
    for term in expanded_terms:
        query_vec = encode_query(term)
        D, I = index.search(np.array([query_vec]), k=10)
        for dist, idx in zip(D[0], I[0]):
            if idx < len(terms):
                row = terms.iloc[idx].to_dict()
                row["score"] = 100 - dist  # 類似度をスコア化
                row["source"] = f"FAISS:{term}"
                all_results.append(row)

    faiss_df = pd.DataFrame(all_results)
    faiss_df = rerank_results_v13(faiss_df)

    # シノニム検索
    synonym_df_filtered = match_synonyms(user_input, synonym_df)

    # 結合
    final_df = merge_faiss_and_synonym_results(faiss_df, synonym_df_filtered)

    return final_df

# UI部分
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
