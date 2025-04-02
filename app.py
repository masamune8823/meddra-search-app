# app.py（分割ZIP対応・GPT拡張＋FAISS＋再ランキング＋シノニム対応）

import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
import zipfile
import pickle
import glob

from helper_functions import (
    expand_query_gpt,
    encode_query,
    rerank_results_v13,
    match_synonyms,
    merge_faiss_and_synonym_results
)

# 分割ZIPファイルを結合して展開する関数
def restore_split_zip(base_name="streamlit_app_bundle.zip", part_ext=".zip.", extract_to="."):
    part_files = sorted(glob.glob(base_name + ".*"))
    if part_files:
        with open(base_name, "wb") as f_out:
            for part in part_files:
                with open(part, "rb") as f_in:
                    f_out.write(f_in.read())
        with zipfile.ZipFile(base_name, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

# バイナリ結合用ユーティリティ（indexやnpy用）
def combine_parts(part_names, output_path):
    with open(output_path, "wb") as output_file:
        for part_name in part_names:
            with open(part_name, "rb") as part_file:
                output_file.write(part_file.read())

# データの読み込み
@st.cache_resource
def load_data():
    # ZIP分割ファイルからの復元
    if not os.path.exists("faiss_index.index"):
        restore_split_zip()

    # faiss_indexがない場合は手動で結合
    if not os.path.exists("faiss_index.index") and os.path.exists("faiss_index_part_a"):
        combine_parts(["faiss_index_part_a", "faiss_index_part_b"], "faiss_index.index")
    if not os.path.exists("meddra_embeddings.npy") and os.path.exists("meddra_embeddings_part_a"):
        combine_parts(["meddra_embeddings_part_a", "meddra_embeddings_part_b"], "meddra_embeddings.npy")

    index = faiss.read_index("faiss_index.index")
    embeddings = np.load("meddra_embeddings.npy")

    with open("term_master_df.pkl", "rb") as f:
        terms = pickle.load(f)
    with open("synonym_df_cat1.pkl", "rb") as f:
        synonym_df = pickle.load(f)

    return index, embeddings, terms, synonym_df

# 検索処理
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
st.markdown("自然言語で症状や所見を入力してください")

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
