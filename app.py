# app.py（完全修正版・分割ZIP対応）

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

# 分割ZIP（.zip.001～）から1つのZIPへ復元する関数
def restore_split_zip(base_name="streamlit_app_bundle.zip", part_count=4):
    if os.path.exists(base_name):
        return  # 既に復元済みならスキップ
    with open(base_name, "wb") as output:
        for i in range(part_count):
            part_name = f"{base_name}.{str(i+1).zfill(3)}"
            if not os.path.exists(part_name):
                raise FileNotFoundError(f"Missing part file: {part_name}")
            with open(part_name, "rb") as part_file:
                output.write(part_file.read())

# ZIPファイルを展開する関数
def unzip_file(zip_path, extract_to="."):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

# データの読み込み関数（ZIP復元込み）
@st.cache_resource
def load_data():
    # ZIP結合＆展開
    if not os.path.exists("faiss_index.index"):
        restore_split_zip()
        unzip_file("streamlit_app_bundle.zip")

    # ファイル存在確認（万が一の展開失敗対策）
    if not os.path.exists("faiss_index.index"):
        raise FileNotFoundError("faiss_index.index が見つかりません。ZIP展開に失敗した可能性があります。")
    if not os.path.exists("meddra_embeddings.npy"):
        raise FileNotFoundError("meddra_embeddings.npy が見つかりません。")
    if not os.path.exists("term_master_df.pkl"):
        raise FileNotFoundError("term_master_df.pkl が見つかりません。")
    if not os.path.exists("synonym_df_cat1.pkl"):
        raise FileNotFoundError("synonym_df_cat1.pkl が見つかりません。")

    # 読み込み処理
    index = faiss.read_index("faiss_index.index")
    embeddings = np.load("meddra_embeddings.npy")
    with open("term_master_df.pkl", "rb") as f:
        terms = pickle.load(f)
    with open("synonym_df_cat1.pkl", "rb") as f:
        synonym_df = pickle.load(f)

    return index, embeddings, terms, synonym_df

# 検索実行関数
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
