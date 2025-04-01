# app.py（シノニム統合＋分割ファイル復元対応）

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

# --- STEP 0: ファイル復元処理 ---
with st.spinner("ZIPファイルを展開中..."):
    zip_path = "streamlit_app_bundle.zip"
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()

# --- STEP 1: データの読み込み ---
@st.cache_resource
def load_data():
    with open("term_master_df.pkl", "rb") as f:
        term_master_df = pickle.load(f)
    with open("synonym_df_cat1.pkl", "rb") as f:
        synonym_df = pickle.load(f)
    embeddings = np.load("meddra_embeddings.npy")
    return term_master_df, embeddings, synonym_df

# --- STEP 2: FAISS index の読み込み ---
@st.cache_resource
def load_faiss_index():
    # 分割ファイルを結合して復元
    with open("faiss_index_part_a", "rb") as f1, open("faiss_index_part_b", "rb") as f2:
        with open("faiss_index.index", "wb") as out_f:
            out_f.write(f1.read())
            out_f.write(f2.read())
    # 復元した index を読み込み
    index = faiss.read_index("faiss_index.index")
    return index

# --- データの読み込み ---
term_master_df, embeddings, synonym_df = load_data()
index = load_faiss_index()

# --- Streamlit UI ---
st.image("https://img.icons8.com/emoji/48/stethoscope-emoji.png", width=40)
st.markdown("## MedDRA検索アプリ")
st.write("症状や記述を入力してください")

query = st.text_input("症状入力", "")

if st.button("検索"):
    if query.strip() == "":
        st.warning("症状を入力してください。")
    else:
        # クエリ拡張
        expanded_terms = expand_query_gpt(query)

        # FAISS検索
        results = []
        for term in expanded_terms:
            query_vec = encode_query(term)
            D, I = index.search(np.array([query_vec]), k=20)
            for dist, idx in zip(D[0], I[0]):
                pt = term_master_df.iloc[idx]["PT_Japanese"]
                results.append({"term": pt, "score": float(1 / (1 + dist))})  # 類似度スコアに変換

        # スコア再計算（仮）
        reranked = rerank_results_v13(results)

        # シノニム補完
        matched = match_synonyms(query, synonym_df)

        # 結合＆表示
        final_results = merge_faiss_and_synonym_results(reranked, matched)

        st.markdown("### 🔍 検索結果")
        if final_results:
            for res in final_results[:10]:
                st.write(f"• {res['term']}（確からしさ: {res['score']:.2f}）")
        else:
            st.info("一致する用語が見つかりませんでした。")
