# app.py（完全修正版・分割ファイル対応・デグレなし）
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import faiss
import os

from helper_functions import (
    expand_query_gpt,
    encode_query,
    rerank_results_v13,
    match_synonyms,
    merge_faiss_and_synonym_results
)

# 🔧 FAISSインデックス復元関数
def restore_faiss_index_from_parts():
    part_a = "faiss_index_part_a"
    part_b = "faiss_index_part_b"
    output = "faiss_index.index"
    if not os.path.exists(output):
        with open(output, "wb") as f_out:
            for part in [part_a, part_b]:
                with open(part, "rb") as f_in:
                    f_out.write(f_in.read())

# 🔧 ベクトル復元関数
def restore_meddra_embeddings_from_parts():
    part_a = "meddra_embeddings_part_a"
    part_b = "meddra_embeddings_part_b"
    output = "meddra_embeddings.npy"
    if not os.path.exists(output):
        with open(output, "wb") as f_out:
            for part in [part_a, part_b]:
                with open(part, "rb") as f_in:
                    f_out.write(f_in.read())

# 🔁 初回キャッシュ用データロード
@st.cache_resource
def load_data():
    restore_faiss_index_from_parts()
    restore_meddra_embeddings_from_parts()

    # ベクトルと用語リストの読み込み
    embeddings = np.load("meddra_embeddings.npy")
    with open("meddra_terms.npy", "rb") as f:
        terms = np.load(f, allow_pickle=True)

    # マスタとシノニム辞書の読み込み
    with open("term_master_df.pkl", "rb") as f:
        term_master_df = pickle.load(f)
    with open("synonym_df_cat1.pkl", "rb") as f:
        synonym_df = pickle.load(f)

    return terms, embeddings, term_master_df, synonym_df

# 🔁 FAISSインデックスの読み込み
@st.cache_resource
def load_faiss_index():
    index = faiss.read_index("faiss_index.index")
    return index

# 💻 UI本体
st.markdown("## 💊 MedDRA検索アプリ")
st.write("症状や記述を入力してください")

user_query = st.text_input("症状入力", "頭痛")

if st.button("検索"):
    if user_query:
        try:
            terms, embeddings, term_master_df, synonym_df = load_data()
            index = load_faiss_index()

            # クエリ拡張（OpenAI API または仮の処理）
            expanded_terms = expand_query_gpt(user_query)

            # 検索処理
            results = []
            for term in expanded_terms:
                query_vec = encode_query(term)
                D, I = index.search(np.array([query_vec]), k=10)
                for score, idx in zip(D[0], I[0]):
                    results.append({"term": terms[idx], "score": float(score)})

            # シノニム検索
            synonym_matches = match_synonyms(expanded_terms, synonym_df)

            # マージして再ランキング
            merged = merge_faiss_and_synonym_results(results, synonym_matches)
            reranked = rerank_results_v13(merged)

            # 結果表示
            df = pd.DataFrame(reranked)
            st.write("### 🔍 検索結果（上位）")
            st.dataframe(df)

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
