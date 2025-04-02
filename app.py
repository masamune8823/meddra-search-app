# app.py（最新版：FAISS+Synonym統合・分割ファイル復元対応済み）
import streamlit as st
import pandas as pd
import pickle
from helper_functions import (
    expand_query_gpt,
    encode_query,
    rerank_results_v13,
    match_synonyms,
    merge_faiss_and_synonym_results
)
import numpy as np
import faiss
import os

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
    restore_faiss_index_from_parts()
    index = faiss.read_index("faiss_index.index")
    return index

# 💻 UI本体
st.markdown("## 💊 MedDRA検索アプリ")
st.write("症状や記述を入力してください")

user_query = st.text_input("症状入力", "頭痛")

if st.button("検索"):
    if user_query:
        terms, embeddings, term_master_df, synonym_df = load_data()
        index = load_faiss_index()

        # クエリ拡張（OpenAI API）
        expanded_terms = expand_query_gpt(user_query)

        # FAISS検索
        results = []
        for term in expanded_terms:
            query_vec = encode_query(term)
            if query_vec.shape[0] != index.d:
                continue
            D, I = index.search(np.array([query_vec]), k=10)
            for score, idx in zip(D[0], I[0]):
                if idx == -1:
                    continue
                row = term_master_df.iloc[idx].to_dict()
                row["score"] = float(score)
                row["source"] = "FAISS"
                results.append(row)

        faiss_df = pd.DataFrame(results)

        # シノニム検索
        synonym_matches = match_synonyms(expanded_terms, synonym_df)

        # マージして再ランキング
        merged = merge_faiss_and_synonym_results(faiss_df, synonym_matches)
        reranked = rerank_results_v13(merged)

        # 結果表示
        st.markdown("### 🔍 検索結果")
        if reranked.empty:
            st.info("該当する結果が見つかりませんでした。")
        else:
            st.dataframe(reranked)
