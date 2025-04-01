# app.py（シノニム統合版）

import streamlit as st
import pandas as pd
import pickle
import io
from helper_functions import expand_query_gpt, encode_query, rerank_results_v13, match_synonyms, merge_faiss_and_synonym_results
import numpy as np
import faiss
import os

@st.cache_resource
def load_faiss_index():
    with open("search_assets_part_a", "rb") as f:
        part_a = f.read()
    with open("search_assets_part_b", "rb") as f:
        part_b = f.read()
    with open("search_assets_part_c", "rb") as f:
        part_c = f.read()
    with open("search_assets_part_d", "rb") as f:
        part_d = f.read()

    combined = part_a + part_b + part_c + part_d
    with open("streamlit_app_bundle.zip", "wb") as f:
        f.write(combined)

    os.system("unzip -o streamlit_app_bundle.zip")
    index = faiss.read_index("faiss_index.index")
    return index

@st.cache_data
def load_data():
    # 分割済みベクトルファイルを復元
    with open("meddra_embeddings_part_a", "rb") as f:
        part_a = f.read()
    with open("meddra_embeddings_part_b", "rb") as f:
        part_b = f.read()
    combined = part_a + part_b
    embeddings = np.load(io.BytesIO(combined))

    # その他ファイル読み込み
    with open("meddra_terms.npy", "rb") as f:
        terms = np.load(f, allow_pickle=True)
    with open("term_master_df.pkl", "rb") as f:
        term_master_df = pickle.load(f)
    with open("synonym_df_cat1.pkl", "rb") as f:
        synonym_df = pickle.load(f)

    return terms, embeddings, synonym_df, term_master_df

# Streamlit UI
st.title("MedDRA検索アプリ")
st.write("症状や記述を入力してください")

user_query = st.text_input("症状入力", "頭痛")

if st.button("検索"):
    index = load_faiss_index()
    terms, embeddings, synonym_df, term_master_df = load_data()

    with st.spinner("検索中..."):
        # クエリ拡張
        expanded_queries = expand_query_gpt(user_query)

        # FAISS検索
        faiss_results = []
        for q in expanded_queries:
            query_vec = encode_query(q)
            D, I = index.search(np.array([query_vec]), k=20)
            for dist, idx in zip(D[0], I[0]):
                faiss_results.append({
                    "term": terms[idx],
                    "score": 10 - dist  # 疑似スコア（近いほど高スコア）
                })

        # シノニムマッチ
        synonym_matches = match_synonyms(user_query, synonym_df)

        # 結果統合
        merged_results = merge_faiss_and_synonym_results(faiss_results, synonym_matches)

        # 再ランキング
        reranked = rerank_results_v13(merged_results)

        # 表示用にPT階層情報を付加
        result_df = pd.DataFrame(reranked)
        result_df = result_df.merge(term_master_df, how="left", left_on="term", right_on="PT_English")

        st.success("検索完了")
        st.dataframe(result_df[["term", "score", "PT_Japanese", "HLT_Japanese", "HLGT_Japanese", "SOC_Japanese"]])
