
import streamlit as st
import pickle
import os
import numpy as np
import pandas as pd

from helper_functions import (
    encode_query,
    search_meddra,
    rerank_results_v13,
    expand_query_gpt,
    predict_soc_category,
    format_keywords,
    rescale_scores,
    add_hierarchy_info,
    load_score_cache,
    save_score_cache
)

@st.cache_resource
def load_faiss_index(path):
    import faiss
    return faiss.read_index(path)

@st.cache_data
def load_assets():
    index = load_faiss_index("faiss_index.index")
    terms = np.load("meddra_terms.npy", allow_pickle=True)
    synonym_df = pd.read_pickle("synonym_df_cat1.pkl")
    term_master_df = pd.read_pickle("term_master_df.pkl")
    return index, terms, synonym_df, term_master_df

# アセットロード
faiss_index, meddra_terms, synonym_df, term_master_df = load_assets()
score_cache = load_score_cache()

st.title("💊 MedDRA 用語検索アプリ")
query = st.text_input("症状を入力してください（日本語）", "")

if st.button("🔍 検索実行") and query:
    st.write("🔄 OpenAIを使って検索語を拡張中...")
    predicted_keywords = expand_query_gpt(query)
    st.write("🔑 拡張キーワード:", format_keywords(predicted_keywords))

    with st.spinner("FAISSで用語検索中..."):
        search_results = []
        for kw in predicted_keywords:
            result = search_meddra(kw, faiss_index, meddra_terms, synonym_df, top_k=20)
            search_results.append(result)
        all_results = pd.concat(search_results).drop_duplicates(subset=["term"]).reset_index(drop=True)

    with st.spinner("OpenAIで再スコアリング中..."):
        all_results = rerank_results_v13(query, all_results, score_cache)

    with st.spinner("MedDRA階層情報を追加中..."):
        all_results = add_hierarchy_info(all_results, term_master_df)

    st.success("検索完了！")
    st.dataframe(all_results)

    save_score_cache(score_cache)
