import streamlit as st
import numpy as np
import pickle
import faiss
import os

from helper_functions import expand_query_gpt, encode_query, rerank_results_v13

@st.cache_resource
def load_faiss_index():
    index = faiss.read_index("faiss_index.index")
    return index

@st.cache_resource
def load_embeddings():
    return np.load("meddra_embeddings.npy")

@st.cache_resource
def load_term_master():
    with open("term_master_df.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_synonym_dict():
    with open("synonym_df_cat1.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_search_assets():
    assets = []
    for part in ['a', 'b', 'c', 'd']:
        filename = f"search_assets_part_{part}"
        with open(filename, "rb") as f:
            assets.append(pickle.load(f))
    return assets

# アセットの読み込み
index = load_faiss_index()
embeddings = load_embeddings()
term_master_df = load_term_master()
synonym_df_cat1 = load_synonym_dict()
search_assets = load_search_assets()

# Streamlit UI
st.set_page_config(page_title="MedDRA検索アプリ", layout="centered")
st.title("💊 MedDRA検索アプリ")
st.write("症状や記述を入力してください")

query = st.text_input("症状入力", value="")

if st.button("検索"):
    if not query.strip():
        st.warning("検索語を入力してください")
    else:
        try:
            # 拡張
            expanded_terms = expand_query_gpt(query, synonym_df_cat1)
            # ベクトル化
            query_vector = encode_query(expanded_terms)
            # FAISS検索
            D, I = index.search(query_vector, k=10)
            # 候補取得
            candidates = [search_assets[i] for i in I[0]]
            # 再ランキング
            reranked_df = rerank_results_v13(query, candidates, term_master_df)
            # 表示
            st.dataframe(reranked_df)

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
