# app.py（シノニム統合版）
import streamlit as st
import pandas as pd
import pickle
from utils import expand_query_gpt, encode_query, rerank_results_v13, match_synonyms, merge_faiss_and_synonym_results
import numpy as np
import faiss
import os

# --- データロード ---
@st.cache_resource
def load_faiss_index():
    index = faiss.read_index("meddra_index.faiss")
    return index

@st.cache_data
def load_data():
    terms = np.load("meddra_terms.npy", allow_pickle=True)
    embeddings = np.load("meddra_embeddings.npy")
    synonym_df = pd.read_pickle("synonym_df_cat1.pkl")  # あらかじめ保存されたシノニム辞書
    return terms.tolist(), embeddings, synonym_df

# --- FAISS検索 ---
def search_faiss(query_vec, embeddings, index, top_k=10):
    D, I = index.search(np.array([query_vec]).astype("float32"), top_k)
    return I[0]

# --- メインUI ---
st.title("MedDRA検索アプリ")
st.write("症状や記述を入力してください")

user_query = st.text_input("症状入力", "頭痛")

if st.button("検索") and user_query:
    with st.spinner("検索中..."):
        # データ読み込み
        terms, embeddings, synonym_df = load_data()
        index = load_faiss_index()

        # クエリ拡張
        expanded_terms = expand_query_gpt(user_query)

        # 拡張語ごとにFAISS検索
        faiss_results = []
        for term in expanded_terms:
            qvec = encode_query(term)
            idxs = search_faiss(qvec, embeddings, index, top_k=10)
            faiss_results.extend([terms[i] for i in idxs])

        # PTコード抽出
        faiss_pt_codes = list(set([item["pt_code"] for item in faiss_results]))

        # シノニムマッチでPTコード補完
        merged_pt_codes = merge_faiss_and_synonym_results(faiss_pt_codes, user_query, synonym_df)

        # PTコードに対応する候補を再構成
        merged_results = [item for item in faiss_results if item["pt_code"] in merged_pt_codes]

        # 再ランキング
        reranked = rerank_results_v13(merged_results)

        # 結果表示
        st.write("### 検索結果：")
        for item in reranked:
            st.write(f"{item['pt_code']}. {item['pt_japanese']}")
            st.write(f"確からしさ: {item['score']}%")
            st.write(f"HLT: {item['hlt']} | HLGT: {item['hlgt']} | SOC: {item['soc']}")
            st.markdown("---")
