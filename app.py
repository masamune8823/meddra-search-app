import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
import os
from helper_functions import expand_query_gpt, encode_query, rerank_results_v13

# =======================
# STEP 1: データ読み込み
# =======================
@st.cache_resource
def load_data():
    # search_assets の分割ファイルを読み込んで結合
    base_path = "."
    parts = []
    for part_name in ["search_assets_part_a", "search_assets_part_b", "search_assets_part_c", "search_assets_part_d"]:
        with open(os.path.join(base_path, part_name), "rb") as f:
            parts.append(pickle.load(f))
    search_assets = {}
    for part in parts:
        search_assets.update(part)

    # ベクトルとFAISSインデックス
    vectors = np.load("meddra_embeddings.npy")
    faiss_index = faiss.read_index("faiss_index.index")

    # term_masterとシソーラス
    with open("term_master_df.pkl", "rb") as f:
        term_master_df = pickle.load(f)
    with open("synonym_df_cat1.pkl", "rb") as f:
        synonym_df_cat1 = pickle.load(f)

    return search_assets, vectors, faiss_index, term_master_df, synonym_df_cat1

search_assets, vectors, faiss_index, term_master_df, synonym_df_cat1 = load_data()

# =======================
# STEP 2: Streamlit UI
# =======================
st.title("🧠 MedDRA 用語検索アプリ")
user_input = st.text_input("症状や訴えを自然文で入力してください", "")

if user_input:
    with st.spinner("検索中..."):
        # 拡張クエリ生成
        expanded_terms = expand_query_gpt(user_input, synonym_df_cat1)
        
        # クエリベクトル化＆FAISS検索
        top_results = []
        for term in expanded_terms:
            query_vec = encode_query(term)
            D, I = faiss_index.search(query_vec, 10)  # 上位10件
            for score, idx in zip(D[0], I[0]):
                top_results.append({
                    "term": search_assets[idx],
                    "score": float(score),
                    "query": term
                })

        # DataFrame化して再スコアリング
        result_df = pd.DataFrame(top_results)
        reranked_df = rerank_results_v13(user_input, result_df)

        # 階層情報付加
        final_df = pd.merge(reranked_df, term_master_df, left_on="term", right_on="PT_English", how="left")

        # 表示
        st.subheader("🔎 検索結果（確からしさスコア順）")
        st.dataframe(final_df[["term", "score", "query", "PT_Japanese", "HLT_Japanese", "HLGT_Japanese", "SOC_Japanese"]])
