
import streamlit as st
import pandas as pd
import numpy as np
import os
import faiss
import pickle

# --- ファイルパスの定義 ---
DATA_DIR = "./"  # Streamlit Cloud 用（/mnt/data は Colab 用）
index_path = os.path.join(DATA_DIR, "faiss_index.index")
terms_path = os.path.join(DATA_DIR, "meddra_terms.npy")
embed_path = os.path.join(DATA_DIR, "meddra_embeddings.npy")
synonym_path = os.path.join(DATA_DIR, "synonym_df_cat1.pkl")
hierarchy_path = os.path.join(DATA_DIR, "term_master_df.pkl")

# --- データの読み込み ---
try:
    with open(synonym_path, "rb") as f:
        synonym_df = pickle.load(f)
except Exception as e:
    synonym_df = None
    st.warning(f"同義語辞書の読み込みに失敗しました: {e}")

try:
    with open(hierarchy_path, "rb") as f:
        term_master_df = pickle.load(f)
except Exception as e:
    term_master_df = None
    st.warning(f"term_master_df の読み込みに失敗しました: {e}")

meddra_terms = np.load(terms_path, allow_pickle=True)
meddra_embeddings = np.load(embed_path)
faiss_index = faiss.read_index(index_path)

# --- 検索機能の読み込み ---
from helper_functions import (
    encode_query,
    search_meddra,
    rerank_results_v13,
    add_hierarchy_info,
    rescale_scores,
    predict_soc_keywords_with_gpt,
)

# --- Streamlit UI ---
st.title("🔍 MedDRA検索アプリ")
query = st.text_input("検索語を入力してください（例：皮膚がかゆい）", "")
use_filter = st.checkbox("GPTによるSOC予測でフィルタリング（推奨）", value=False)

if st.button("検索") and query:
    with st.spinner("検索中..."):

        # 🔍 類義語 + FAISS検索
        results = search_meddra(query, faiss_index, meddra_terms, synonym_df, top_k=20)

        # 🎯 GPTベースの再スコアリング
        reranked = rerank_results_v13(query, results, top_n=10)

        # 🧱 階層情報を追加（フィルタ前に必須）
        final_results = add_hierarchy_info(reranked, term_master_df)

        # 🧠 GPTによるSOCフィルタリング
        if use_filter:
            try:
                predicted_keywords = predict_soc_keywords_with_gpt(query)
                final_results = final_results[
                    final_results[["SOC", "HLGT", "HLT"]].astype(str).apply(
                        lambda x: x.str.contains("|".join(predicted_keywords)).any(), axis=1
                    )
                ]
            except Exception as e:
                st.warning(f"フィルタ処理でエラーが発生しました: {e}")

        # 🔢 スコア再スケーリング
        final_results = rescale_scores(final_results)

        # 📋 表示整形
        final_results = final_results.rename(columns={
            "term": "用語",
            "score": "確からしさ（％）",
            "HLT_Japanese": "HLT",
            "HLGT_Japanese": "HLGT",
            "SOC_Japanese": "SOC"
        })[["用語", "確からしさ（％）", "HLT", "HLGT", "SOC"]]

        st.success("検索完了")
        st.dataframe(final_results, use_container_width=True)

        # 📎 CSVダウンロード
        csv = final_results.to_csv(index=False).encode("utf-8")
        st.download_button("📥 検索結果をCSVでダウンロード", data=csv, file_name="meddra_results.csv", mime="text/csv")
