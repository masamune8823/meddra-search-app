import streamlit as st
import pandas as pd
import numpy as np
import os
import faiss
import pickle
import re

from helper_functions import (
    encode_query,
    search_meddra,
    rerank_results_v13,
    add_hierarchy_info,
    rescale_scores,
    predict_soc_keywords_with_gpt,
    clean_predicted_keywords
)

# --- ファイルパスの定義 ---
DATA_DIR = "/mnt/data"
index_path = os.path.join(DATA_DIR, "faiss_index.index")
terms_path = os.path.join(DATA_DIR, "meddra_terms.npy")
embed_path = os.path.join(DATA_DIR, "meddra_embeddings.npy")
synonym_path = os.path.join(DATA_DIR, "synonym_df_cat1.pkl")
term_master_path = os.path.join(DATA_DIR, "term_master_df.pkl")

# --- データの読み込み ---
with open(synonym_path, "rb") as f:
    synonym_df = pickle.load(f)

with open(term_master_path, "rb") as f:
    term_master_df = pickle.load(f)

meddra_terms = np.load(terms_path, allow_pickle=True)
meddra_embeddings = np.load(embed_path)
faiss_index = faiss.read_index(index_path)

# --- Streamlit UI ---
st.title("🔍 MedDRA検索アプリ（日本語シノニム対応）")

query = st.text_input("検索語を入力してください（例：皮膚がかゆい）", "")
use_filter = st.checkbox("GPTによるSOC予測でフィルタリング（推奨）", value=False)

if st.button("検索") and query:
    with st.spinner("検索中..."):
        # 🔍 検索（synonym_df + FAISS検索）
        results = search_meddra(query, faiss_index, meddra_terms, synonym_df, top_k=20)

        if results.empty:
            st.warning("検索結果が見つかりませんでした。")
        else:
            # 🎯 Top10件を再スコアリング（GPTベース）
            reranked = rerank_results_v13(query, results, top_n=10)

            # 🧱 階層情報の付加（HLT/HLGT/SOC）
            final_results = add_hierarchy_info(reranked, term_master_df)

            # 📊 GPTで関連SOCカテゴリを予測し、フィルタリング
            if use_filter:
                predicted_keywords = predict_soc_keywords_with_gpt(query)
                predicted_keywords = clean_predicted_keywords(predicted_keywords)
                try:
                    final_results = final_results[
                        final_results[["SOC_Japanese", "HLGT_Japanese", "HLT_Japanese"]]
                        .astype(str)
                        .apply(lambda x: x.str.contains("|".join(predicted_keywords)).any(), axis=1)
                    ]
                except KeyError:
                    st.warning("フィルタリング対象の列が見つかりませんでした")

            # 🔢 スコア再スケーリング（％表示）
            final_results = rescale_scores(final_results)

            # 📋 表示整形
            final_results = final_results.rename(columns={
                "term": "用語",
                "score": "確からしさ（％）",
                "HLT_Japanese": "HLT",
                "HLGT_Japanese": "HLGT",
                "SOC_Japanese": "SOC"
            })

            columns_to_show = [c for c in ["用語", "確からしさ（％）", "HLT", "HLGT", "SOC"] if c in final_results.columns]
            st.success("検索完了")
            st.dataframe(final_results[columns_to_show], use_container_width=True)

            csv = final_results.to_csv(index=False).encode("utf-8")
            st.download_button("📥 結果をCSVでダウンロード", data=csv, file_name="meddra_results.csv", mime="text/csv")
