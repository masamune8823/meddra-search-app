import streamlit as st
import pandas as pd
import numpy as np
import os
import faiss
import pickle

from helper_functions import (
    encode_query,
    search_meddra,
    rerank_results_v13,
    add_hierarchy_info,
    rescale_scores,
    predict_soc_keywords_with_gpt
)

# --- ファイルパスの定義 ---
index_path = "faiss_index.index"
terms_path = "meddra_terms.npy"
embed_path = "meddra_embeddings.npy"
synonym_path = "synonym_df_cat1.pkl"
hierarchy_path = "term_master_df.pkl"

# --- データの読み込み ---
with open(synonym_path, "rb") as f:
    synonym_df = pickle.load(f)

with open(hierarchy_path, "rb") as f:
    term_master_df = pickle.load(f)

meddra_terms = np.load(terms_path, allow_pickle=True)
meddra_embeddings = np.load(embed_path)
faiss_index = faiss.read_index(index_path)

# --- Streamlit UI ---
st.title("🔍 MedDRA検索アプリ")

query = st.text_input("検索語を入力してください（例：皮膚がかゆい）", "")

use_filter = st.checkbox("GPTによるSOC予測でフィルタリング（推奨）", value=False)

if st.button("検索") and query:
    with st.spinner("検索中..."):

        # 🔍 類似語検索
        results = search_meddra(query, faiss_index, meddra_terms, synonym_df, top_k=20)

        # 🎯 Top10を再スコアリング
        reranked = rerank_results_v13(query, results, top_n=10)

        # 🧱 階層情報を付加
        final_results = add_hierarchy_info(reranked, term_master_df)

        # 📊 GPTで関連SOCカテゴリを予測し、フィルタリング
        if use_filter:
            try:
                predicted_keywords = predict_soc_keywords_with_gpt(query)
                # 日本語列名を使ってフィルタ
                final_results = final_results[
                    final_results[["SOC_Japanese", "HLGT_Japanese", "HLT_Japanese"]]
                    .astype(str)
                    .apply(lambda x: x.str.contains("|".join(predicted_keywords)).any(), axis=1)
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
        })

        show_cols = [col for col in ["用語", "確からしさ（％）", "HLT", "HLGT", "SOC"] if col in final_results.columns]

        st.success("検索完了")
        st.dataframe(final_results[show_cols], use_container_width=True)

        # CSVダウンロード
        csv = final_results[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button("📥 結果をCSVでダウンロード", data=csv, file_name="meddra_results.csv", mime="text/csv")
