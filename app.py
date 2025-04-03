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
    predict_soc_keywords_with_gpt,
)

# --- ファイルパスの定義 ---
index_path = "faiss_index.index"
terms_path = "meddra_terms.npy"
embed_path = "meddra_embeddings.npy"
synonym_path = "synonym_df_cat1.pkl"
term_master_path = "term_master_df.pkl"

# --- データの読み込み ---
try:
    faiss_index = faiss.read_index(index_path)
    meddra_terms = np.load(terms_path, allow_pickle=True)
    meddra_embeddings = np.load(embed_path)
    with open(synonym_path, "rb") as f:
        synonym_df = pickle.load(f)
    with open(term_master_path, "rb") as f:
        term_master_df = pickle.load(f)
except Exception as e:
    st.error(f"初期データの読み込みに失敗しました: {e}")
    st.stop()

# --- Streamlit UI ---
st.title("🔍 MedDRA検索アプリ")

query = st.text_input("検索語を入力してください（例：皮膚がかゆい）", "")

use_filter = st.checkbox("🧠 GPTによるSOC予測でフィルタリング（推奨）", value=False)

if st.button("検索") and query:
    with st.spinner("検索中..."):

        # 🔍 類似検索（synonym展開込み）
        results = search_meddra(query, faiss_index, meddra_terms, synonym_df, top_k=20)

        if results.empty:
            st.warning("候補が見つかりませんでした。")
            st.stop()

        # 🎯 GPT再スコアリング（Top10）
        reranked = rerank_results_v13(query, results, top_n=10)

        # 🧱 階層情報の追加（term_master_df使用）
        final_results = add_hierarchy_info(reranked, term_master_df)

        # 🧠 GPTによるSOCキーワード予測でフィルタリング（任意）
        if use_filter:
            try:
                predicted_keywords = predict_soc_keywords_with_gpt(query)
                st.markdown("#### 🧠 GPT予測キーワード（整形後）")
                st.write(predicted_keywords)

                filter_cols = ["SOC_Japanese", "HLGT_Japanese", "HLT_Japanese"]
                filter_cols = [col for col in filter_cols if col in final_results.columns]

                if filter_cols:
                    mask = final_results[filter_cols].astype(str).apply(
                        lambda x: x.str.contains("|".join(predicted_keywords)), axis=1
                    )
                    final_results = final_results[mask]
                else:
                    st.warning("階層情報が不足しているため、フィルタリングをスキップしました。")
            except Exception as e:
                st.warning(f"フィルタ処理でエラーが発生しました: {e}")

        # 🔢 スコア整形（0〜100％）
        final_results = rescale_scores(final_results)

        # 📋 表示整形
        final_results = final_results.rename(columns={
            "term": "用語",
            "確からしさ（％）": "確からしさ（％）",
            "HLT_Japanese": "HLT",
            "HLGT_Japanese": "HLGT",
            "SOC_Japanese": "SOC"
        })

        display_cols = [col for col in ["用語", "確からしさ（％）", "HLT", "HLGT", "SOC"] if col in final_results.columns]

        st.success("検索完了")
        st.dataframe(final_results[display_cols], use_container_width=True)

        # 📥 CSVダウンロード
        csv = final_results[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button("📥 結果をCSVでダウンロード", data=csv, file_name="meddra_results.csv", mime="text/csv")
