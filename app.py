import streamlit as st
import pandas as pd
import numpy as np
import openai
import os
import faiss

from helper_functions import (
    search_meddra,
    rerank_results_v13,
    predict_soc_keywords_with_gpt,
    filter_by_predicted_soc,
    rescale_scores,
    add_hierarchy_info
)

# 🔑 OpenAI APIキーの取得（環境変数経由）
openai.api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")

# 📥 ユーザー入力
st.title("💊 MedDRA 用語検索アプリ")
query = st.text_input("症状を入力してください（例：「ズキズキ」「皮膚がかゆい」など）")

# ✅ オプション選択
use_filter = st.checkbox("GPTによるSOCフィルタを適用する", value=True)

# 🔍 各種検索リソース読み込み
faiss_index = faiss.read_index("faiss_index.index")
meddra_terms = np.load("meddra_terms.npy", allow_pickle=True)
synonym_df = pd.read_pickle("synonym_df_cat1.pkl")
term_master_df = pd.read_pickle("term_master_df.pkl")

# 🚀 検索ボタン押下時の処理
if st.button("検索") and query:
    with st.spinner("検索中..."):

        # 🔍 MedDRA検索（synonym_df + FAISS）
        results = search_meddra(query, faiss_index, meddra_terms, synonym_df, top_k=20)

        # 🧱 MedDRA階層を付与
        final_results = add_hierarchy_info(results, term_master_df)

        # 📊 GPTで関連SOCカテゴリを予測し、フィルタリング
        if use_filter:
            predicted_keywords = predict_soc_keywords_with_gpt(query)
            final_results = filter_by_predicted_soc(final_results, predicted_keywords)

        # 🎯 スコア再スケーリング
        final_results = rescale_scores(final_results)

        # 💬 GPT再スコアリング（Top10件）
        reranked = rerank_results_v13(query, final_results, top_n=10)
        reranked = add_hierarchy_info(reranked, term_master_df)
        reranked = rescale_scores(reranked)

        # 📊 結果表示
        st.subheader("🔎 検索結果（Top 10）")
        st.dataframe(reranked)

        # 💾 ダウンロードリンク
        csv = reranked.to_csv(index=False).encode("utf-8-sig")
        st.download_button("結果をCSVでダウンロード", data=csv, file_name="meddra_results.csv", mime="text/csv")
