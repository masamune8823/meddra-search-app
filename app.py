
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import openai
import os

from helper_functions import (
    search_meddra,
    rerank_results_v13,
    predict_soc_keywords_with_gpt,
    filter_by_predicted_soc,
    rescale_scores,
    add_hierarchy_info,
)

# OpenAI APIキーは環境変数から取得
openai.api_key = os.getenv("OPENAI_API_KEY", "your-api-key")

# FAISSインデックスと用語データの読み込み
faiss_index = faiss.read_index("faiss_index.index")
meddra_terms = np.load("meddra_terms.npy", allow_pickle=True)
synonym_df = pd.read_pickle("synonym_df_cat1.pkl")
term_master_df = pd.read_pickle("term_master_df.pkl")

# Streamlit UI
st.title("🔍 MedDRA用語検索ツール")
query = st.text_input("症状や訴えを入力してください（例：ズキズキする痛み）")
use_filter = st.checkbox("関連カテゴリで絞り込み（SOCフィルタ）", value=True)

if st.button("検索") and query:
    with st.spinner("検索中..."):
        # MedDRA検索（synonym_df + FAISS）
        results = search_meddra(query, faiss_index, meddra_terms, synonym_df, top_k=20)

        # Top10件を再スコアリング（GPTベース）
        reranked = rerank_results_v13(query, results, top_n=10)

        # 階層情報の付加
        final_results = add_hierarchy_info(reranked, term_master_df)

        # GPTで関連SOCカテゴリを予測し、フィルタリング
        if use_filter:
            predicted_keywords = predict_soc_keywords_with_gpt(query)
            final_results = filter_by_predicted_soc(final_results, predicted_keywords)

        # スコア再スケーリング
        final_results = rescale_scores(final_results)

        # 結果表示
        st.success("検索完了！")
        st.dataframe(final_results)

        # CSV出力
        csv = final_results.to_csv(index=False).encode("utf-8-sig")
        st.download_button("結果をCSVでダウンロード", csv, "meddra_results.csv", "text/csv")

