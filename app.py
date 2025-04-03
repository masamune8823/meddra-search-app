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
    add_hierarchy_info,
    rescale_scores,
)

# OpenAI APIキー（環境変数から取得）
openai.api_key = os.getenv("OPENAI_API_KEY")

# 🔍 各種検索リソース読み込み
faiss_index = faiss.read_index("faiss_index.index")
meddra_terms = np.load("meddra_terms.npy", allow_pickle=True)
synonym_df = pd.read_pickle("synonym_df_cat1.pkl")
term_master_df = pd.read_pickle("term_master_df.pkl")

# 💡 Streamlit UI
st.set_page_config(layout="wide")
st.title("💊 MedDRA 自然言語検索システム")
query = st.text_input("症状や状態を入力してください（例：ズキズキする、吐き気など）")

if st.button("検索実行") and query:
    with st.spinner("検索中..."):
        # Step 1: synonym + FAISS検索
        initial_df = search_meddra(query, faiss_index, meddra_terms, synonym_df)

        # Step 2: GPT再スコアリング
        reranked_df = rerank_results_v13(query, initial_df)

        # Step 3: 階層情報付加
        enriched_df = add_hierarchy_info(reranked_df, term_master_df)

        # Step 4: SOC予測とフィルタ（任意）
        soc_keywords = predict_soc_keywords_with_gpt(query)
        filtered_df = filter_by_predicted_soc(enriched_df, soc_keywords)

        # Step 5: スコア整形
        final_df = rescale_scores(filtered_df)

        # 表示・出力
        st.write("🔍 検索結果", final_df[["term", "確からしさ（％）", "HLT", "HLGT", "SOC", "source"]])
        csv = final_df.to_csv(index=False)
        st.download_button("検索結果をCSVでダウンロード", csv, file_name="search_results.csv", mime="text/csv")
