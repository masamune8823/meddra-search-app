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

        # 📊 結果表示（整形済DataFrameを前提とする）

        # 📊 結果表示（整形済DataFrameを前提とする）

        # 📊 結果表示（整形済DataFrameを前提とする）
        st.subheader("🔎 検索結果（スコア順）")

        if not reranked.empty:
            # 列名を日本語に変換（念のため再確認）
            reranked = reranked.rename(columns={
                "term": "用語",
                "score": "確からしさ（％）",
                "HLT_Japanese": "HLT",
                "HLGT_Japanese": "HLGT",
                "SOC_Japanese": "SOC"
            })

            # スコア降順にソート
            sorted_df = reranked.sort_values(by="確からしさ（％）", ascending=False).reset_index(drop=True)

            # 存在する列だけ抽出
            display_columns = [col for col in ["用語", "確からしさ（％）", "HLT", "HLGT", "SOC"] if col in sorted_df.columns]

            # 表示（存在する列のみ）
            st.dataframe(sorted_df[display_columns])

            # ダウンロードボタン
            csv_download = sorted_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="結果をCSVで保存",
                data=csv_download,
                file_name="meddra_results.csv",
                mime="text/csv"
            )
        else:
            st.warning("検索結果がありません。")
        if not reranked.empty:
            # 列名を日本語に変換（念のため再確認）
            reranked = reranked.rename(columns={
                "term": "用語",
                "score": "確からしさ（％）",
                "HLT_Japanese": "HLT",
                "HLGT_Japanese": "HLGT",
                "SOC_Japanese": "SOC"
            })

            # スコア降順にソート
            sorted_df = reranked.sort_values(by="確からしさ（％）", ascending=False).reset_index(drop=True)

            # 表示（列の並びを明示）
            st.dataframe(sorted_df[["用語", "確からしさ（％）", "HLT", "HLGT", "SOC"]])

            # ダウンロードボタン
            csv_download = sorted_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="結果をCSVで保存",
                data=csv_download,
                file_name="meddra_results.csv",
                mime="text/csv"
            )
        else:
            st.warning("検索結果がありません。")
        if not reranked.empty:
            # スコア降順にソート
            sorted_df = reranked.sort_values(by="確からしさ（％）", ascending=False).reset_index(drop=True)

            # 表示（列の並びを明示）
            st.dataframe(sorted_df[["用語", "確からしさ（％）", "HLT", "HLGT", "SOC"]])

            # ダウンロードボタン
            csv_download = sorted_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="結果をCSVで保存",
                data=csv_download,
                file_name="meddra_results.csv",
                mime="text/csv"
            )
        else:
            st.warning("検索結果がありません。")
        # 💾 ダウンロードリンク
        csv = reranked.to_csv(index=False).encode("utf-8-sig")
        st.download_button("結果をCSVでダウンロード", data=csv, file_name="meddra_results.csv", mime="text/csv")
