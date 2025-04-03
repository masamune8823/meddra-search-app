
import streamlit as st
import pandas as pd
import os
import openai

from helper_functions import (
    search_meddra,
    rerank_results_v13,
    load_term_master_df,
    predict_soc_keywords_with_gpt,
    filter_by_predicted_soc
)

st.set_page_config(page_title="MedDRA検索アプリ", layout="wide")
st.title("💊 MedDRA 自然言語検索システム")
st.write("自然文から適切なMedDRA PT用語を検索します。")

query = st.text_input("🔍 症状や状態を入力してください（例：ズキズキ、吐き気 など）")
enable_soc_filter = st.checkbox("GPTによるSOCカテゴリフィルタを有効にする", value=False)

# OpenAIのAPIキー（環境変数で管理 or secrets.toml 推奨）
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai

term_master_path = "term_master_df.pkl"
if os.path.exists(term_master_path):
    term_master_df = pd.read_pickle(term_master_path)
else:
    st.error("❌ term_master_df.pkl が見つかりません")
    st.stop()

if st.button("検索実行") and query.strip():
    with st.spinner("検索中..."):
        raw_results = search_meddra(query, top_k_per_method=5)
        reranked = rerank_results_v13(raw_results, query, top_n=10)

        reranked_df = pd.DataFrame(reranked, columns=["term", "score", "source"])
        merged = reranked_df.merge(
            term_master_df[["PT_Japanese", "HLT_Japanese", "HLGT_Japanese", "SOC_Japanese"]],
            how="left", left_on="term", right_on="PT_Japanese"
        )
        merged = merged.rename(columns={
            "HLT_Japanese": "HLT", "HLGT_Japanese": "HLGT", "SOC_Japanese": "SOC"
        })
        result_df = merged[["term", "score", "HLT", "HLGT", "SOC", "source"]]

        # GPTフィルタ適用（トグル有効時のみ）
        if enable_soc_filter:
            soc_keywords = predict_soc_keywords_with_gpt(query, client)
            result_df = filter_by_predicted_soc(result_df, soc_keywords)

        # スコアの再スケーリング（オプション）
        min_score, max_score = result_df["score"].min(), result_df["score"].max()
        if max_score > min_score:
            result_df["確からしさ（％）"] = ((result_df["score"] - min_score) / (max_score - min_score)) * 100
        else:
            result_df["確からしさ（％）"] = 100
        result_df["確からしさ（％）"] = result_df["確からしさ（％）"].round(1)

    if not result_df.empty:
        st.dataframe(result_df[["term", "確からしさ（％）", "HLT", "HLGT", "SOC", "source"]], use_container_width=True)
        csv = result_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("📥 結果をCSVでダウンロード", csv, file_name=f"meddra_results_{query}.csv")
    else:
        st.warning("該当する結果がありませんでした。")
