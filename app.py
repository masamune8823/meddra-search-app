
import streamlit as st
import pandas as pd
import os
import openai

from helper_functions import (
    search_meddra,
    rerank_results_v13,
    load_term_master_df,
    predict_soc_keywords_with_gpt,
    filter_by_predicted_soc,
    rescale_scores
)

st.set_page_config(page_title="MedDRA検索アプリ", layout="wide")
st.title("💊 MedDRA 自然言語検索アプリ")
st.markdown("自然文から、ぴったりのMedDRA PT用語を探します🧠")

query = st.text_input("📝 症状を入力（例：ズキズキ、吐き気）")
apply_gpt_filter = st.checkbox("🤖 GPTフィルタで意味的に関連ある用語だけに絞る", value=True)
apply_rescale = st.checkbox("🎯 スコアを0〜100％に補正して表示", value=True)

openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai

term_master_path = "term_master_df.pkl"
if os.path.exists(term_master_path):
    term_master_df = pd.read_pickle(term_master_path)
else:
    st.error("❌ term_master_df.pkl が見つかりません")
    st.stop()

if st.button("🚀 検索する") and query.strip():
    with st.spinner("おまかせ検索中...⏳"):
        raw_results = search_meddra(query, top_k_per_method=5)
        reranked = rerank_results_v13(raw_results, query, top_n=10)

        df = pd.DataFrame(reranked, columns=["term", "score", "source"])
        df = df.merge(
            term_master_df[["PT_Japanese", "HLT_Japanese", "HLGT_Japanese", "SOC_Japanese"]],
            how="left", left_on="term", right_on="PT_Japanese"
        )
        df = df.rename(columns={
            "HLT_Japanese": "HLT", "HLGT_Japanese": "HLGT", "SOC_Japanese": "SOC"
        })
        df = df[["term", "score", "HLT", "HLGT", "SOC", "source"]]

        if apply_gpt_filter:
            soc_keywords = predict_soc_keywords_with_gpt(query, client)
            df = filter_by_predicted_soc(df, soc_keywords)

        if apply_rescale and not df.empty:
            df = rescale_scores(df, score_col="score")

    if not df.empty:
        st.success("✅ 検索完了！")
        st.dataframe(df[["term", "確からしさ（％)", "HLT", "HLGT", "SOC", "source"]], use_container_width=True)
        csv = df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("📥 CSVでダウンロード", csv, file_name=f"meddra_results_{query}.csv")
    else:
        st.warning("結果が見つかりませんでした。もう少し具体的に入力してみてください。")
