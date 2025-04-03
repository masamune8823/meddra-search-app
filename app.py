
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
)

# ✅ OpenAI APIキーは環境変数から取得
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai

# 🔍 各種検索リソース読み込み
faiss_index = faiss.read_index("faiss_index.index")
faiss_index_synonym = faiss.read_index("faiss_index_synonym.index")
meddra_terms = np.load("meddra_terms.npy", allow_pickle=True)
synonym_df = pd.read_pickle("synonym_df_cat1.pkl")
term_master_df = pd.read_pickle("term_master_df.pkl")

# ✅ Streamlit UI設定
st.set_page_config(page_title="MedDRA検索アプリ", layout="wide")
st.title("💊 MedDRA 自然言語検索アプリ")
st.markdown("医師記載語・口語に対応したMedDRA PT検索支援ツール")

query = st.text_input("📝 症状を入力してください（例：ズキズキ、皮膚がかゆい）")
use_gpt_filter = st.checkbox("🤖 意味的フィルタ（GPTベース）を使う", value=True)
use_score_rescale = st.checkbox("🎯 スコアを0〜100％に補正して表示", value=True)

if st.button("🚀 検索する") and query:
    with st.spinner("検索中です...しばらくお待ちください"):
        raw_results = search_meddra(
            query=query,
            faiss_index=faiss_index,
            faiss_index_synonym=faiss_index_synonym,
            synonym_df=synonym_df,
            meddra_terms=meddra_terms
        )
        reranked = rerank_results_v13(
            raw_results, query=query, client=client, top_n=10
        )

        df = pd.DataFrame(reranked, columns=["term", "score", "source"])
        df = df.merge(
            term_master_df[["PT_Japanese", "HLT_Japanese", "HLGT_Japanese", "SOC_Japanese"]],
            how="left",
            left_on="term",
            right_on="PT_Japanese"
        ).rename(columns={
            "HLT_Japanese": "HLT",
            "HLGT_Japanese": "HLGT",
            "SOC_Japanese": "SOC"
        })

        df = df[["term", "score", "HLT", "HLGT", "SOC", "source"]]

        if use_gpt_filter:
            soc_keywords = predict_soc_keywords_with_gpt(query, client)
            df = filter_by_predicted_soc(df, soc_keywords)

        if use_score_rescale and not df.empty:
            df = rescale_scores(df, score_col="score")

    if not df.empty:
        st.success("✅ 検索完了！")
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("📥 結果をCSVでダウンロード", csv, file_name=f"meddra_results_{query}.csv")
    else:
        st.warning("結果が見つかりませんでした。もう少し具体的な表現で試してみてください。")
