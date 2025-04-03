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
    add_hierarchy_info
)

# 🔐 OpenAI APIキー（環境変数から取得）
openai.api_key = os.getenv("OPENAI_API_KEY")

# 📦 各種ファイル読み込み
faiss_index = faiss.read_index("faiss_index.index")
faiss_index_synonym = faiss.read_index("faiss_index_synonym.index")
meddra_terms = np.load("meddra_terms.npy", allow_pickle=True)
synonym_df = pd.read_pickle("synonym_df_cat1.pkl")
term_master_df = pd.read_pickle("term_master_df.pkl")

# 🌐 Streamlitページ設定
st.set_page_config(page_title="MedDRA 自然言語検索システム", layout="wide")
st.title("💊 MedDRA 自然言語検索システム")
st.write("自然文から適切なMedDRA PT用語を検索します。")
query = st.text_input("🔍 症状や状態を入力してください（例：ズキズキする、吐き気など）")

# 🔘 オプション
apply_soc_filter = st.checkbox("関連するSOCカテゴリでフィルタする", value=True)
rescale_score = st.checkbox("確からしさスコアを0-100に補正する", value=True)

# 🚀 検索ボタン
if st.button("検索実行") and query:

    # 🔍 検索（synonym + FAISS + 再スコア）
    results = search_meddra(
        query=query,
        faiss_index=faiss_index,
        faiss_index_synonym=faiss_index_synonym,
        meddra_terms=meddra_terms,
        synonym_df=synonym_df
    )

    # ✅ スコア順にソート
    reranked = rerank_results_v13(query, results)

    # 🧠 LLMベースでSOC予測 → フィルタ
    if apply_soc_filter:
        predicted_keywords = predict_soc_keywords_with_gpt(query)
        reranked = filter_by_predicted_soc(reranked, predicted_keywords)

    # 🎯 スコア再スケーリング
    if rescale_score:
        reranked = rescale_scores(reranked)

    # 🧱 階層情報を付与（PT → HLT → HLGT → SOC）
    final_results = add_hierarchy_info(reranked, term_master_df)

    # 📊 結果表示
    st.dataframe(final_results[["term", "確からしさ（％）", "HLT", "HLGT", "SOC", "source"]], use_container_width=True)

    # 💾 ダウンロードリンク
    csv = final_results.to_csv(index=False)
    st.download_button("📥 検索結果をCSVでダウンロード", csv, file_name="meddra_search_results.csv", mime="text/csv")
