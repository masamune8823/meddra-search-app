import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from helper_functions import (
    encode_query,
    search_meddra,
    rerank_results_v13,
    add_hierarchy_info,
    rescale_scores,
    predict_soc_category,
    format_keywords,
)

# ---------------- 初期設定 ---------------- #
st.set_page_config(page_title="MedDRA検索アプリ", page_icon="🔍")
st.title("\U0001f50d MedDRA検索アプリ")

# ---------------- ファイル読み込み ---------------- #
@st.cache_resource
def load_assets():
    with open("search_assets_part_a", "rb") as f:
        faiss_index = pickle.load(f)
    with open("search_assets_part_b", "rb") as f:
        meddra_terms = pickle.load(f)
    with open("search_assets_part_c", "rb") as f:
        synonym_df = pickle.load(f)
    with open("search_assets_part_d", "rb") as f:
        term_master_df = pickle.load(f)
    return faiss_index, meddra_terms, synonym_df, term_master_df

faiss_index, meddra_terms, synonym_df, term_master_df = load_assets()

# ---------------- ユーザー入力 ---------------- #
query = st.text_input("検索語を入力してください（例：皮膚がかゆい）", value="ズキズキ")
use_soc_filter = st.checkbox("GPTによるSOC予測でフィルタリング（推奨）", value=True)

# ---------------- 検索処理 ---------------- #
if st.button("検索"):
    if not query.strip():
        st.warning("検索語を入力してください。")
    else:
        with st.spinner("キーワードを解析中..."):
            predicted_keywords = predict_soc_category(query)
            st.subheader("\ud83e\udd13 GPT予測キーワード（整形後）")
            st.write(predicted_keywords)

        with st.spinner("FAISSで用語検索中..."):
            search_results = []
            for kw in predicted_keywords:
                result = search_meddra(kw, faiss_index, meddra_terms, synonym_df, top_k=20)
                search_results.append(result)
            all_results = pd.concat(search_results).drop_duplicates(subset=["term"]).reset_index(drop=True)

        with st.spinner("再スコアリング中（GPT）..."):
            reranked = rerank_results_v13(query, all_results)
            reranked = rescale_scores(reranked)

        with st.spinner("階層情報を付加中..."):
            final_results = add_hierarchy_info(reranked, term_master_df)

        if use_soc_filter:
            try:
                soc_prediction = predict_soc_category(query)
                final_results = final_results[final_results["SOC"].isin(soc_prediction)]
            except Exception as e:
                st.warning(f"フィルタ処理でエラーが発生しました: {e}")

        st.success("検索完了")
        st.dataframe(final_results[["term", "score", "HLT", "HLGT", "SOC"]].rename(columns={"term": "用語", "score": "確からしさ (%)"}))

        csv = final_results.to_csv(index=False).encode("utf-8")
        st.download_button("\ud83d\udcc6 結果をCSVでダウンロード", data=csv, file_name="meddra_results.csv", mime="text/csv")
        
        # 🔍 テスト用ボタン（← ここが追記部分）
        if st.button("🔍 テスト実行（ズキズキ）"):
            from test_meddra_full_pipeline import run_test_pipeline
            run_test_pipeline()
        # updated

