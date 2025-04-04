import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import faiss

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
    try:
        faiss_index = faiss.read_index("faiss_index.index")
    except Exception as e:
        st.error(f"FAISSインデックスの読み込みに失敗しました: {e}")
        raise e

    try:
        meddra_terms = np.load("meddra_terms.npy", allow_pickle=True)
        synonym_df = pickle.load(open("synonym_df_cat1.pkl", "rb"))
        term_master_df = pickle.load(open("term_master_df.pkl", "rb"))
    except Exception as e:
        st.error(f"データファイルの読み込みに失敗しました: {e}")
        raise e

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
            st.subheader("🧠 GPT予測キーワード（整形後）")
            st.write(predicted_keywords)

        with st.spinner("FAISSで用語検索中..."):
            search_results = []
            for kw in predicted_keywords:
                result = search_meddra(kw, faiss_index, meddra_terms, synonym_df, top_k=20)
                search_results.append(result)
            all_results = pd.concat(search_results).drop_duplicates(subset=["term"]).reset_index(drop=True)

        with st.spinner("再スコアリング中（GPT）..."):
            reranked = rerank_results_v13(query, all_results)
            reranked["score"] = rescale_scores(reranked["Relevance"].tolist())

        with st.spinner("階層情報を付加中..."):
            st.write("列名チェック（reranked）:", reranked.columns.tolist())  # ← ここ追加
            final_results = add_hierarchy_info(reranked, term_master_df)

        if use_soc_filter:
            try:
                soc_prediction = predict_soc_category(query)
                final_results = final_results[final_results["SOC"].isin(soc_prediction)]
            except Exception as e:
                st.warning(f"フィルタ処理でエラーが発生しました: {e}")

        st.success("検索完了")

        expected_cols = ["term", "score", "HLT", "HLGT", "SOC"]
        available_cols = [col for col in expected_cols if col in final_results.columns]

        # 表示
        st.dataframe(
              final_results[available_cols].rename(columns={"term": "用語", "score": "確からしさ (%)"})
        )

        # CSV生成時に encoding を指定する
        csv = final_results.to_csv(index=False, encoding="utf-8-sig")

        # ダウンロードボタン
        st.download_button("📆 結果をCSVでダウンロード", data=csv, file_name="meddra_results.csv", mime="text/csv")
        
        # 🔍 テスト用ボタン（← ここが追記部分）
        if st.button("🔍 テスト実行（ズキズキ）"):
            from test_meddra_full_pipeline import run_test_pipeline
            run_test_pipeline()
        # updated

