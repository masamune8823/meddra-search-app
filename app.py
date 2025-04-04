import streamlit as st
import pandas as pd
import numpy as np
import faiss
import openai
import pickle
import os

from helper_functions import (
    encode_query,
    search_meddra,
    rerank_results_v13,
    add_hierarchy_info,
    rescale_scores,
    predict_soc_keywords_with_gpt,
)

# データファイルの読み込み
terms_path = "meddra_terms.npy"
embed_path = "meddra_embeddings.npy"
index_path = "faiss_index.index"
synonym_path = "synonym_df_cat1.pkl"
hierarchy_path = "term_master_df.pkl"

meddra_terms = np.load(terms_path, allow_pickle=True)
meddra_embeddings = np.load(embed_path)
faiss_index = faiss.read_index(index_path)

with open(synonym_path, "rb") as f:
    synonym_df = pickle.load(f)

with open(hierarchy_path, "rb") as f:
    term_master_df = pickle.load(f)

# Streamlit アプリ本体
st.title("🔍 MedDRA検索アプリ")
query = st.text_input("検索語を入力してください（例：ズキズキ）")
use_filter = st.checkbox("🧠 GPTによるSOC予測でフィルタリング（推奨）", value=True)

if st.button("検索") and query:
    with st.spinner("検索中..."):

        # GPTでSOCカテゴリを予測（必要な場合）
        predicted_keywords = []
        if use_filter:
            try:
                predicted_keywords = predict_soc_keywords_with_gpt(query)
                st.markdown("#### 🧠 GPT予測キーワード（整形後）")
                st.write(predicted_keywords)
            except Exception as e:
                st.warning(f"GPTフィルタ処理中にエラーが発生しました: {e}")

        # FAISS検索（+synonym対応済み）
        search_results = []
        for kw in [query] + predicted_keywords:
            try:
                result = search_meddra(kw, faiss_index, meddra_terms, synonym_df, top_k=20)
                search_results.append(result)
            except Exception as e:
                st.warning(f"検索処理でエラー: {e}")

        if search_results:
            all_results = pd.concat(search_results).drop_duplicates(subset=["term"]).reset_index(drop=True)
        else:
            all_results = pd.DataFrame(columns=["term", "score"])

        # GPT再スコアリング（Top10）
        reranked = rerank_results_v13(query, all_results, top_n=10)

        # MedDRA階層情報付加
        reranked = add_hierarchy_info(reranked, term_master_df)

        # GPTフィルタ適用（任意）
        if use_filter and predicted_keywords:
            try:
                reranked = reranked[
                    reranked[["SOC_Japanese", "HLGT_Japanese", "HLT_Japanese"]].astype(str).apply(
                        lambda x: x.str.contains("|".join(predicted_keywords)).any(), axis=1
                    )
                ]
            except Exception as e:
                st.warning(f"フィルタ処理でエラーが発生しました: {e}")

        # スコア整形
        reranked = rescale_scores(reranked)

        # 列名変換＋表示
        reranked = reranked.rename(columns={
            "term": "用語",
            "score": "確からしさ（％）",
            "HLT_Japanese": "HLT",
            "HLGT_Japanese": "HLGT",
            "SOC_Japanese": "SOC"
        })
        display_cols = [col for col in ["用語", "確からしさ（％）", "HLT", "HLGT", "SOC"] if col in reranked.columns]
        st.success("検索完了")
        st.dataframe(reranked[display_cols])

        # ダウンロード
        csv = reranked[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button("📥 結果をCSVでダウンロード", data=csv, file_name="meddra_results.csv", mime="text/csv")
