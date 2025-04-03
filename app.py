
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import openai
import os
import pickle
import re

from helper_functions import (
    search_meddra,
    rerank_results_v13,
    predict_soc_keywords_with_gpt,
    add_hierarchy_info
)

# term_master_df 読み込み（あれば）
term_master_df = None
try:
    with open("term_master_df.pkl", "rb") as f:
        term_master_df = pickle.load(f)
except Exception as e:
    st.warning(f"term_master_df を読み込めませんでした: {e}")

def clean_keywords(raw_keywords):
    cleaned = []
    for kw in raw_keywords:
        # 行頭の番号や記号除去 → 例："1. かゆみ" → "かゆみ"
        kw = re.sub(r"^[0-9０-９]+[\.．、:\s]*", "", kw)
        kw = kw.strip("・ 0123456789.。、:：\n")
        if len(kw) > 1:
            cleaned.append(kw)
    return cleaned

def main():
    st.title("🔎 MedDRA検索アプリ")
    query = st.text_input("検索クエリを入力してください")

    if query:
        st.markdown("## 🔍 入力クエリ")
        st.write(query)

        # GPTでSOCカテゴリを予測（クエリ拡張）
        with st.spinner("GPTで拡張語を生成中..."):
            raw_keywords = predict_soc_keywords_with_gpt(query)
            cleaned_keywords = clean_keywords(raw_keywords)
            st.markdown("#### 🧠 GPT予測キーワード（整形後）")
            st.write(cleaned_keywords)

        # 類似語検索（FAISS）
        with st.spinner("FAISSで用語検索中..."):
            search_results = []
            for kw in cleaned_keywords:
                result = search_meddra(kw)
                search_results.append(result)
            all_results = pd.concat(search_results).drop_duplicates(subset=["term"]).reset_index(drop=True)

        # 再スコアリング
        with st.spinner("再スコアリング中..."):
            reranked = rerank_results_v13(all_results, query)

        # 階層情報追加
        if term_master_df is not None:
            reranked = add_hierarchy_info(reranked, term_master_df)

        # 列名変換
        reranked = reranked.rename(columns={
            "term": "用語",
            "score": "確からしさ（％）",
            "HLT_Japanese": "HLT",
            "HLGT_Japanese": "HLGT",
            "SOC_Japanese": "SOC"
        })

        # 並べ替え
        if "確からしさ（％）" in reranked.columns:
            sorted_df = reranked.sort_values(by="確からしさ（％）", ascending=False).reset_index(drop=True)
        else:
            sorted_df = reranked

        # 表示
        display_columns = [col for col in ["用語", "確からしさ（％）", "HLT", "HLGT", "SOC"] if col in sorted_df.columns]
        st.markdown("## 📝 検索結果（スコア順）")
        st.dataframe(sorted_df[display_columns])

        # ダウンロード
        csv = sorted_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 結果をCSVでダウンロード", data=csv, file_name="meddra_results.csv", mime="text/csv")

if __name__ == "__main__":
    main()
