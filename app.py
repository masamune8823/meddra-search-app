
# app.py
import streamlit as st
import pandas as pd
import os

from helper_functions import (
    expand_query_gpt,
    encode_query,
    search_meddra,
    rerank_results_v13,
    add_hierarchy_info,
    load_term_master_df
)

# ページ設定
st.set_page_config(page_title="MedDRA検索アプリ", layout="wide")

st.title("💊 MedDRA 自然言語検索システム")
st.write("自然文から適切なMedDRA PT用語を検索します。")

# クエリ入力
query = st.text_input("🔍 症状や状態を入力してください（例：ズキズキ、吐き気 など）")

# term_master_df のロード（毎回明示）
term_master_path = os.path.join("term_master_df.pkl")
if os.path.exists(term_master_path):
    term_master_df = pd.read_pickle(term_master_path)
else:
    st.error("❌ term_master_df.pkl が見つかりません")
    st.stop()

# 検索実行
if st.button("検索実行") and query.strip():
    with st.spinner("検索中..."):
        raw_results = search_meddra(query, top_k_per_method=5)
        reranked = rerank_results_v13(raw_results, query, top_n=10)
        final_results = add_hierarchy_info(reranked, term_master_df)

    # 結果表示
    if final_results:
        df = pd.DataFrame(final_results)
        df.columns = ["用語", "確からしさ（％）", "HLT", "HLGT", "SOC", "出典"]
        st.dataframe(df, use_container_width=True)

        # CSVダウンロード
        csv = df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="📥 検索結果をCSVでダウンロード",
            data=csv,
            file_name=f"meddra_results_{query}.csv",
            mime="text/csv"
        )
    else:
        st.warning("検索結果が見つかりませんでした。")
