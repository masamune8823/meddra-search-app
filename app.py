
# app.py
import streamlit as st
import pandas as pd
import os

from helper_functions import (
    search_meddra,
    rerank_results_v13,
    load_term_master_df
)

# ✅ Streamlit Cloud 用：階層情報を UI 側で後付け補完
def fill_hierarchy_if_missing(results, term_master_df):
    df = pd.DataFrame(results, columns=["term", "score", "HLT", "HLGT", "SOC", "source"])
    if "PT_Japanese" not in term_master_df.columns:
        st.error("term_master_df に 'PT_Japanese' 列がありません。")
        return results

    merged = df.merge(
        term_master_df[["PT_Japanese", "HLT_Japanese", "HLGT_Japanese", "SOC_Japanese"]],
        how="left",
        left_on="term",
        right_on="PT_Japanese"
    )

    for col_from, col_to in [
        ("HLT_Japanese", "HLT"),
        ("HLGT_Japanese", "HLGT"),
        ("SOC_Japanese", "SOC")
    ]:
        merged[col_to] = merged[col_to].where(merged[col_to] != "", merged[col_from].fillna(""))

    final_df = merged[["term", "score", "HLT", "HLGT", "SOC", "source"]]
    return list(final_df.itertuples(index=False, name=None))


# ページ設定
st.set_page_config(page_title="MedDRA検索アプリ", layout="wide")

st.title("💊 MedDRA 自然言語検索システム")
st.write("自然文から適切なMedDRA PT用語を検索します。")

# クエリ入力
query = st.text_input("🔍 症状や状態を入力してください（例：ズキズキ、吐き気 など）")

# term_master_df のロード（GitHub/Streamlit Cloud 用パス）
term_master_path = "term_master_df.pkl"
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
        final_results = fill_hierarchy_if_missing(reranked, term_master_df)

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
