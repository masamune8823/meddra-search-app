
import streamlit as st
import pandas as pd
import pickle
from helper_functions import (
    expand_query_gpt,
    encode_query,
    search_meddra,
    rerank_results_v13,
    load_term_master_df
)

# 階層情報の後付け補完（スコア結果に対し term_master_df からマージ）
def fill_hierarchy_if_missing(results, term_master_df):
    df = pd.DataFrame(results, columns=["term", "score", "HLT", "HLGT", "SOC", "source"])
    missing_mask = df["HLT"] == ""
    if not missing_mask.any():
        return results  # 補完不要

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


# タイトル
st.title("💊 MedDRA検索システム")

# クエリ入力
query = st.text_input("症状や訴えを入力してください（例：頭がズキズキする）")

# term_master_df を読み込み
term_master_df = load_term_master_df("/mnt/data/term_master_df.pkl")

# 実行ボタン
if st.button("検索") and query:
    with st.spinner("検索中..."):
        keywords = expand_query_gpt(query)
        st.markdown("**🔍 拡張語（GPT）:**")
        st.write(", ".join(keywords))

        # 検索と再スコア
        results = []
        for kw in keywords:
            hits = search_meddra(kw)
            results.extend(hits)

        reranked = rerank_results_v13(results, query)

        # 階層情報を後付け補完
        final_results = fill_hierarchy_if_missing(reranked, term_master_df)

        # 表示
        df = pd.DataFrame(final_results, columns=["PT名", "確からしさ（％）", "HLT", "HLGT", "SOC", "拡張語"])
        st.dataframe(df, use_container_width=True)

        # CSV保存オプション
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 CSVダウンロード", data=csv, file_name="meddra_results.csv", mime="text/csv")
