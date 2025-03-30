
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import faiss
# ✅ 明示的に .py をつけてモジュール解釈を安定させる場合もある
from utils import expand_query_gpt, encode_query, rerank_results_v13  # utils.py は同階層
import os

@st.cache_resource
def load_faiss_and_data():
    index = faiss.read_index("/mnt/data/faiss_index.index")
    embeddings = np.load("/mnt/data/meddra_embeddings.npy")
    with open("/mnt/data/meddra_terms.npy", "rb") as f:
        terms = np.load(f, allow_pickle=True)
    with open("/mnt/data/term_master_df.pkl", "rb") as f:
        master_df = pickle.load(f)
    return index, terms, master_df

faiss_index, meddra_terms, term_master_df = load_faiss_and_data()

st.set_page_config(page_title="MedDRA検索システム", layout="wide")
st.title("🩺 MedDRA検索システム（プロトタイプUI）")

st.header("1. 医師記載用語の入力")
user_input = st.text_area("自然言語で記載された症状や出来事を入力してください：", height=100)

if st.button("🔍 検索実行") and user_input:
    st.header("2. クエリ拡張（キーワード抽出）")
    expanded_terms = expand_query_gpt(user_input)
    st.write("抽出されたキーワード：", expanded_terms)

    all_results = []
    for term in expanded_terms:
        query_vec = encode_query(term)
        scores, indices = faiss_index.search(np.array([query_vec]), k=20)
        for score, idx in zip(scores[0], indices[0]):
            result = {
                "term": meddra_terms[idx],
                "score": float(score),
                "query_term": term
            }
            all_results.append(result)

    reranked = rerank_results_v13(user_input, all_results)

    results_df = pd.DataFrame(reranked)
    merged_df = pd.merge(results_df, term_master_df, how="left", left_on="term", right_on="PT_English")
    merged_df = merged_df[["score", "term", "PT_Japanese", "HLT_Japanese", "HLGT_Japanese", "SOC_Japanese"]].copy()
    merged_df["確からしさ"] = (merged_df["score"] * 10).clip(0, 100).round(1).astype(str) + "%"
    merged_df.rename(columns={"term": "PT_English"}, inplace=True)
    merged_df = merged_df[["確からしさ", "PT_Japanese", "PT_English", "HLT_Japanese", "HLGT_Japanese", "SOC_Japanese"]]

    st.header("3. 検索結果（Top10候補＋確からしさ）")
    st.dataframe(merged_df.head(10), use_container_width=True)

    st.header("4. 出力と検索履歴")
    st.download_button("💾 結果をCSVでダウンロード", merged_df.to_csv(index=False).encode("utf-8"), file_name="meddra_results.csv", mime="text/csv")

else:
    st.info("上のテキスト欄に入力し、[検索実行]を押してください。")
