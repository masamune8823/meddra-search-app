import streamlit as st
import pandas as pd
import pickle
import numpy as np
import faiss
import os
from helper_functions import expand_query_gpt, encode_query, rerank_results_v13, match_synonyms, merge_faiss_and_synonym_results

# zipファイルからの展開（streamlit_app_bundle.zip）
if os.path.exists("streamlit_app_bundle.zip"):
    st.write("📦 ZIPファイルを展開中...")
    unzip_log = os.popen("unzip -o streamlit_app_bundle.zip").read()
    st.text(unzip_log)

# ファイル存在確認
st.write("📁 カレントディレクトリ:", os.getcwd())
st.write("📄 ファイル一覧:", os.listdir())

# データ読み込み関数
@st.cache_resource
def load_data():
    terms = np.load("meddra_terms.npy", allow_pickle=True)
    embeddings_part_a = np.load("meddra_embeddings_part_a", allow_pickle=True)
    embeddings_part_b = np.load("meddra_embeddings_part_b", allow_pickle=True)
    embeddings = np.concatenate((embeddings_part_a, embeddings_part_b))

    with open("term_master_df.pkl", "rb") as f:
        term_master_df = pickle.load(f)
    with open("synonym_df_cat1.pkl", "rb") as f:
        synonym_df = pickle.load(f)

    return terms, embeddings, synonym_df, term_master_df

@st.cache_resource
def load_faiss_index():
    return faiss.read_index("faiss_index.index")

# UI
st.title("💊 MedDRA検索アプリ")
st.write("症状や記述を入力してください")
user_input = st.text_input("症状入力", "頭痛")

if st.button("検索"):
    with st.spinner("検索中..."):
        try:
            # データの読み込み
            terms, embeddings, synonym_df, term_master_df = load_data()
            index = load_faiss_index()

            # クエリ拡張（GPT）
            expanded_queries = expand_query_gpt(user_input)

            # 各クエリをエンコードしてFAISS検索
            all_results = []
            for q in expanded_queries:
                query_vec = encode_query(q)
                D, I = index.search(np.array([query_vec]), k=10)
                for score, idx in zip(D[0], I[0]):
                    all_results.append({
                        "term": terms[idx],
                        "score": float(score),
                        "source": f"FAISS ({q})"
                    })

            # シノニム辞書とのマッチ
            synonym_matches = match_synonyms(user_input, synonym_df)
            all_results.extend(synonym_matches)

            # 再スコアリング
            final_results = rerank_results_v13(all_results)

            # 結果の統合・整形
            merged = merge_faiss_and_synonym_results(final_results, term_master_df)
            st.success("検索完了！")
            st.dataframe(merged)

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
