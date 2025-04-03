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
    add_hierarchy_info
)

def main():
    st.title("MedDRA検索アプリ")
    query = st.text_input("検索クエリを入力してください")

    if query:
        st.write(f"🔍 入力クエリ: {query}")
        # 簡易表示（例示）
        results = search_meddra(query)
        reranked = rerank_results_v13(results, query)

        # 階層情報付加（存在すれば）
        if 'term_master_df' in locals():
            reranked = add_hierarchy_info(reranked, term_master_df)

        st.dataframe(reranked)

if __name__ == '__main__':
    main()
