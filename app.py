import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import faiss

from helper_functions import (
    encode_query,
    search_meddra,
    rerank_results_batch,
    add_hierarchy_info,
    rescale_scores,
    predict_soc_category,
    format_keywords,
    suggest_similar_terms,      
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

        with st.spinner("再スコアリング中（GPT一括）..."):
            score_cache = {}  # ✅ 追加（APIコールを繰り返さないためのキャッシュ）
            reranked = rerank_results_batch(query, all_results, score_cache)
            reranked["score"] = rescale_scores(reranked["Relevance"].tolist())

        with st.spinner("階層情報を付加中..."):
            st.write("列名チェック（reranked）:", reranked.columns.tolist())  # ← ここ追加
            final_results = add_hierarchy_info(reranked, term_master_df)
            st.write("🧩 final_results の列一覧:", final_results.columns.tolist())  # ← 🔍 SOC列があるか確認

            # ✅ ここから追加：マージ確認ログ（STEP 5-B）
            st.write("🔍 マージ対象語数:", len(reranked))
            st.write("🔍 階層付与後件数:", len(final_results))

            unmatched_terms = set(reranked["term"]) - set(final_results["PT_English"].dropna())
            if unmatched_terms:
                st.warning("🧯 階層マスタに一致しなかった用語（PT_English）:")
                st.write(list(unmatched_terms))

        if use_soc_filter:
             try:
                soc_prediction = predict_soc_category(query)
                # ✅ 修正：SOC_Japanese にフィルタを適用
                # 🔍 NaN対策 + フィルタ
                final_results = final_results[final_results["SOC_Japanese"].fillna("").astype(str).str.contains(soc_prediction)]
                st.write(f"🔍 フィルタ前: {len(final_results)} 件 → フィルタ後: {len(final_results[final_results['SOC_Japanese'].fillna('').str.contains(soc_prediction)])} 件")

             except Exception as e:
                st.warning(f"フィルタ処理でエラーが発生しました: {e}")

        st.success("検索完了")

        # ✅ 表示する列を日本語の階層構造で拡張
        display_cols = [
            "term", "score",
            "PT_Japanese", "HLT_Japanese", "HLGT_Japanese", "SOC_Japanese"
        ]
        available_cols = [col for col in display_cols if col in final_results.columns]

        # 表示（列名も日本語に置き換え）
        st.dataframe(
            final_results[available_cols].rename(columns={
                "term": "用語（再スコア対象語）",
                "score": "確からしさ (%)",
                "PT_Japanese": "PT（日本語）",
                "HLT_Japanese": "HLT（日本語）",
                "HLGT_Japanese": "HLGT（日本語）",
                "SOC_Japanese": "SOC（日本語）"
            })
        )

        # CSV生成時に encoding を指定する
        csv = final_results.to_csv(index=False, encoding="utf-8-sig")

        # ダウンロードボタン
        st.download_button("📆 結果をCSVでダウンロード", data=csv, file_name="meddra_results.csv", mime="text/csv")
        
        # 🔍 テスト用ボタン
        if st.button("🔍 テスト実行（ズキズキ）"):
            from test_meddra_full_pipeline import run_test_pipeline
            run_test_pipeline()

        # ✅ ステップA：意味的に近い用語候補を表示（ズキズキ → 頭痛など）
        with st.expander("🧠 類似語候補を表示（ベクトル検索）"):
            similar_terms = suggest_similar_terms(query, faiss_index, meddra_terms, top_k=10)
            st.write("💡 入力語に意味的に近い用語候補:")
            for i, term in enumerate(similar_terms, 1):
                st.markdown(f"{i}. {term}")
