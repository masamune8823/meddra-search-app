import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import faiss
QUERY_CACHE_PATH = "data/query_expansion_cache.pkl"

from helper_functions import (
    encode_query,
    search_meddra,
    rerank_results_batch,
    add_hierarchy_info,
    rescale_scores,
    predict_soc_category,
    format_keywords,
    suggest_similar_terms, 
    load_score_cache, 
    load_query_cache, 
    add_hierarchy_info_jp,
    expand_query_gpt,
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

        # ✅ synonym_df.pkl のみ読み込み
        synonym_path = "data/synonym_df.pkl"
        if not os.path.exists(synonym_path):
            st.error("❌ synonym_df.pkl が存在しません。先に作成してください。")
            raise FileNotFoundError("synonym_df.pkl not found")

        synonym_df = pickle.load(open(synonym_path, "rb"))

        # ✅ カラム名チェック
        if not {"variant", "PT_Japanese"}.issubset(synonym_df.columns):
            st.error("❌ synonym_df に必要なカラム（variant / PT_Japanese）がありません。")
            raise ValueError("synonym_df のカラム不一致")

        term_master_df = pickle.load(open("term_master_df.pkl", "rb"))

    except Exception as e:
        st.error(f"データファイルの読み込みに失敗しました: {e}")
        raise e

    return faiss_index, meddra_terms, synonym_df, term_master_df




faiss_index, meddra_terms, synonym_df, term_master_df = load_assets()

# キャッシュ読み込み
score_cache = load_score_cache("score_cache.pkl")
query_cache = load_query_cache("query_expansion_cache.pkl")

# ✅ Streamlitサイドバーにキャッシュ削除ボタンを追加
if st.sidebar.button("🗑️ スコアキャッシュを削除"):
    if os.path.exists("score_cache.pkl"):
        os.remove("score_cache.pkl")
        score_cache = {}
        st.sidebar.success("✅ score_cache.pkl を削除しました。再実行時に再作成されます。")
    else:
        st.sidebar.warning("⚠️ score_cache.pkl はすでに存在しません。")

# ✅ Streamlitサイドバーにクエリ拡張キャッシュ削除ボタンを追加
if st.sidebar.button("🗑️ 拡張語キャッシュを削除"):
    if os.path.exists(QUERY_CACHE_PATH):
        os.remove(QUERY_CACHE_PATH)
        query_cache = {}
        st.sidebar.success("拡張語キャッシュを削除しました。")
    else:
        st.sidebar.warning("拡張語キャッシュは存在しません。")


# ---------------- ユーザー入力 ---------------- #
query = st.text_input("検索語を入力してください（例：皮膚がかゆい）", value="ズキズキ")
use_soc_filter = st.checkbox("GPTによるSOC予測でフィルタリング（推奨）", value=True)

# ---------------- 検索処理 ---------------- #
if st.button("検索"):
    if not query.strip():
        st.warning("検索語を入力してください。")
    else:
        with st.spinner("キーワードを解析中..."):
            # ✅ STEP 1: クエリ拡張（例：「ズキズキ」→ "headache", "migraine", ...）
            predicted_keywords = expand_query_gpt(query, query_cache)

            # ✅ STEP 2: SOCカテゴリ予測（例：「神経系障害」など、フィルタ用）
            soc_prediction = predict_soc_category(query)

            # ✅ STEP 3: キャッシュ使用の表示
            if query in query_cache:
                st.info("✅ クエリ拡張キャッシュを使用しました。")
            else:
                st.info("🆕 新しい拡張語を生成しました（キャッシュ追加済）。")

        # ✅ STEP 3.5: デバッグ表示（拡張語の確認）
        st.subheader("🧠 GPT予測キーワード（整形後）")
        st.write(predicted_keywords)

        # ✅ STEP 4: FAISS検索
        with st.spinner("FAISSで用語検索中..."):
            search_results = []
            for kw in predicted_keywords:
                result = search_meddra(kw, faiss_index, meddra_terms, synonym_df, top_k=20)
                search_results.append(result)
            all_results = pd.concat(search_results).drop_duplicates(subset=["term"]).reset_index(drop=True)
            
        # ✅ STEP 5: GPT再スコアリング
        with st.spinner("再スコアリング中（GPT一括）..."):
            score_cache = {}  # ✅ 追加（APIコールを繰り返さないためのキャッシュ）
            reranked = rerank_results_batch(query, all_results, score_cache)
            reranked["score"] = rescale_scores(reranked["Relevance"].tolist())
            
        # ✅ STEP 6: MedDRA階層付加
        with st.spinner("階層情報を付加中..."):
            st.write("列名チェック（reranked）:", reranked.columns.tolist())  # ← ここ追加
            final_results = add_hierarchy_info_jp(reranked, term_master_df)
            st.write("🧩 final_results の列一覧:", final_results.columns.tolist())  # ← 🔍 SOC列があるか確認

            st.write("🔍 マージ対象語数:", len(reranked))
            st.write("🔍 階層付与後件数:", len(final_results))

            unmatched_terms = set(reranked["term"]) - set(final_results["PT_English"].dropna())
            if unmatched_terms:
                st.warning("🧯 階層マスタに一致しなかった用語（PT_English）:")
                st.write(list(unmatched_terms))
        # ✅ STEP 7: SOCフィルタ
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

        # ✅ STEP 8: 結果表示  表示する列を日本語の階層構造で拡張
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

        # ✅ キャッシュの保存（検索完了後）
        with open("score_cache.pkl", "wb") as f:
            pickle.dump(score_cache, f)

        with open("query_expansion_cache.pkl", "wb") as f:
            pickle.dump(query_cache, f)

        # ✅ ステップA：意味的に近い用語候補を表示（ズキズキ → 頭痛など）
        with st.expander("🧠 類似語候補を表示（ベクトル検索）"):
            similar_terms = suggest_similar_terms(query, faiss_index, meddra_terms, top_k=10)
            st.write("💡 入力語に意味的に近い用語候補:")
            for i, term in enumerate(similar_terms, 1):
                st.markdown(f"{i}. {term}")
