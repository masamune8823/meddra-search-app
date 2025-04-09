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
            
        # ✅ STEP 5.5: LLT → PT の補完処理（term → PT_Japanese に正規化）
        try:
            llt_df = pd.read_csv("data/1_low_level_term_j.csv", encoding="utf-8-sig")
            if not os.path.exists(llt_path):
                raise FileNotFoundError(f"{llt_path} が見つかりません。")

            llt_df = pd.read_csv(llt_path)
            llt_to_pt = dict(zip(llt_df["LLT_Japanese"], llt_df["PT_Japanese"]))
            reranked["term_mapped"] = reranked["term"].map(llt_to_pt).fillna(reranked["term"])
            st.write("🧭 term → PT変換後のユニーク語数:", reranked["term_mapped"].nunique())

            # ✅ デバッグ：変換後のユニーク語一覧（抜粋）
            mapped_terms = reranked["term_mapped"].unique().tolist()
            st.write("📌 term_mapped（変換後）抜粋:", mapped_terms[:10])

            # ✅ デバッグ：PT_Japanese にマッチしなかった term_mapped のチェック
            pt_set = set(term_master_df["PT_Japanese"].dropna())
            unmatched_pt = set(reranked["term_mapped"]) - pt_set
            st.warning("🧯 PT_Japanese に存在しない term_mapped（上位10件）:")
            st.write(list(unmatched_pt)[:10])

        except Exception as e:
            st.warning(f"LLT→PT変換処理でエラーが発生しました: {e}")
            reranked["term_mapped"] = reranked["term"]  # fallback を必ず作成
            
            
            # ✅ デバッグ：変換後のユニーク語一覧（抜粋）
            mapped_terms = reranked["term_mapped"].unique().tolist()
            st.write("📌 term_mapped（変換後）抜粋:", mapped_terms[:10])

            # ✅ デバッグ：PT_Japanese にマッチしなかった term_mapped のチェック
            pt_set = set(term_master_df["PT_Japanese"].dropna())
            unmatched_pt = set(reranked["term_mapped"]) - pt_set
            st.warning("🧯 PT_Japanese に存在しない term_mapped（上位10件）:")
            st.write(list(unmatched_pt)[:10])



            # ✅ STEP 6: MedDRA階層付加
            with st.spinner("階層情報を付加中..."):

                # STEP 6.1: term列の準備（term_mapped → term にリネーム or fallback で空列追加）
                if "term_mapped" in reranked.columns:
                    df_for_merge = reranked.rename(columns={"term_mapped": "term"}).copy()
                elif "term" in reranked.columns:
                    df_for_merge = reranked.copy()
                else:
                    st.warning("⚠️ 'term' 列が存在しないため、空列を追加します。")
                    df_for_merge = reranked.copy()
                    df_for_merge["term"] = ""

                # STEP 6.2: デバッグ出力
                try:
                    if "term" in df_for_merge.columns:
                        preview = df_for_merge["term"].dropna().astype(str).unique().tolist()
                        st.write("🧭 term列（階層付加用）のユニーク値（抜粋）:", preview[:10])
                        st.write("📌 df_for_merge のカラム一覧:", df_for_merge.columns.tolist())
                except Exception as e:
                    st.warning(f"⚠️ term列のプレビュー表示中にエラー: {e}")

                # STEP 6.3: 階層情報をマージ（term_mapped → PT_Japanese）
                try:
                    if "term_mapped" not in reranked.columns:
                        reranked["term_mapped"] = reranked["term"]

                    # ✅ term_master_dfに "term" 列があれば削除（念のため）
                    term_master_clean = term_master_df.drop(columns=["term"], errors="ignore")

                final_results = pd.merge(
                        reranked,
                        term_master_clean,
                        how="left",
                        left_on="term_mapped",
                        right_on="PT_Japanese",
                        suffixes=("", "_master")
                    )

                    # ✅ 重複カラムがある場合、除去（Streamlitエラー防止）
                    if final_results.columns.duplicated().any():
                        final_results = final_results.loc[:, ~final_results.columns.duplicated()]

                    st.write("🧩 final_results の列一覧（直後）:", final_results.columns.tolist())
                except Exception as e:
                    st.error(f"❌ 階層マスタとのマージでエラー: {e}")
                    final_results = reranked.copy()



                # ✅ STEP 6.4: マージ後の確認と未一致チェック
                st.write("🧩 final_results の列一覧（STEP 6.4）:", final_results.columns.tolist())
                st.write("🔍 マージ対象語数:", len(df_for_merge))
                st.write("🔍 階層付与後件数:", len(final_results))
                st.write("📂 term_master_df の列一覧:", term_master_df.columns.tolist())
                
                base_terms = set(df_for_merge["term"]) if "term" in df_for_merge.columns else set()
                hier_terms = set(final_results["PT_Japanese"].dropna()) if "PT_Japanese" in final_results.columns else set()

                unmatched_terms = base_terms - hier_terms
                if unmatched_terms:
                    st.warning("🧯 階層マスタに一致しなかった用語（PT_Japanese）:")
                    st.write(list(unmatched_terms)[:10])


                
            # ✅ STEP 7: SOCフィルタ
            if use_soc_filter:
                try:
                    soc_prediction = predict_soc_category(query)
                    if "SOC_Japanese" in final_results.columns:
                        final_results = final_results[
                            final_results["SOC_Japanese"].fillna("").astype(str).str.contains(soc_prediction)
                        ]
                        st.write(f"🔍 フィルタ前: {len(df_for_merge)} 件 → フィルタ後: {len(final_results)} 件")
                    else:
                        st.warning("⚠️ final_results に 'SOC_Japanese' 列が存在しません。フィルタをスキップします。")
                except Exception as e:
                    st.warning(f"フィルタ処理でエラーが発生しました: {e}")

                st.success("検索完了")

            # STEP 8: 表示対象カラム（存在チェック付き）
            display_cols = [
                "term", "score",
                "PT_Japanese", "HLT_Japanese", "HLGT_Japanese", "SOC_Japanese"
            ]
            available_cols = [col for col in display_cols if col in final_results.columns]

            # STEP 8.1: 日本語に変換して表示
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
