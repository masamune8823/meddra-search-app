# helper_functions.py

def expand_query_gpt(user_query):
    # クエリをOpenAIなどで拡張する（ダミー実装）
    return [user_query, "拡張語1", "拡張語2"]

def encode_query(query):
    # クエリをベクトル化（仮のベクトル）
    return [0.1, 0.2, 0.3]

def rerank_results_v13(results):
    # スコアの高い順に並べ替える仮のロジック
    return sorted(results, key=lambda x: x["score"], reverse=True)

def match_synonyms(df, synonym_df):
    # PT名またはLLT名がシノニム辞書にあるかどうかでフラグを立てる
    df["matched_synonym"] = df["PT_Japanese"].isin(synonym_df["synonym"])
    return df

def merge_faiss_and_synonym_results(faiss_df, synonym_df):
    # FAISS検索結果とシノニムマッチを結合（重複排除の上で結合）
    combined_df = faiss_df.copy()
    synonym_only = synonym_df[~synonym_df["PT_Japanese"].isin(faiss_df["PT_Japanese"])]
    return pd.concat([combined_df, synonym_only], ignore_index=True)
