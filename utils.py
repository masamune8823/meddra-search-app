def expand_query_gpt(user_query):
    # クエリをOpenAIなどで拡張
    return [user_query, "拡張語1", "拡張語2"]

def encode_query(query):
    # クエリをエンベディング
    return [0.1, 0.2, 0.3]  # 例

def rerank_results_v13(results):
    # 再ランキングロジック（仮）
    return sorted(results, key=lambda x: x["score"], reverse=True)

def search_meddra(query, term_master_df):
    # シンプルな部分一致ベースの検索関数
    query = query.lower()
    df = term_master_df.copy()
    df["score"] = df["PT_Japanese"].str.lower().apply(lambda x: 100 if query in x else 0)
    result_df = df[df["score"] > 0].sort_values(by="score", ascending=False).head(10)
    return result_df
