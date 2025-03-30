def expand_query_gpt(user_query):
    # クエリをOpenAIなどで拡張
    return [user_query, "拡張語1", "拡張語2"]

def encode_query(query):
    # クエリをエンベディング
    return [0.1, 0.2, 0.3]  # 例

def rerank_results_v13(results):
    # 再ランキングロジック（仮）
    return sorted(results, key=lambda x: x["score"], reverse=True)
