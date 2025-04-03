
    import openai
    import pickle
    import os
    import pandas as pd

    # キャッシュ読み込み（存在しなければ空dict）
    def load_soc_cache(path="soc_predict_cache.pkl"):
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return {}

    def save_soc_cache(cache, path="soc_predict_cache.pkl"):
        with open(path, "wb") as f:
            pickle.dump(cache, f)

    # GPTによるSOCカテゴリ予測
    def predict_soc_keywords_with_gpt(query, client, model="gpt-3.5-turbo", cache_path="soc_predict_cache.pkl"):
        cache = load_soc_cache(cache_path)
        if query in cache:
            return cache[query]

        prompt = f"""
以下の症状は、MedDRAで定義されるどのSOCカテゴリに最も関連していますか？

症状: 「{query}」

可能性のあるSOCカテゴリ（日本語で1～3個程度）を、簡潔なキーワードで出力してください。
        """

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        lines = response.choices[0].message.content.strip().splitlines()
        keywords = [kw.replace("-", "").strip("・:：●- ") for kw in lines if kw]
        cache[query] = keywords
        save_soc_cache(cache, cache_path)
        return keywords

    # GPT予測に基づく階層フィルタ
    def filter_by_predicted_soc(results_df, soc_keywords):
        if not soc_keywords:
            return results_df

        mask = results_df[["SOC", "HLGT", "HLT"]].apply(
            lambda cols: cols.astype(str).str.contains("|".join(soc_keywords), case=False).any(axis=1),
            axis=1
        )
        return results_df[mask].copy()

    # term_master_df 読み込み
    def load_term_master_df(path):
        return pd.read_pickle(path)

    # ダミー関数（本番では置換）
    def search_meddra(query, top_k_per_method=5):
        return []

    def rerank_results_v13(results, query, top_n=10):
        return []
