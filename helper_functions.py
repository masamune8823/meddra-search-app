import numpy as np
import pandas as pd
import faiss
import openai
import os
import pickle

from sentence_transformers import SentenceTransformer

# OpenAI API Key を環境変数から取得
openai.api_key = os.getenv("OPENAI_API_KEY")

# モデルの準備
encoder_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# ベクトルエンコード
def encode_query(text: str) -> np.ndarray:
    return encoder_model.encode([text])[0]

# synonym_df の FAISS 検索
def match_synonyms(query, synonym_df, synonym_index, synonym_embeddings, top_k=5):
    query_vec = encode_query(query).astype("float32")
    D, I = synonym_index.search(np.array([query_vec]), top_k)

    results = []
    for idx, score in zip(I[0], D[0]):
        if 0 <= idx < len(synonym_df):
            synonym = synonym_df.iloc[idx]["synonym"]
            results.append({
                "term": synonym,
                "score": float(score),
                "source": "synonym_faiss"
            })
    return results

# synonym_faiss + faiss_results の統合
def merge_faiss_and_synonym_results(faiss_results, synonym_candidates):
    all_results = faiss_results + synonym_candidates
    merged = {}
    for r in all_results:
        key = (r["term"], r["source"])
        if key not in merged or r["score"] > merged[key]["score"]:
            merged[key] = r
    return list(merged.values())

# クエリ拡張（GPT使用・キャッシュ付き）
def expand_query_gpt(query, cache_path="/mnt/data/query_expansion_cache.pkl"):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    if query in cache:
        return cache[query]

    prompt = f"次の医療症状の言い換え・俗称・略語・誤記などを5つ挙げてください: {query}"

    try:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        choices = response.choices[0].message.content.strip().splitlines()
        terms = [c.strip("・-・：: ") for c in choices if c.strip()]
        cache[query] = terms
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        return terms
    except Exception as e:
        print("⚠️ GPTエラー: ", e)
        return [query]

# 再ランキング（GPT）
def rerank_results_v13(results, query, top_n=10, cache_path="/mnt/data/score_cache.pkl"):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    def get_score(q, t):
        key = (q, t)
        if key in cache:
            return cache[key]
        prompt = f"""以下は医療症状に関する言葉です。

ユーザーの訴え: {q}
候補語: {t}

この2つが医学的にどれくらい近い意味かを0〜100点で評価してください。数値のみ出力してください。"""
        try:
            from openai import OpenAI
            client = OpenAI()
            res = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            score = int("".join(filter(str.isdigit, res.choices[0].message.content.strip())))
        except Exception as e:
            print("⚠️ GPTエラー: ", e)
            score = 30
        cache[key] = score
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        return score

    for r in results:
        r["certainty"] = get_score(query, r["term"])

    return sorted(results, key=lambda x: x["certainty"], reverse=True)[:top_n]

# MedDRA階層情報の付加
def add_hierarchy_info(results, term_master_df):
    df = pd.DataFrame(results)
    merged = df.merge(term_master_df, how="left", left_on="term", right_on="PT_Japanese")
    merged = merged.rename(columns={
        "HLT_Japanese": "HLT",
        "HLGT_Japanese": "HLGT",
        "SOC_Japanese": "SOC",
        "certainty": "確からしさ（％）",
        "term": "用語",
        "source": "出典"
    })
    return merged[["用語", "確からしさ（％）", "HLT", "HLGT", "SOC", "出典"]]

# term_master_df のロード関数
def load_term_master_df(path="/mnt/data/term_master_df.pkl"):
    return pd.read_pickle(path)
