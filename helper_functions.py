
import openai
import numpy as np
from sentence_transformers import SentenceTransformer, util
import pickle
import pandas as pd

# GPTを使ったクエリ拡張
def expand_query_gpt(original_query, cache={}, model="gpt-3.5-turbo"):
    if original_query in cache:
        return cache[original_query]

    prompt = f"""以下は医療症状に関する言葉です。
ユーザーの訴え：「{original_query}」
この訴えに関連する検索キーワードを3つ、簡潔な単語や句で日本語で挙げてください。"""

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "あなたは医療用語に詳しいアシスタントです。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        keywords = response['choices'][0]['message']['content']
        expanded = [kw.strip("・- 
") for kw in keywords.strip().split("
") if kw.strip()]
        cache[original_query] = expanded
        return expanded
    except Exception as e:
        print(f"Error in expand_query_gpt: {e}")
        return []

# クエリベクトル生成
def encode_query(query, model):
    return model.encode(query, convert_to_tensor=True)

# FAISS検索の結果とスコアをDataFrameにまとめる
def search_meddra(query_vec, faiss_index, terms, embeddings, top_k=10):
    import faiss
    query_np = np.array([query_vec]).astype("float32")
    D, I = faiss_index.search(query_np, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < len(terms):
            results.append({
                "Term": terms[idx],
                "Score": float(score)
            })
    return pd.DataFrame(results)

# GPTスコアリング（再ランキング）
def rerank_results_v13(original_query, candidate_terms, cache={}, model="gpt-3.5-turbo"):
    scored = []
    for term in candidate_terms:
        cache_key = (original_query, term)
        if cache_key in cache:
            score = cache[cache_key]
        else:
            prompt = f"""以下は医療症状に関するクエリと候補用語のペアです。
ユーザーの訴え：「{original_query}」
候補用語：「{term}」
この候補がどの程度関連しているかを100点満点で評価してください。
ただし、単なる言葉の一致ではなく、意味的な妥当性を重視してください。
スコアのみを整数で出力してください。"""
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "あなたは医療分野の専門家です。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
                score_str = response['choices'][0]['message']['content'].strip()
                score = int(''.join(filter(str.isdigit, score_str.splitlines()[0])))
                score = max(0, min(score, 100))  # 範囲制限
                cache[cache_key] = score
            except Exception as e:
                print(f"Error scoring ({original_query}, {term}): {e}")
                score = 0
        scored.append((term, score))

    df = pd.DataFrame(scored, columns=["Term", "Relevance"])
    return df.sort_values("Relevance", ascending=False).reset_index(drop=True)

# 階層情報を付加する
def add_hierarchy_info(df, term_master_df):
    df = df.copy()
    return df.merge(term_master_df, how="left", left_on="Term", right_on="PT_Japanese")

# term_master_df の読み込み関数（Streamlit Cloud対応）
def load_term_master_df(path="/mount/src/meddra-search-app/search_assets/term_master_df.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
