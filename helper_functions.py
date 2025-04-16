
# キャッシュファイルの読み込み（必要に応じて呼び出し）
def load_score_cache(path="score_cache.pkl"):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except:
        return {}

def load_query_cache(path="query_expansion_cache.pkl"):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except:
        return {}

from openai import OpenAI
client = OpenAI()

import os
import re
import pickle
import openai
import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Hugging Face トークンを Streamlit Secrets から取得
hf_token = os.getenv("HF_TOKEN")

# トークン付きでモデルロード
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", use_auth_token=hf_token)

# OpenAI APIキー（環境変数から取得）
openai.api_key = os.getenv("OPENAI_API_KEY")

# クエリをベクトル化
def encode_query(text):
    return model.encode([text])[0]

# ✅ 改良版 検索処理（部分一致 + 辞書 + FAISS）v2
def search_meddra_v2(query, faiss_index, meddra_terms, synonym_df, top_k_faiss=10, matched_from_label=None):
    import pandas as pd

    results = []
    matched_terms = set()

    # ✅ 1. シノニム辞書（variant → PT_Japanese）
    if synonym_df is not None and "variant" in synonym_df.columns:
        synonym_hits = synonym_df[synonym_df["variant"] == query]
        for _, row in synonym_hits.iterrows():
            term = row["PT_Japanese"]
            if term not in matched_terms:
                results.append({"term": term, "score": 1.0, "matched_from": "シノニム辞書検索"})
                matched_terms.add(term)

    # ✅ 2. 正規辞書照合（部分一致）
    for term in meddra_terms:
        if isinstance(term, str) and query.lower() in term.lower():
            if term not in matched_terms:
                results.append({"term": term, "score": 1.0, "matched_from": "正規辞書照合検索"})
                matched_terms.add(term)

    # ✅ 3. FAISSベクトル検索
    from helper_functions import encode_query
    query_vector = encode_query(query).astype(np.float32)
    distances, indices = faiss_index.search(np.array([query_vector]), top_k_faiss)
    for i in range(len(indices[0])):
        idx = indices[0][i]
        if idx < len(meddra_terms):
            term_raw = meddra_terms[idx]
            # ✅ "English / 日本語" の形式なら日本語だけ使う
            term = term_raw.split("/")[-1].strip() if "/" in term_raw else term_raw.strip()

            results.append({
                "term": term,
                "score": float(distances[0][i]),
                "matched_from": matched_from_label or " FAISSベクトル検索"
            })

    return pd.DataFrame(results)



#  ベクトル類似によるPT候補提示
def suggest_similar_terms(query, faiss_index, meddra_terms, top_k=10):
    """
    クエリに意味的に近いPT候補を、ベクトル検索により取得する。
    """
    query_vector = encode_query(query).astype(np.float32)
    distances, indices = faiss_index.search(np.array([query_vector]), top_k)
    suggestions = [meddra_terms[idx] for idx in indices[0] if idx < len(meddra_terms)]
    return suggestions
   
# 回答からスコア抽出（単純実装）
def extract_score_from_response(response_text):
    for word in ["10", "９", "8", "７", "6", "5", "4", "3", "2", "1", "0"]:
        if word in response_text:
            try:
                return float(word)
            except:
                continue
    return 5.0  # fallback

# スコアの再スケーリング
def rescale_scores(scores):
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [100.0 for _ in scores]
    return [100.0 * (s - min_score) / (max_score - min_score) for s in scores]

# ✅ 再ランキング処理（GPT一括呼び出し版）
# ✅ GPT再ランキング処理（1メッセージ形式）
def rerank_results_batch(query, candidates, score_cache=None):
    if score_cache is None:
        score_cache = {}

    top_candidates = candidates.head(10)

    # 未スコアの term だけを抽出
    new_terms = []
    for i, row in top_candidates.iterrows():
        term = row["term"]
        if (query, term) not in score_cache:
            new_terms.append(term)
    # ✅ スコア対象の語数と中身をStreamlitで表示（デバッグ用）
    import streamlit as st
    # st.write("🧪 スコア未評価語数:", len(new_terms), "件")
    # st.write("🧪 未評価語リスト:", new_terms)
    
    if new_terms:
        # 🔧 プロンプト組み立て（1メッセージに全term）
        prompt = f"""以下の記述「{query}」に対して、各用語がどれくらい意味的に一致するかを教えてください。
一致度を 0〜10 の数値で記述してください。

"""
        for idx, term in enumerate(new_terms, 1):
            prompt += f"{idx}. {term}\n"

        prompt += "\n形式：\n1. 7\n2. 5\n... のように記載してください。"

        messages = [
            {"role": "system", "content": "あなたは医療用語の関連性を数値で判断する専門家です。"},
            {"role": "user", "content": prompt}
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0,
            )
            content = response.choices[0].message.content

            # ✅ Streamlitログ表示（デバッグ用）
            import streamlit as st
            # st.subheader("🧾 GPTレスポンス内容（一括形式）")
            #  st.code(content)

            # 数値抽出（形式：1. 7）
            for line in content.strip().split("\n"):
                if "." in line:
                    parts = line.split(".")
                    try:
                        idx = int(parts[0].strip())
                        score = extract_score_from_response(line)
                        term = new_terms[idx - 1]
                        score_cache[(query, term)] = score
                    except:
                        continue
        except Exception as e:
            for term in new_terms:
                score_cache[(query, term)] = 5.0  # fallback

    # スコアをまとめて返す
    scored = [(term, score_cache.get((query, term), 5.0)) for term in top_candidates["term"]]
    df = pd.DataFrame(scored, columns=["term", "Relevance"])
    return df.sort_values(by="Relevance", ascending=False)


# GPTでSOCカテゴリを予測
def predict_soc_category(query):
    messages = [
        {"role": "system", "content": "あなたはMedDRAのSOCカテゴリを特定する専門家です。"},
        {"role": "user", "content": f"次の症状に最も関連するMedDRAのSOCカテゴリを1つだけ、日本語で簡潔に答えてください（例：「神経系障害」など）。\n\n症状: {query}"}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "不明"

# クエリ拡張（GPT使用）
def expand_query_gpt(query, query_cache=None):
    if query_cache is not None and query in query_cache:
        return query_cache[query]

    # GPTへ問い合わせ（具体的な症状名を求めるように改善）
    messages = [
        {"role": "system", "content": "あなたは日本語のあいまいな症状表現を、正確な英語の医学用語に変換する専門家です。"},
        {"role": "user", "content": f"""
以下の日本語の症状「{query}」について、具体的に考えられる医学的な症状名や疾患名（英語）を3つ予測してください。

例：「ズキズキ」→ "headache", "migraine", "throbbing pain"

・曖昧なカテゴリ（例：神経系障害）ではなく、具体的な症状名や疾患名を出力してください。
・必ず3つ、カンマ区切りで出力してください。
"""}
    ]

def expand_query_gpt(query, query_cache=None):
    if query_cache is not None and query in query_cache:
        return query_cache[query]

    # GPTへ問い合わせ（具体的な症状名を求めるように改善）
    messages = [
        {"role": "system", "content": "あなたは日本語のあいまいな症状表現を、正確な英語の医学用語に変換する専門家です。"},
        {"role": "user", "content": f"""
以下の日本語の症状「{query}」について、具体的に考えられる医学的な症状名や疾患名（英語）を3つ予測してください。

例：「ズキズキ」→ "headache", "migraine", "throbbing pain"

・曖昧なカテゴリ（例：神経系障害）ではなく、具体的な症状名や疾患名を出力してください。
・必ず3つ、カンマ区切りで出力してください。
"""}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
        )
        response_text = response.choices[0].message.content.strip()

        # ✅ Streamlitログでレスポンス確認
        import streamlit as st
        # st.subheader("📥 GPT 生レスポンス内容（拡張語）")
        # st.code(response_text)

        # 🔧 整形処理（番号 or カンマ対応）
        raw_lines = response_text.strip().split("\n")

        keywords = []
        for line in raw_lines:
            line = re.sub(r'^\d+\.\s*', '', line)
            line = line.strip().strip('"')
            if line:
                keywords.append(line)

        if not keywords:
            keywords = [kw.strip().strip('"') for kw in response_text.split(",") if kw.strip()]

        if query_cache is not None:
            query_cache[query] = keywords
        return keywords

    except Exception as e:
        return ["headache", "fever", "pain"]


# 表示整形（キーワードリスト）
def format_keywords(keywords):
    return "、".join(keywords)

# MedDRA階層情報を付与
def add_hierarchy_info(df, term_master_df):
    merged = pd.merge(df, term_master_df, how="left", left_on="term", right_on="PT_English")
    return merged
    
# ✅ 日本語PT（PT_Japanese）で階層情報をマージする関数
def add_hierarchy_info_jp(df, term_master_df):
    return pd.merge(df, term_master_df, how="left", left_on="term", right_on="PT_Japanese")
