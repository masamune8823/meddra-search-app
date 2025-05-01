
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
def search_meddra_v2(original_input, query, faiss_index, meddra_terms, synonym_df, top_k_faiss=10, matched_from_label=None):
    import pandas as pd

    results = []
    matched_terms = set()

    # ✅ 1. シノニム辞書（variant → PT_Japanese）
    if synonym_df is not None and "variant" in synonym_df.columns:
        synonym_hits = synonym_df[synonym_df["variant"] == query]
        for _, row in synonym_hits.iterrows():
            term = row["PT_Japanese"]
            if term not in matched_terms:
                results.append({"original_input": original_input,"term": term, "score": 1.0, "matched_from": "シノニム辞書"})
                matched_terms.add(term)

    # ✅ 2. 正規辞書照合（部分一致）
    for term in meddra_terms:
        if isinstance(term, str) and query.lower() in term.lower():
            if term not in matched_terms:
                results.append({ "original_input": original_input,"query": query,"term": term, "score": 1.0, "matched_from": "MedDRA辞書（用語一致）"})
                matched_terms.add(term)

    # ✅ 3. FAISSベクトル検索
    from helper_functions import encode_query
    query_vector = encode_query(query).astype(np.float32)
    distances, indices = faiss_index.search(np.array([query_vector]), top_k_faiss)
    for i in range(len(indices[0])):
        idx = indices[0][i]
        if idx < len(meddra_terms):
            term_raw = meddra_terms[idx]
            term = term_raw.strip()

            results.append({
                "original_input": original_input,
                "query": query,
                "term": term,
                "score": float(distances[0][i]),
                "matched_from": matched_from_label or " MedDRA辞書（類似語一致）"
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
def extract_score_from_response(line):
    # 例: "1. 7" → 7 を抽出、先頭に数字とピリオドがある形式だけ対象
    import re
    match = re.match(r"^\d+\.\s*([0-9０-９]+)", line.strip())
    if match:
        try:
            return float(match.group(1).replace("０", "0").replace("１", "1").replace("２", "2")
                                         .replace("３", "3").replace("４", "4").replace("５", "5")
                                         .replace("６", "6").replace("７", "7").replace("８", "8")
                                         .replace("９", "9"))
        except:
            pass
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
def rerank_results_batch(original_input, candidates, score_cache=None):
    if score_cache is None:
        score_cache = {}

    top_candidates = candidates.head(10)
    
    # ✅ 拡張語（query）を candidates から取得（必ずここで定義）
    query = candidates["query"].iloc[0] if "query" in candidates.columns else ""
    
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
        prompt = f"""以下の記述「{original_input}」に対して、各用語が症状または疾患の記述内容としてどれだけ意味的に一致しているかを評価してください。
一致度を 0〜10 の数値で記述してください。

「一致」の判断は以下を参考にしてください：
- 10：完全に一致している（意味が同じ）
- 7〜9：かなり近いが微妙に異なる場合もある
- 4〜6：部分的に関連しているが異なる可能性がある
- 1〜3：わずかに関連するが、別のPTの方が明らかに妥当
- 0：まったく無関係

"""
        for idx, term in enumerate(new_terms, 1):
            prompt += f"{idx}. {term}\n"

        prompt += "\n形式：\n1. 7\n2. 5\n... のように記載してください。"
        
        messages = [
            {"role": "system", "content": "あなたは医薬品安全監視業務において、自然言語の症状・疾患記述とMedDRA用語（PT）との意味的一致度を専門的に評価する担当者です。"},
            {"role": "user", "content": prompt}
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0,
            )
            content = response.choices[0].message.content

            # ✅ Streamlitログ表示（必要に応じて有効化）
            import streamlit as st
            # st.subheader("🧾 GPTレスポンス内容（一括形式）")
            # st.code(content)

            for line in content.strip().split("\n"):
                if "." in line:
                    try:
                        idx_part, rest = line.split(".", 1)
                        score_str = rest.split(":")[-1].strip()  # 🔍 「:」以降のスコアだけ取り出す
                        idx = int(idx_part.strip())
                        score = float(score_str)
                        term = new_terms[idx - 1]
                        score_cache[(original_input, term)] = score
                    except Exception as e:
                        st.warning(f"❌ スコア抽出失敗: line='{line}' | error={e}")
                        continue
        except Exception as e:
            # 🧪 デバッグ出力: GPTスコアリングで例外発生（初回の50%原因調査用）
            st.warning(f"❌ GPTスコアリングで例外発生（fallback発動）: {e}")
    
            for term in new_terms:
                score_cache[(original_input, term)] = 5.0  # fallback

    import streamlit as st

    # st.write("🚀 rerank_results_batch() に到達しました")  # 関数が呼ばれているかチェック
    
    # st.subheader("🧪 保存語と表示語の一致・キャッシュHIT確認")

    # new_terms ログ出力
    # st.text("【new_terms（保存対象）】")
    # st.write(new_terms)

    # top_candidates["term"] ログ出力
    # st.text("【top_candidates['term']（取得対象）】")
    # st.write(top_candidates["term"].tolist())

    # score_cache ログ出力
    # st.text("【score_cache の内容】")
    # st.write(score_cache)

    # termごとに一致・HITチェック
    for display_term in top_candidates["term"]:
        key = (original_input, display_term)
        hit = key in score_cache
        match_found = any(
            display_term.strip() == saved_term.strip()
            for saved_term in new_terms
        )
        # st.write(f"{key} → {'✅ HIT' if hit else '❌ MISS'} | term一致: {'✅' if match_found else '❌'}")

    # new_terms の内容確認
    # st.write("🟡 new_terms:", new_terms)

    # if new_terms:
        # st.success("✅ new_terms が存在し、スコア評価ブロックに入りました")
    
    

    # スコアをまとめて返す
    scored = [(term, score_cache.get((original_input, term), 5.0)) for term in top_candidates["term"]]
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
