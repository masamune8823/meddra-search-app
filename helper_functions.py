import numpy as np
import pandas as pd
import re
import openai
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def encode_query(text):
    return model.encode([text])[0]

def search_meddra(query, faiss_index, terms, synonym_df, top_k=20):
    queries = [query]
    if query in synonym_df.index:
        synonyms = synonym_df.loc[query]["synonyms"]
        if isinstance(synonyms, str):
            synonyms = [synonyms]
        queries += synonyms

    results = []
    for q in queries:
        q_vec = encode_query(q)
        D, I = faiss_index.search(np.array([q_vec]), top_k)
        for i, d in zip(I[0], D[0]):
            results.append({"term": terms[i], "score": float(1 - d)})
    return pd.DataFrame(results)

def rerank_results_v13(query, df, top_n=10):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt_template = lambda term: f"以下の症状にどれだけ関連があるかを100点満点で評価してください：{query}
候補：{term}"

    reranked = []
    for _, row in df.head(top_n).iterrows():
        prompt = prompt_template(row['term'])
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "あなたは医療専門家です。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
            )
            score_text = response["choices"][0]["message"]["content"]
            score = int(re.search(r"\d+", score_text).group())
        except:
            score = 50
        reranked.append({"term": row["term"], "score": score})
    return pd.DataFrame(reranked)

def add_hierarchy_info(df, term_master_df):
    df["term_normalized"] = df["term"].str.lower().str.strip()
    term_master_df["term_normalized"] = term_master_df["PT_Japanese"].str.lower().str.strip()
    return df.merge(term_master_df, on="term_normalized", how="left")

def rescale_scores(df):
    if "score" not in df.columns or df.empty:
        return df
    max_score = df["score"].max()
    min_score = df["score"].min()
    if max_score == min_score:
        df["score"] = 100
    else:
        df["score"] = (df["score"] - min_score) / (max_score - min_score) * 100
    return df

def predict_soc_keywords_with_gpt(query):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "あなたは医療分野に詳しいAIです。与えられた症状に関連するMedDRAのSOCカテゴリを3つ、日本語で簡潔に予測してください。"},
            {"role": "user", "content": f"以下の症状に関連するSOCカテゴリを教えてください：\n\n{query}"}
        ],
        temperature=0.3,
    )
    text = response["choices"][0]["message"]["content"]
    keywords = [kw.strip("・ 、。
") for kw in text.split() if kw.strip()]
    return keywords[:3]

def clean_predicted_keywords(raw_list):
    cleaned = []
    for item in raw_list:
        item = item.strip()
        item = re.sub(r'^\d+[\.\)\:\-]?', '', item)
        item = item.split("：")[0].split(":")[0]
        if item:
            cleaned.append(item.strip())
    return list(set(cleaned))
