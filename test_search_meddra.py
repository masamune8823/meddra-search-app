
import pandas as pd
import numpy as np
import pickle
import faiss
from helper_functions import search_meddra

# データの読み込み
index_path = "/mnt/data/faiss_index.index"
terms_path = "/mnt/data/meddra_terms.npy"
embed_path = "/mnt/data/meddra_embeddings.npy"
synonym_path = "/mnt/data/synonym_df_cat1.pkl"

faiss_index = faiss.read_index(index_path)
meddra_terms = np.load(terms_path, allow_pickle=True)
meddra_embeddings = np.load(embed_path)

with open(synonym_path, "rb") as f:
    synonym_df = pickle.load(f)

# テストクエリ
test_query = "ズキズキ"

# FAISS + synonym検索
results = search_meddra(test_query, faiss_index, meddra_terms, synonym_df, top_k=10)

# 結果出力
print("🔍 検索結果:")
print(results[["term", "score"]])
