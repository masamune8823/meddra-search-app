import faiss
import numpy as np
import pandas as pd

# FAISS index の読み込み
faiss_index = faiss.read_index("/mount/src/meddra-search-app/faiss_index.index")

# ベクトルや用語リストの読み込み
meddra_terms = np.load("/mount/src/meddra-search-app/meddra_terms.npy", allow_pickle=True)

# 日本語シノニム辞書の読み込み（synonym_df）
synonym_df = pd.read_pickle("/mount/src/meddra-search-app/synonym_df_cat1.pkl")
