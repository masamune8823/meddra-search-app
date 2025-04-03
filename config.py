# config.py

import faiss
import numpy as np
import pandas as pd

# FAISS index の読み込み
faiss_index = faiss.read_index("faiss_index.index")

# ベクトルや用語リストの読み込み
meddra_terms = np.load("meddra_terms.npy", allow_pickle=True)

# synonym_df の読み込み（新たに追加）
synonym_df = pd.read_pickle("synonym_df_cat1.pkl")

