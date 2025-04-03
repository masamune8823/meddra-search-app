# config.py

import faiss
import numpy as np
import os

# パス設定（必要に応じて修正）
BASE_PATH = "/mount/src/meddra-search-app"

# FAISSインデックスとベクトルの読み込み
faiss_index_path = os.path.join(BASE_PATH, "faiss_index.index")
meddra_embeddings_path = os.path.join(BASE_PATH, "meddra_embeddings.npy")
meddra_terms_path = os.path.join(BASE_PATH, "meddra_terms.npy")

# FAISSインデックスの読み込み
faiss_index = faiss.read_index(faiss_index_path)

# ベクトルと用語の読み込み
meddra_embeddings = np.load(meddra_embeddings_path)
meddra_terms = np.load(meddra_terms_path, allow_pickle=True)
