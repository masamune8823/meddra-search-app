import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

# 入力パス
terms_path = "data/meddra_terms_v2.npy"

# 出力ファイル
embedding_output_path = "data/meddra_embeddings_v2.npy"
faiss_index_path = "data/faiss_index_v2.index"

# 1. モデル読み込み（MiniLM推奨）
model = SentenceTransformer("model_miniLM")

# 2. 用語読み込み
terms = np.load(terms_path, allow_pickle=True)
print(f"✅ 用語数: {len(terms)}")

# 3. ベクトル化
print("🔄 ベクトル化中...")
embeddings = model.encode(terms.tolist(), show_progress_bar=True)

# 4. FAISSインデックス作成
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 5. 保存
np.save(embedding_output_path, embeddings)
faiss.write_index(index, faiss_index_path)

print(f"✅ ベクトル保存: {embedding_output_path}")
print(f"✅ FAISSインデックス保存: {faiss_index_path}")
