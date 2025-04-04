import os
import pickle
import numpy as np
import pandas as pd
from helper_functions import (
    encode_query,
    search_meddra,
    rerank_results_v13,
    add_hierarchy_info,
    rescale_scores
)

# ファイルパス設定
DATA_DIR = "data"
index_path = os.path.join(DATA_DIR, "faiss_index.index")
terms_path = os.path.join(DATA_DIR, "meddra_terms.npy")
embed_path = os.path.join(DATA_DIR, "meddra_embeddings.npy")
synonym_path = os.path.join(DATA_DIR, "synonym_df_cat1.pkl")
hierarchy_path = os.path.join(DATA_DIR, "term_master_df.pkl")

# データ読み込み
faiss_index = None
meddra_terms = None
synonym_df = None
term_master_df = None

try:
    import faiss
    faiss_index = faiss.read_index(index_path)
    meddra_terms = np.load(terms_path, allow_pickle=True)
    with open(synonym_path, "rb") as f:
        synonym_df = pickle.load(f)
    with open(hierarchy_path, "rb") as f:
        term_master_df = pickle.load(f)
except Exception as e:
    print(f"❌ データ読み込みエラー: {e}")
    exit(1)

# テストクエリ
test_query = "ズキズキ"

# 検索 → 再スコア → 階層付加 → スコア整形
try:
    results = search_meddra(test_query, faiss_index, meddra_terms, synonym_df, top_k=20)
    reranked = rerank_results_v13(test_query, results, top_n=10)
    final_results = add_hierarchy_info(reranked, term_master_df)
    final_results = rescale_scores(final_results)
    final_results = final_results.rename(columns={
        "term": "用語",
        "score": "確からしさ（％）",
        "HLT_Japanese": "HLT",
        "HLGT_Japanese": "HLGT",
        "SOC_Japanese": "SOC"
    })[["用語", "確からしさ（％）", "HLT", "HLGT", "SOC"]]
    print("✅ 上位10件の検索結果：")
    print(final_results)
except Exception as e:
    print(f"❌ 処理中にエラーが発生しました: {e}")
