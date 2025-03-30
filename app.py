import os
import zipfile
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import faiss
from utils import expand_query_gpt, encode_query, rerank_results_v13

# 分割ファイルの復元
def restore_split_file(output_path, parts, folder="."):
    with open(os.path.join(folder, output_path), "wb") as outfile:
        for part in parts:
            part_path = os.path.join(folder, f"{output_path}_part_{part}")
            if os.path.exists(part_path):
                with open(part_path, "rb") as infile:
                    outfile.write(infile.read())
            else:
                raise FileNotFoundError(f"{part_path} が見つかりません")

# search_assets.zip の復元と展開
def restore_search_assets():
    zip_path = "search_assets.zip"
    parts = ["a", "b", "c", "d"]
    restore_split_file(zip_path, parts, folder=".")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")  # ZIP内のファイルを現在のフォルダへ展開

# meddra_embeddings.npy の復元
def restore_embeddings():
    output_path = "meddra_embeddings.npy"
    parts = ["a", "b"]
    restore_split_file(output_path, parts, folder=".")

# 呼び出し
restore_search_assets()
restore_embeddings()

# データとFAISSの読み込み（全てルート直下）
@st.cache_resource
def load_faiss_and_data():
    index = faiss.read_index("faiss_index.index")
    embeddings = np.load("meddra_embeddings.npy")
    with open("meddra_terms.npy", "rb") as f_
