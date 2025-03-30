import streamlit as st
import os
import zipfile
import pickle
import numpy as np
import pandas as pd
from utils import search_meddra  # 検索関数は utils.py に定義

# ✅ ZIP展開が必要なファイル（faiss_index, search_assets）
def restore_zip_file(zip_base_name, parts, folder="."):
    zip_path = os.path.join(folder, f"{zip_base_name}.zip")
    with open(zip_path, "wb") as output:
        for part in parts:
            part_path = os.path.join(folder, f"{zip_base_name}_{part}")
            if not os.path.exists(part_path):
                raise FileNotFoundError(f"{part_path} が見つかりません")
            with open(part_path, "rb") as input_file:
                output.write(input_file.read())
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(folder)
    os.remove(zip_path)

# ✅ バイナリ（npyなど）ファイルの単純結合（zip展開なし）
def restore_binary_file(output_name, parts, folder="."):
    output_path = os.path.join(folder, output_name)
    with open(output_path, "wb") as output:
        for part in parts:
            part_path = os.path.join(folder, part)
            if not os.path.exists(part_path):
                raise FileNotFoundError(f"{part_path} が見つかりません")
            with open(part_path, "rb") as input_file:
                output.write(input_file.read())

# ✅ 必要なすべての復元処理を実行
def restore_search_assets():
    restore_zip_file("faiss_index", ["part_a", "part_b"])
    restore_zip_file("search_assets", ["part_a", "part_b", "part_c", "part_d"])
    restore_binary_file("meddra_embeddings.npy", ["meddra_embeddings_part_a", "meddra_embeddings_part_b"])

# ✅ 最初に検索用データを復元
restore_search_assets()

# ✅ マスタデータ読み込み
with open("term_master_df.pkl", "rb") as f:
    term_master_df = pickle.load(f)

# ✅ Streamlit UI
st.title("MedDRA検索アプリ")
query = st.text_input("症状や記述を入力してください", "")

if query:
    results = search_meddra(query, term_master_df)
    st.write("検索結果：")
    for i, row in results.iterrows():
        st.markdown(f"**{i+1}. {row['PT_Japanese']}**")
        st.text(f"確からしさ: {row['score']}%")
        st.text(f"HLT: {row['HLT_Japanese']} | HLGT: {row['HLGT_Japanese']} | SOC: {row['SOC_Japanese']}")
        st.markdown("---")
