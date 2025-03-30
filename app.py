import streamlit as st
import os
import zipfile
from utils import search_meddra  # 検索関数（別ファイル）
import pickle

def restore_split_file(zip_base_name, parts, folder="."):
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

def restore_search_assets():
    restore_split_file("faiss_index", ["part_a", "part_b"])
    restore_split_file("meddra_embeddings", ["part_a", "part_b"])
    restore_split_file("search_assets", ["part_a", "part_b", "part_c", "part_d"])

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
