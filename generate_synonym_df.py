# generate_synonym_df.py
import pandas as pd
import pickle
import os

# 入力ファイル（Excel）パス
xlsx_path = "data/日本語シノニムV28.0一覧.xlsx"

# 出力ファイルパス
output_path = "data/synonym_df.pkl"

# 読み込みとカラム整形
df = pd.read_excel(xlsx_path)

# カラム存在チェック
if not {"llt_s_kanji", "pt_kanji"}.issubset(df.columns):
    raise ValueError("必要なカラム（llt_s_kanji / pt_kanji）が見つかりません")

# カラム名を統一し、欠損排除
synonym_df = df.rename(columns={
    "llt_s_kanji": "variant",
    "pt_kanji": "PT_Japanese"
})[["variant", "PT_Japanese"]].dropna().query("variant != '' and PT_Japanese != ''").reset_index(drop=True)

# 保存
with open(output_path, "wb") as f:
    pickle.dump(synonym_df, f)

print(f"✅ synonym_df.pkl を作成しました: {output_path}")
