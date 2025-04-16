import pandas as pd
import pickle
import os

# 入力元Excel
xlsx_path = "data/日本語シノニムV28.0一覧.xlsx"

# 出力先
output_path = "data/synonym_df.pkl"

# Excel読み込み
df = pd.read_excel(xlsx_path)

# ✅ 統合対象カラム（variant候補）と PTカラム
variant_cols = ["llt_s_kanji", "llt_kanji", "llt_name", "pt_kanji", "pt_name"]
pt_col = "pt_kanji"  # PT_Japaneseとして使う列

# カラムチェック
missing = [col for col in variant_cols + [pt_col] if col not in df.columns]
if missing:
    raise ValueError(f"❌ 欠損カラム: {missing}")

# ✅ 各variant列を横断して「variant⇔PT_Japanese」辞書構造を構築
records = []

for _, row in df.iterrows():
    pt_japanese = row[pt_col]
    if pd.isna(pt_japanese) or str(pt_japanese).strip() == "":
        continue

    for col in variant_cols:
        val = row[col]
        if pd.isna(val) or str(val).strip() == "":
            continue

        records.append({
            "variant": str(val).strip(),
            "PT_Japanese": str(pt_japanese).strip()
        })

# ✅ DataFrame化＋重複排除
synonym_df = pd.DataFrame(records).drop_duplicates().reset_index(drop=True)

# ✅ 保存
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "wb") as f:
    pickle.dump(synonym_df, f)

print(f"✅ synonym_df.pkl を再生成しました（件数: {len(synonym_df)}）")
