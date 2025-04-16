import pandas as pd
import numpy as np
import os

# 入力CSVパス
input_path = "data/MedDRA_Integrated.csv"

# 出力npyパス
output_path = "data/meddra_terms_v2.npy"

# ✅【修正後】対象カラム（variant候補） → 用語候補のすべてのカラムを網羅
variant_columns = [
    "PT_Japanese", "PT_English",
    "LLT_Japanese", "Term_English",
    "HLT_Japanese", "HLGT_Japanese", "SOC_Japanese"
]

# CSV読み込み
df = pd.read_csv(input_path, encoding="utf-8")

# variantリスト構築
variant_list = []
for idx, row in df.iterrows():
    pt_id = row["LLT_or_PT_ID"]
    for col in variant_columns:
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            variant_list.append(f"{str(val).strip()}")

# 重複排除
variant_list = list(set(variant_list))

# 保存（npy形式）
np.save(output_path, np.array(variant_list, dtype=object))
print(f"✅ meddra_terms_v2.npy を保存しました: {output_path}")
print(f"📊 登録語数: {len(variant_list)}")
