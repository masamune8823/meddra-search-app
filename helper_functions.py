
def filter_by_predicted_soc(df, keywords):
    if not keywords:
        return df

    cols = ["SOC", "HLGT", "HLT"]
    existing_cols = [col for col in cols if col in df.columns]

    if not existing_cols:
        print("⚠️ SOC/HLGT/HLT列が見つかりません。フィルタをスキップします。")
        return df

    mask = df[existing_cols].apply(
        lambda x: x.astype(str).str.contains("|".join(keywords)).any(), axis=1
    )
    return df[mask].copy()
