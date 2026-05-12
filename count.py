import pandas as pd
import json

# 读取数据
df = pd.read_csv("processed_data/application_test_selected_features.csv")  # 也可以换成 pd.read_excel("data.xlsx")

# 自动识别 categorical 和 numeric 特征
categorical_cols = df.select_dtypes(include=["object", "category"]).columns
numeric_cols = df.select_dtypes(include=["number"]).columns

result = {
    "categorical_features": {},
    "numeric_features": {}
}

# 统计 categorical 特征
for col in categorical_cols:
    counts = df[col].value_counts(dropna=False)
    proportions = df[col].value_counts(normalize=True, dropna=False)

    result["categorical_features"][col] = {
        "num_categories": int(df[col].nunique(dropna=False)),
        "categories": {}
    }

    for category in counts.index:
        if pd.isna(category):
            category_key = "NaN"
        else:
            category_key = str(category)

        result["categorical_features"][col]["categories"][category_key] = {
            "count": int(counts[category]),
            "proportion": float(proportions[category])
        }

# 统计 numeric 特征
total_rows = len(df)

for col in numeric_cols:
    non_nan_count = df[col].notna().sum()
    nan_count = df[col].isna().sum()

    result["numeric_features"][col] = {
        "value_count": int(non_nan_count),
        "value_proportion": float(non_nan_count / total_rows),
        "nan_count": int(nan_count),
        "nan_proportion": float(nan_count / total_rows)
    }

# 保存为 JSON 文件
with open("test_features_summary.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print("结果已保存到 features_summary.json")