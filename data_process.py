"""
Home Credit Default Risk - Preprocessing data
Outputs:
- A preprocessed CSV file ready for modelling
- A preprocessing report JSON file
- A preprocessing summary CSV file
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

# ============================================================
# 1. CONFIG: set initially
# ============================================================
# Input
INPUT_FILE = "raw_data/application_train.csv"
# Output
OUTPUT_FILE = "processed_data/application_train_processed.csv"
REPORT_FILE = "processed_data/process_report.json"
SUMMARY_FILE = "processed_data/process_summary.csv"

IS_TRAIN = True # If it is train set

# Rare category threshold. Categories with frequency lower than 0.01 will become "Rare".
RARE_THRESHOLD = 0.01
# Create missing indicators for columns with missing rate >= this value.
MISSING_INDICATOR_THRESHOLD = 0.05
# Drop columns with missing rate >= this value.
DROP_HIGH_MISSING_THRESHOLD = 0.85
# Outlier handle
CLIP_QUANTILES = (0.01, 0.99)

def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:  
    # If the denominator is 0, it is replaced with NaN to avoid infinity.
    return numerator / denominator.replace({0: np.nan})

# ============================================================
# 2. Main preprocessing function
# ============================================================

def preprocess_application_df(
    df: pd.DataFrame,
    is_train: bool = True,
    rare_threshold: float = RARE_THRESHOLD,
    missing_indicator_threshold: float = MISSING_INDICATOR_THRESHOLD,
    clip_quantiles=CLIP_QUANTILES,
    drop_high_missing_threshold: float = DROP_HIGH_MISSING_THRESHOLD,
):
    data = df.copy()
    report = {}   # JSON report
    report["initial_shape"] = list(data.shape)

    # --------------------------------------------------------
    # Step 1: Remove duplicate rows
    # --------------------------------------------------------
    duplicate_rows = int(data.duplicated().sum())
    if duplicate_rows > 0:
        data = data.drop_duplicates()
    report["duplicate_rows_removed"] = duplicate_rows

    # --------------------------------------------------------
    # Step 2: Separate ID and target columns
    # --------------------------------------------------------
    id_cols = [c for c in ["SK_ID_CURR"] if c in data.columns]
    target_cols = [c for c in ["TARGET"] if c in data.columns and is_train]

    ids = data[id_cols].copy() if id_cols else pd.DataFrame(index=data.index)
    target = data[target_cols].copy() if target_cols else pd.DataFrame(index=data.index)
    # get candidate features
    features = data.drop(columns=id_cols + target_cols, errors="ignore")

    # --------------------------------------------------------
    # Step 3: Handle known anomalous value in DAYS_EMPLOYED
    # --------------------------------------------------------
    # In this dataset, 365243 in DAYS_EMPLOYED is a known placeholder/anomaly.
    # We create an indicator and then replace the anomalous value with NaN.
    if "DAYS_EMPLOYED" in features.columns:
        features["DAYS_EMPLOYED_ANOMALY"] = (
            features["DAYS_EMPLOYED"] == 365243
        ).astype(int)   # if is, 1

        report["days_employed_365243_count"] = int(
            (features["DAYS_EMPLOYED"] == 365243).sum()
        )
        # NaN
        features.loc[features["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
    else:
        report["days_employed_365243_count"] = 0

    # --------------------------------------------------------
    # Step 4: Domain-specific credit risk features
    # --------------------------------------------------------
    # 本次申请的贷款金额 / 总收入
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(features.columns):
        features["CREDIT_INCOME_RATIO"] = safe_divide(
            features["AMT_CREDIT"], features["AMT_INCOME_TOTAL"]
        )
    # 本次申请的贷款金额 / 本次贷款的年金
    if {"AMT_CREDIT", "AMT_ANNUITY"}.issubset(features.columns):
        features["CREDIT_ANNUITY_RATIO"] = safe_divide(
            features["AMT_CREDIT"], features["AMT_ANNUITY"]
        )
    # 贷款所购商品价格 / 本次申请的贷款金额
    if {"AMT_GOODS_PRICE", "AMT_CREDIT"}.issubset(features.columns):
        features["GOODS_CREDIT_RATIO"] = safe_divide(
            features["AMT_GOODS_PRICE"], features["AMT_CREDIT"]
        )
    # 总收入 / 家庭人数
    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(features.columns):
        features["INCOME_PER_FAMILY_MEMBER"] = safe_divide(
            features["AMT_INCOME_TOTAL"], features["CNT_FAM_MEMBERS"]
        )
    # 孩子数量 / 家庭人数
    if {"CNT_CHILDREN", "CNT_FAM_MEMBERS"}.issubset(features.columns):
        features["CHILDREN_FAMILY_RATIO"] = safe_divide(
            features["CNT_CHILDREN"], features["CNT_FAM_MEMBERS"]
        )
    # 工作天数 / 出生天数
    if {"DAYS_EMPLOYED", "DAYS_BIRTH"}.issubset(features.columns):
        features["EMPLOYED_TO_AGE_RATIO"] = safe_divide(
            features["DAYS_EMPLOYED"], features["DAYS_BIRTH"]
        )

    # --------------------------------------------------------
    # Step 5: Aggregate EXT_SOURCE external score features
    # --------------------------------------------------------
    ext_cols = [
        col for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        if col in features.columns
    ]

    if ext_cols:
        features["EXT_SOURCE_MEAN"] = features[ext_cols].mean(axis=1)
        features["EXT_SOURCE_STD"] = features[ext_cols].std(axis=1)
        features["EXT_SOURCE_N_MISSING"] = features[ext_cols].isna().sum(axis=1)
        features["EXT_SOURCE_PRODUCT"] = features[ext_cols].prod(axis=1)
    # condisder if drop

    # --------------------------------------------------------
    # Step 6: Aggregate document flag features
    # --------------------------------------------------------
    doc_cols = [col for col in features.columns if col.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        features["DOCUMENT_FLAGS_SUM"] = features[doc_cols].sum(axis=1)
    features = features.drop(columns=doc_cols, errors="ignore")

    # --------------------------------------------------------
    # Step 7: Aggregate contact information features
    # --------------------------------------------------------
    contact_cols = [
        col for col in [
            "FLAG_MOBIL",
            "FLAG_EMP_PHONE",
            "FLAG_WORK_PHONE",
            "FLAG_CONT_MOBILE",
            "FLAG_PHONE",
            "FLAG_EMAIL",
        ]
        if col in features.columns
    ]

    if contact_cols:
        features["CONTACT_FLAGS_SUM"] = features[contact_cols].sum(axis=1)
    features = features.drop(columns=contact_cols, errors="ignore")

    # --------------------------------------------------------
    # Step 8: Create missingness indicators
    # --------------------------------------------------------
    missing_rates = features.isna().mean()
    missing_indicator_cols = missing_rates[
        missing_rates >= missing_indicator_threshold
    ].index.tolist()

    if missing_indicator_cols:
        missing_indicators = pd.DataFrame(
            {
                f"{col}_MISSING": features[col].isna().astype(int)
                for col in missing_indicator_cols
            },
            index=features.index,
        )
        features = pd.concat([features, missing_indicators], axis=1)

    report["missing_indicator_columns_created"] = len(missing_indicator_cols)
    report["missing_indicator_columns"] = missing_indicator_cols

    # --------------------------------------------------------
    # Step 9: Drop extremely high-missing columns
    # --------------------------------------------------------
    high_missing_cols = missing_rates[
        missing_rates >= drop_high_missing_threshold
    ].index.tolist()

    if high_missing_cols:
        features = features.drop(columns=high_missing_cols, errors="ignore")

    report["high_missing_columns_dropped"] = high_missing_cols

    # --------------------------------------------------------
    # Step 10: Outlier clipping and log transformation
    # --------------------------------------------------------
    amount_like_cols = [
        col for col in features.columns
        if col.startswith("AMT_") or col in [
            "CNT_CHILDREN",
            "CNT_FAM_MEMBERS",
            "OWN_CAR_AGE",
            "CREDIT_INCOME_RATIO",
            "CREDIT_ANNUITY_RATIO",
            "INCOME_PER_FAMILY_MEMBER",
        ]
    ]

    amount_like_cols = [
        col for col in amount_like_cols
        if col in features.columns and pd.api.types.is_numeric_dtype(features[col])
    ]

    clipped_columns = {}
    log_features = {}
    q_low, q_high = clip_quantiles

    for col in amount_like_cols:
        non_missing = features[col].dropna()

        if len(non_missing) > 10:
            lower, upper = non_missing.quantile([q_low, q_high]).values

            if np.isfinite(lower) and np.isfinite(upper) and lower < upper:
                outlier_count = int(((features[col] < lower) | (features[col] > upper)).sum())
                features[col] = features[col].clip(lower, upper)

                if outlier_count > 0:
                    clipped_columns[col] = {
                        "lower": float(lower),
                        "upper": float(upper),
                        "values_clipped": outlier_count,
                    }

                # Add log1p features for non-negative numerical variables.
                if features[col].min(skipna=True) >= 0:
                    log_features[f"LOG1P_{col}"] = np.log1p(features[col])

    if log_features:
        log_df = pd.DataFrame(log_features, index=features.index)
        features = pd.concat([features, log_df], axis=1)

    report["clipped_columns"] = clipped_columns
    report["log_features_created"] = list(log_features.keys())

    # --------------------------------------------------------
    # Step 11: Categorical processing
    # - Fill missing values with "Missing"
    # - Group rare categories into "Rare"
    # - Create frequency encoding features
    # - Apply one-hot encoding
    # --------------------------------------------------------
    categorical_cols = features.select_dtypes(include=["object", "category"]).columns.tolist()
    frequency_features = {}
    rare_category_report = {}

    for col in categorical_cols:
        features[col] = features[col].astype("object").where(
            features[col].notna(), "Missing"
        )

        freq_before = features[col].value_counts(normalize=True, dropna=False)
        rare_values = freq_before[freq_before < rare_threshold].index.tolist()

        if rare_values:
            features[col] = features[col].where(~features[col].isin(rare_values), "Rare")

        rare_category_report[col] = [str(x) for x in rare_values]

        freq_after = features[col].value_counts(normalize=True, dropna=False)
        frequency_features[f"FREQ_{col}"] = features[col].map(freq_after).astype(float)

    if frequency_features:
        freq_df = pd.DataFrame(frequency_features, index=features.index)
        features = pd.concat([features, freq_df], axis=1)

    # --------------------------------------------------------
    # Step 12: Numerical missing value imputation
    # --------------------------------------------------------
    features = features.replace([np.inf, -np.inf], np.nan)

    numeric_cols = features.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    imputed_numeric_cols = []

    for col in numeric_cols:
        if features[col].isna().any():
            median_value = features[col].median()
            if pd.isna(median_value):
                median_value = 0
            features[col] = features[col].fillna(median_value)
            imputed_numeric_cols.append(col)

    report["numeric_columns_imputed_with_median"] = imputed_numeric_cols

    # --------------------------------------------------------
    # Step 13: One-hot encoding for categorical columns
    # --------------------------------------------------------
    if categorical_cols:
        features = pd.get_dummies(
            features,
            columns=categorical_cols,
            dummy_na=False,
            drop_first=False,
        )

    # Convert boolean columns to 0/1 integers.
    bool_cols = features.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        features[bool_cols] = features[bool_cols].astype(np.uint8)

    # Final cleaning: no missing values or infinite values.
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

    report["categorical_columns_encoded"] = categorical_cols
    report["rare_categories_grouped"] = rare_category_report
    report["remaining_missing_after_processing"] = int(features.isna().sum().sum())
    report["final_feature_shape"] = list(features.shape)

    # --------------------------------------------------------
    # Step 14: Combine ID, TARGET, and processed features
    # --------------------------------------------------------
    processed_df = pd.concat(
        [
            ids.reset_index(drop=True),
            target.reset_index(drop=True),
            features.reset_index(drop=True),
        ],
        axis=1,
    )

    report["final_output_shape"] = list(processed_df.shape)
    
    return processed_df, report

# ============================================================
# 3. Create summary file
# ============================================================

def create_summary(raw_df: pd.DataFrame, processed_df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    Create a simple summary table for reports or presentations.
    """
    summary_items = {
        "input_rows": raw_df.shape[0],
        "input_columns": raw_df.shape[1],
        "output_rows": processed_df.shape[0],
        "output_columns": processed_df.shape[1],
        "duplicate_rows_removed": report.get("duplicate_rows_removed", 0),
        "days_employed_365243_count": report.get("days_employed_365243_count", 0),
        "missing_indicator_columns_created": report.get("missing_indicator_columns_created", 0),
        "high_missing_columns_dropped_count": len(report.get("high_missing_columns_dropped", [])),
        "clipped_columns_count": len(report.get("clipped_columns", {})),
        "log_features_created_count": len(report.get("log_features_created", [])),
        "categorical_columns_encoded_count": len(report.get("categorical_columns_encoded", [])),
        "remaining_missing_after_processing": report.get("remaining_missing_after_processing", None),
    }

    if "TARGET" in raw_df.columns:
        summary_items["target_0_count"] = int((raw_df["TARGET"] == 0).sum())
        summary_items["target_1_count"] = int((raw_df["TARGET"] == 1).sum())
        summary_items["target_1_rate"] = float((raw_df["TARGET"] == 1).mean())

    return pd.DataFrame(
        {"item": list(summary_items.keys()), "value": list(summary_items.values())}
    )

def main():
    script_dir = Path(__file__).resolve().parent
    input_path = script_dir / INPUT_FILE
    output_path = script_dir / OUTPUT_FILE
    report_path = script_dir / REPORT_FILE
    summary_path = script_dir / SUMMARY_FILE

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            f"Please put {INPUT_FILE} in the same folder as this Python file, "
            f"or change INPUT_FILE in the CONFIG section."
        )

    print("Loading data...")
    raw_df = pd.read_csv(input_path)
    print(f"Raw data shape: {raw_df.shape}")

    print("Preprocessing data...")
    processed_df, report = preprocess_application_df(raw_df, is_train=IS_TRAIN)
    print(f"Processed data shape: {processed_df.shape}")

    print("Saving processed CSV...")
    processed_df.to_csv(output_path, index=False)

    print("Saving preprocessing report JSON...")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Saving preprocessing summary CSV...")
    summary_df = create_summary(raw_df, processed_df, report)
    summary_df.to_csv(summary_path, index=False)

    print("\nDone!")
    print(f"Processed file saved to: {output_path}")
    print(f"Report file saved to:    {report_path}")
    print(f"Summary file saved to:   {summary_path}")

if __name__ == "__main__":
    main()
