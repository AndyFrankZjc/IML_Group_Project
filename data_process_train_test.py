import os
import pickle
import numpy as np
import pandas as pd


# ============================================================
# 0. 运行模式和路径设置
# ============================================================
# mode = "train"：在训练集上 fit 预处理规则，并保存规则
# mode = "test" ：在测试集上只 transform，直接复用训练集保存的规则
mode = "test"  # 改成 "test" 即可处理测试集

train_input_path = "processed_data/application_train_selected_features.csv"
train_output_path = "processed_data/application_train_processed.csv"

test_input_path = "processed_data/application_test_selected_features.csv"
test_output_path = "processed_data/application_test_processed.csv"

# 保存所有训练集确定的预处理规则
config_path = "processed_data/preprocess_config.pkl"

# 单独保存各类 encoding 字典，方便之后自己调用
binary_encoding_path = "processed_data/binary_encoding_dict.pkl"
one_hot_encoding_path = "processed_data/one_hot_encoding_dict.pkl"
frequency_encoding_path = "processed_data/frequency_encoding_dict.pkl"
rare_categories_path = "processed_data/rare_categories_dict.pkl"


# ============================================================
# 1. 通用函数
# ============================================================
def safe_divide(numerator, denominator):
    """
    安全除法。
    如果分母为 0，则先替换为 NaN，避免产生 inf 或 -inf。
    """
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def add_engineered_features(df):
    """
    构造 EXT_SOURCE 统计特征和比例特征。
    该函数 train/test 共用，保证两边生成同样的衍生列。
    """
    df = df.copy()

    # =========================
    # EXT_SOURCE 相关特征
    # =========================
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    existing_ext_cols = [col for col in ext_cols if col in df.columns]

    if len(existing_ext_cols) > 0:
        df["EXT_SOURCE_MEAN"] = df[existing_ext_cols].mean(axis=1, skipna=True)
        # ddof=0 可以避免只有一个非缺失值时 std 为 NaN
        df["EXT_SOURCE_STD"] = df[existing_ext_cols].std(axis=1, skipna=True, ddof=0)
        df["EXT_SOURCE_N_MISSING"] = df[existing_ext_cols].isna().sum(axis=1)
        # min_count=1 可以避免三个 EXT_SOURCE 全缺失时 product 被算成 1
        df["EXT_SOURCE_PRODUCT"] = df[existing_ext_cols].prod(axis=1, skipna=True, min_count=1)

    # =========================
    # 比例特征
    # =========================
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["CREDIT_INCOME_RATIO"] = safe_divide(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])

    if {"AMT_CREDIT", "AMT_ANNUITY"}.issubset(df.columns):
        df["CREDIT_ANNUITY_RATIO"] = safe_divide(df["AMT_CREDIT"], df["AMT_ANNUITY"])

    if {"AMT_GOODS_PRICE", "AMT_CREDIT"}.issubset(df.columns):
        df["GOODS_CREDIT_RATIO"] = safe_divide(df["AMT_GOODS_PRICE"], df["AMT_CREDIT"])

    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(df.columns):
        df["INCOME_PER_FAMILY_MEMBER"] = safe_divide(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])

    if {"CNT_CHILDREN", "CNT_FAM_MEMBERS"}.issubset(df.columns):
        df["CHILDREN_FAMILY_RATIO"] = safe_divide(df["CNT_CHILDREN"], df["CNT_FAM_MEMBERS"])

    if {"DAYS_EMPLOYED", "DAYS_BIRTH"}.issubset(df.columns):
        df["EMPLOYED_TO_AGE_RATIO"] = safe_divide(df["DAYS_EMPLOYED"], df["DAYS_BIRTH"])

    return df


def split_id_target(df):
    """
    将 SK_ID_CURR 和 TARGET 隔离出来。
    后续所有预处理只作用于特征列。
    """
    keep_cols = []

    if "SK_ID_CURR" in df.columns:
        keep_cols.append("SK_ID_CURR")

    if "TARGET" in df.columns:
        keep_cols.append("TARGET")

    id_target_df = df[keep_cols].copy()
    X = df.drop(columns=keep_cols).copy()

    return X, id_target_df, keep_cols


def final_check_and_fill(X):
    """
    最后确保没有 missing values 或 infinite values。
    """
    X = X.replace([np.inf, -np.inf], np.nan)

    for col in X.columns:
        if X[col].isna().sum() > 0:
            if pd.api.types.is_numeric_dtype(X[col]):
                fill_value = X[col].median()
                if pd.isna(fill_value):
                    fill_value = 0
                X[col] = X[col].fillna(fill_value)
            else:
                mode_value = X[col].mode(dropna=True)
                if mode_value.empty:
                    X[col] = X[col].fillna("UNKNOWN")
                else:
                    X[col] = X[col].fillna(mode_value[0])

    return X


# ============================================================
# 2. 训练集：fit + transform
# ============================================================
def process_train():
    df = pd.read_csv(train_input_path)
    print("原始训练集形状:", df.shape)

    # =========================
    # 1. 去除重复行
    # =========================
    df = df.drop_duplicates().reset_index(drop=True)
    print("去重后训练集形状:", df.shape)

    # =========================
    # 2. 隔离 SK_ID_CURR 和 TARGET
    # =========================
    X, id_target_df, keep_cols = split_id_target(df)
    print("隔离出的列:", keep_cols)
    print("进入预处理的训练特征形状:", X.shape)

    # =========================
    # 3. 将 DAYS_EMPLOYED 中的 365243 替换为 NaN
    # =========================
    if "DAYS_EMPLOYED" in X.columns:
        X["DAYS_EMPLOYED"] = X["DAYS_EMPLOYED"].replace(365243, np.nan)

    # =========================
    # 4. missing indicator
    #    只根据训练集缺失率 >= 15% 的特征决定 indicator_cols
    #    test set 之后直接复用 indicator_cols
    # =========================
    missing_rate = X.isna().mean()
    indicator_cols = missing_rate[missing_rate >= 0.15].index.tolist()

    for col in indicator_cols:
        X[col + "_MISSING_INDICATOR"] = X[col].isna().astype(int)

    print("创建 missing indicator 的特征:")
    print(indicator_cols)

    # =========================
    # 5. 构造衍生特征
    # =========================
    X = add_engineered_features(X)

    # =========================
    # 6. 对 4 个 AMT 特征做 1% - 99% 分位数截断
    #    截断边界只在训练集计算，之后保存给 test set 用
    # =========================
    amt_clip_cols = [
        "AMT_CREDIT",
        "AMT_INCOME_TOTAL",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE"
    ]

    clip_bounds = {}

    for col in amt_clip_cols:
        if col in X.columns:
            lower = X[col].quantile(0.01)
            upper = X[col].quantile(0.99)
            clip_bounds[col] = {"lower": lower, "upper": upper}
            X[col] = X[col].clip(lower=lower, upper=upper)

    # =========================
    # 7. 区分数值型和类别型特征
    # =========================
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    # =========================
    # 8. 缺失值填补
    #    numerical: median
    #    categorical: most frequency
    #    填充值只在训练集计算，之后保存给 test set 用
    # =========================
    numerical_impute_dict = {}
    categorical_impute_dict = {}

    for col in numerical_cols:
        median_value = X[col].median()
        if pd.isna(median_value):
            median_value = 0
        numerical_impute_dict[col] = median_value
        X[col] = X[col].fillna(median_value)

    for col in categorical_cols:
        mode_value = X[col].mode(dropna=True)
        if mode_value.empty:
            most_freq_value = "UNKNOWN"
        else:
            most_freq_value = mode_value[0]
        categorical_impute_dict[col] = most_freq_value
        X[col] = X[col].fillna(most_freq_value)

    # =========================
    # 9. Categorical 转数值
    #    train 负责确定所有编码规则，并保存：
    #    binary_encoding_dict
    #    one_hot_encoding_dict
    #    frequency_encoding_dict
    #    rare_categories_dict
    # =========================
    binary_encoding_dict = {}
    one_hot_encoding_dict = {}
    frequency_encoding_dict = {}
    rare_categories_dict = {}
    constant_encoding_dict = {}

    one_hot_cols = []

    for col in categorical_cols:
        n_unique = X[col].nunique(dropna=False)

        # -------------------------
        # 2 类：映射为 0 和 1
        # -------------------------
        if n_unique == 2:
            unique_values = sorted(X[col].unique(), key=lambda x: str(x))
            mapping = {
                unique_values[0]: 0,
                unique_values[1]: 1
            }
            binary_encoding_dict[col] = mapping
            X[col] = X[col].map(mapping).astype(int)

        # -------------------------
        # 3 到 7 类：one-hot encoding
        # -------------------------
        elif 3 <= n_unique <= 7:
            one_hot_cols.append(col)

        # -------------------------
        # 7 类以上：RARE 合并 + frequency encoding
        # -------------------------
        elif n_unique > 7:
            freq = X[col].value_counts(normalize=True)

            rare_categories = freq[freq < 0.02].index.tolist()
            rare_categories_dict[col] = rare_categories

            X[col] = X[col].replace(rare_categories, "RARE")

            freq_after_rare = X[col].value_counts(normalize=True)
            freq_mapping = freq_after_rare.to_dict()

            frequency_encoding_dict[col] = freq_mapping
            X[col] = X[col].map(freq_mapping).astype(float)

        # -------------------------
        # 只有 1 类：直接转成 0
        # -------------------------
        else:
            constant_encoding_dict[col] = 0
            X[col] = 0

    # =========================
    # 10. 对 3 到 7 类特征做 one-hot encoding
    #     并保存 train 中实际生成的 dummy 列名
    # =========================
    for col in one_hot_cols:
        dummies = pd.get_dummies(
            X[col],
            prefix=col,
            drop_first=False,
            dtype=int
        )

        one_hot_encoding_dict[col] = {
            "dummy_columns": dummies.columns.tolist()
        }

        X = pd.concat(
            [X.drop(columns=[col]), dummies],
            axis=1
        )

    # =========================
    # 11. 最终确保没有 missing values 或 infinite values
    # =========================
    X = final_check_and_fill(X)

    # 保存最终训练特征列顺序；test 最后必须对齐到这个顺序
    final_feature_cols = X.columns.tolist()

    # =========================
    # 12. 保存所有 train 规则
    # =========================
    preprocess_config = {
        "keep_cols": keep_cols,
        "indicator_cols": indicator_cols,
        "clip_bounds": clip_bounds,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "numerical_impute_dict": numerical_impute_dict,
        "categorical_impute_dict": categorical_impute_dict,
        "binary_encoding_dict": binary_encoding_dict,
        "one_hot_encoding_dict": one_hot_encoding_dict,
        "frequency_encoding_dict": frequency_encoding_dict,
        "rare_categories_dict": rare_categories_dict,
        "constant_encoding_dict": constant_encoding_dict,
        "final_feature_cols": final_feature_cols
    }

    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, "wb") as f:
        pickle.dump(preprocess_config, f)

    # 这些单独保存出来，是为了你之后可以自己直接读取使用
    with open(binary_encoding_path, "wb") as f:
        pickle.dump(binary_encoding_dict, f)

    with open(one_hot_encoding_path, "wb") as f:
        pickle.dump(one_hot_encoding_dict, f)

    with open(frequency_encoding_path, "wb") as f:
        pickle.dump(frequency_encoding_dict, f)

    with open(rare_categories_path, "wb") as f:
        pickle.dump(rare_categories_dict, f)

    print("预处理规则已保存至:", config_path)
    print("binary encoding dict 已保存至:", binary_encoding_path)
    print("one-hot encoding dict 已保存至:", one_hot_encoding_path)
    print("frequency encoding dict 已保存至:", frequency_encoding_path)
    print("rare categories dict 已保存至:", rare_categories_path)

    # =========================
    # 13. 加回 SK_ID_CURR 和 TARGET
    # =========================
    df_encoded = pd.concat(
        [id_target_df.reset_index(drop=True), X.reset_index(drop=True)],
        axis=1
    )

    # =========================
    # 14. 最终检查并保存
    # =========================
    numeric_df = df_encoded.select_dtypes(include=[np.number])
    print("最终训练集形状:", df_encoded.shape)
    print("最终 missing values 数量:", df_encoded.isna().sum().sum())
    print("最终 infinite values 数量:", np.isinf(numeric_df).sum().sum())

    os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
    df_encoded.to_csv(train_output_path, index=False)
    print("处理后的训练集已保存至:", train_output_path)


# ============================================================
# 3. 测试集：只 transform，不重新 fit
# ============================================================
def process_test():
    with open(config_path, "rb") as f:
        config = pickle.load(f)

    df = pd.read_csv(test_input_path)
    print("原始测试集形状:", df.shape)

    # =========================
    # 1. 去除重复行
    # =========================
    df = df.drop_duplicates().reset_index(drop=True)
    print("去重后测试集形状:", df.shape)

    # =========================
    # 2. 隔离 SK_ID_CURR 和可能存在的 TARGET
    # =========================
    X, id_target_df, keep_cols = split_id_target(df)
    print("隔离出的列:", keep_cols)
    print("进入预处理的测试特征形状:", X.shape)

    # =========================
    # 3. DAYS_EMPLOYED 异常值替换
    # =========================
    if "DAYS_EMPLOYED" in X.columns:
        X["DAYS_EMPLOYED"] = X["DAYS_EMPLOYED"].replace(365243, np.nan)

    # =========================
    # 4. 使用训练集确定的 indicator_cols 创建 missing indicator
    #    不在 test set 中重新判断缺失率
    # =========================
    for col in config["indicator_cols"]:
        if col in X.columns:
            X[col + "_MISSING_INDICATOR"] = X[col].isna().astype(int)
        else:
            X[col + "_MISSING_INDICATOR"] = 0

    # =========================
    # 5. 构造衍生特征
    # =========================
    X = add_engineered_features(X)

    # =========================
    # 6. 使用训练集保存的 quantile 边界做截断
    #    不在 test set 中重新计算 quantile
    # =========================
    for col, bounds in config["clip_bounds"].items():
        if col in X.columns:
            X[col] = X[col].clip(lower=bounds["lower"], upper=bounds["upper"])

    # =========================
    # 7. 使用训练集保存的 median / most frequency 填补缺失值
    #    不在 test set 中重新计算 median / mode
    # =========================
    for col, median_value in config["numerical_impute_dict"].items():
        if col in X.columns:
            X[col] = X[col].fillna(median_value)

    for col, most_freq_value in config["categorical_impute_dict"].items():
        if col in X.columns:
            X[col] = X[col].fillna(most_freq_value)

    # =========================
    # 8. 使用训练集保存的 categorical 编码规则
    #    test 不重新判断 2 类 / 3-7 类 / 7 类以上
    # =========================
    binary_encoding_dict = config["binary_encoding_dict"]
    one_hot_encoding_dict = config["one_hot_encoding_dict"]
    frequency_encoding_dict = config["frequency_encoding_dict"]
    rare_categories_dict = config["rare_categories_dict"]
    constant_encoding_dict = config["constant_encoding_dict"]

    # -------------------------
    # 8.1 binary encoding
    # -------------------------
    for col, mapping in binary_encoding_dict.items():
        if col in X.columns:
            X[col] = X[col].map(mapping)
            # test 中未见过的新类别，用 0 兜底
            X[col] = X[col].fillna(0).astype(int)
        else:
            X[col] = 0

    # -------------------------
    # 8.2 one-hot encoding
    # -------------------------
    for col, info in one_hot_encoding_dict.items():
        dummy_columns = info["dummy_columns"]

        if col in X.columns:
            dummies = pd.get_dummies(
                X[col],
                prefix=col,
                drop_first=False,
                dtype=int
            )
        else:
            dummies = pd.DataFrame(index=X.index)

        # train 中有、test 中没有的 dummy 列补 0
        for dummy_col in dummy_columns:
            if dummy_col not in dummies.columns:
                dummies[dummy_col] = 0

        # test 中新出现但 train 中没有的 dummy 列直接丢弃
        dummies = dummies[dummy_columns]

        if col in X.columns:
            X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
        else:
            X = pd.concat([X, dummies], axis=1)

    # -------------------------
    # 8.3 frequency encoding
    # -------------------------
    for col, freq_mapping in frequency_encoding_dict.items():
        if col in X.columns:
            rare_categories = rare_categories_dict.get(col, [])

            # train 中被判定为低频的类别，test 中也合并为 RARE
            X[col] = X[col].replace(rare_categories, "RARE")

            # test 中训练集没见过的新类别，map 后会变 NaN
            X[col] = X[col].map(freq_mapping)

            # 新类别用 RARE 的训练集频率兜底；如果训练集没有 RARE，则用 0
            X[col] = X[col].fillna(freq_mapping.get("RARE", 0)).astype(float)
        else:
            X[col] = freq_mapping.get("RARE", 0)

    # -------------------------
    # 8.4 constant encoding
    # -------------------------
    for col, value in constant_encoding_dict.items():
        X[col] = value

    # =========================
    # 9. 对齐训练集最终特征列
    #    train 有但 test 没有的列补 0；
    #    test 多出来的列删除；
    #    最终顺序与 train 完全一致。
    # =========================
    final_feature_cols = config["final_feature_cols"]

    for col in final_feature_cols:
        if col not in X.columns:
            X[col] = 0

    X = X[final_feature_cols]

    # =========================
    # 10. 最后确保没有 missing values 或 infinite values
    # =========================
    X = final_check_and_fill(X)

    # =========================
    # 11. 加回 SK_ID_CURR 和可能存在的 TARGET
    # =========================
    df_encoded = pd.concat(
        [id_target_df.reset_index(drop=True), X.reset_index(drop=True)],
        axis=1
    )

    # =========================
    # 12. 最终检查并保存
    # =========================
    numeric_df = df_encoded.select_dtypes(include=[np.number])
    print("最终测试集形状:", df_encoded.shape)
    print("最终 missing values 数量:", df_encoded.isna().sum().sum())
    print("最终 infinite values 数量:", np.isinf(numeric_df).sum().sum())

    os.makedirs(os.path.dirname(test_output_path), exist_ok=True)
    df_encoded.to_csv(test_output_path, index=False)
    print("处理后的测试集已保存至:", test_output_path)


# ============================================================
# 4. 主程序入口
# ============================================================
if __name__ == "__main__":
    if mode == "train":
        process_train()
    elif mode == "test":
        process_test()
    else:
        raise ValueError("mode 只能是 'train' 或 'test'")
