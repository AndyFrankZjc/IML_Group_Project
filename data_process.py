import numpy as np
import pandas as pd
import pickle

input_path = "processed_data/application_train_selected_features.csv"
output_path = "processed_data/application_train_processed.csv"

df = pd.read_csv(input_path)
print("原始数据形状:", df.shape)

# =========================
# 1. 去除重复行
# =========================
df = df.drop_duplicates().reset_index(drop=True)
print("去重后数据形状:", df.shape)

# =========================
# 2. 将 DAYS_EMPLOYED 中的 365243 替换为 NaN
#    将 CODE_GENDER 中的 XNA 替换为 NaN
#    将 NAME_FAMILY_STATUS 中的 Unknown 替换为 NaN
# =========================
df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
df["CODE_GENDER"] = df["CODE_GENDER"].replace("XNA", np.nan)
df["NAME_FAMILY_STATUS"] = df["NAME_FAMILY_STATUS"].replace("Unknown", np.nan)

# =========================
# 3. 设置 missing indicator
#    如果某个特征缺失率 >= 15%，
#    则新增一列：原特征名 + "_MISSING_INDICATOR"
# =========================
id_col = ["SK_ID_CURR"]
target_col = ["TARGET"]
exclude_cols = id_col  + target_col
# get features
feature_cols = [col for col in df.columns if col not in exclude_cols]

missing_rate = df[feature_cols].isna().mean()

indicator_cols = missing_rate[missing_rate >= 0.15].index.tolist()

for col in indicator_cols:
    df[col + "_MISSING_INDICATOR"] = df[col].isna().astype(int)

print("创建 missing indicator 的特征:")
print(indicator_cols)

# =========================
# 4. 构造 EXT_SOURCE 相关特征
#    对 EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3 求：
#    EXT_SOURCE_MEAN，EXT_SOURCE_STD，EXT_SOURCE_N_MISSING，EXT_SOURCE_PRODUCT
# =========================
ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1, skipna=True)
# ddof=0 可以避免只有一个非缺失值时 std 直接变 NaN
df["EXT_SOURCE_STD"] = df[ext_cols].std(axis=1, skipna=True, ddof=0)
df["EXT_SOURCE_N_MISSING"] = df[ext_cols].isna().sum(axis=1)
df["EXT_SOURCE_PRODUCT"] = df[ext_cols].prod(axis=1, skipna=True)

# =========================
# 5. 构造比率特征
#    注意分母为 0 时替换为 NaN，避免 infinite values
# =========================
def safe_divide(numerator, denominator):
    # 如果分母为 0，则先替换为 NaN，
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator

# 本次申请的贷款金额 / 总收入
if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
    df["CREDIT_INCOME_RATIO"] = safe_divide(
        df["AMT_CREDIT"],
        df["AMT_INCOME_TOTAL"]
    )
# 本次申请的贷款金额 / 本次贷款的年金
if {"AMT_CREDIT", "AMT_ANNUITY"}.issubset(df.columns):
    df["CREDIT_ANNUITY_RATIO"] = safe_divide(
        df["AMT_CREDIT"],
        df["AMT_ANNUITY"]
    )
# 贷款所购商品价格 / 本次申请的贷款金额
if {"AMT_GOODS_PRICE", "AMT_CREDIT"}.issubset(df.columns):
    df["GOODS_CREDIT_RATIO"] = safe_divide(
        df["AMT_GOODS_PRICE"],
        df["AMT_CREDIT"]
    )
# 总收入 / 家庭人数
if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(df.columns):
    df["INCOME_PER_FAMILY_MEMBER"] = safe_divide(
        df["AMT_INCOME_TOTAL"],
        df["CNT_FAM_MEMBERS"]
    )
# 孩子数量 / 家庭人数
if {"CNT_CHILDREN", "CNT_FAM_MEMBERS"}.issubset(df.columns):
    df["CHILDREN_FAMILY_RATIO"] = safe_divide(
        df["CNT_CHILDREN"],
        df["CNT_FAM_MEMBERS"]
    )
# 工作天数 / 出生天数
if {"DAYS_EMPLOYED", "DAYS_BIRTH"}.issubset(df.columns):
    df["EMPLOYED_TO_AGE_RATIO"] = safe_divide(
        df["DAYS_EMPLOYED"],
        df["DAYS_BIRTH"]
    )

# =========================
# 6. 对 4 个 AMT 开头的特征做 1% - 99% 分位数截断
#    outlier handle，即 winsorization / clipping
# =========================
amt_clip_cols = [
    "AMT_CREDIT",
    "AMT_INCOME_TOTAL",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE"
]

for col in amt_clip_cols:
    if col in df.columns:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=lower, upper=upper)

# =========================
# 7. 缺失值填补
#     numerical 类型：median
#     categorical 类型：most frequency
# =========================
categorical_cols = df[feature_cols].select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numerical_cols = [
    col for col in feature_cols
    if col not in categorical_cols
]

for col in numerical_cols:
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)

for col in categorical_cols:
    most_freq_value = df[col].mode(dropna=True)[0]
    df[col] = df[col].fillna(most_freq_value)

# =========================
# 8. Categorical 转数值
#     规则：
#     1）类别只有 2 类：
#        替换为 0, 1
#
#     2）类别有 3 到 5 类：
#        one-hot encoding
#
#     3）类别超过 5 类：
#        先把出现概率 < 0.02 的类别合并为 "RARE"
#        然后做 frequency encoding
# =========================
df_encoded = df.copy()

# 重新获取类别列，因为前面已经填补过缺失值
categorical_cols = df_encoded[feature_cols].select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

one_hot_cols = []
# 用来记录每个 frequency encoding 特征中，哪些原始类别被合并成 RARE
# 后续处理 test set 时可以复用
frequency_encoding_dict = {}
rare_categories_dict = {}

for col in categorical_cols:
    n_unique = df_encoded[col].nunique(dropna=False)

    # 2 类：映射为 0 和 1
    if n_unique == 2:
        unique_values = sorted(df_encoded[col].unique())
        mapping = {
            unique_values[0]: 0,
            unique_values[1]: 1
        }
        df_encoded[col] = df_encoded[col].map(mapping).astype(int)

    # 3 到 5 类：one-hot encoding
    elif 3 <= n_unique <= 5:
        one_hot_cols.append(col)

    # 5 类以上：RARE 合并 + frequency encoding
    elif n_unique > 5:
        # 处理训练集
        freq = df_encoded[col].value_counts(normalize=True)

        rare_categories = freq[freq < 0.02].index.tolist()
        rare_categories_dict[col] = rare_categories

        df_encoded[col] = df_encoded[col].replace(
            rare_categories,
            "RARE"
        )

        freq_after_rare = df_encoded[col].value_counts(normalize=True)
        freq_mapping = freq_after_rare.to_dict()

        frequency_encoding_dict[col] = freq_mapping

        df_encoded[col] = df_encoded[col].map(freq_mapping).astype(float)
        

        # 处理测试集
        """
        with open("frequency_encoding_dict.pkl", "rb") as f:
            frequency_encoding_dict = pickle.load(f)

        with open("rare_categories_dict.pkl", "rb") as f:
            rare_categories_dict = pickle.load(f)

        freq_mapping = frequency_encoding_dict[col]
        rare_categories = rare_categories_dict[col]

        # test set 中，训练集里被判定为低频的类别也合并为 RARE
        df_encoded[col] = df_encoded[col].replace(rare_categories, "RARE")
        # 用训练集记录的频率替换类别
        df_encoded[col] = df_encoded[col].map(freq_mapping)
        """

# 处理训练集      
with open("frequency_encoding_dict.pkl", "wb") as f:
    pickle.dump(frequency_encoding_dict, f)

with open("rare_categories_dict.pkl", "wb") as f:
    pickle.dump(rare_categories_dict, f)

    
# 对 3 到 5 类的类别特征做 one-hot encoding
if len(one_hot_cols) > 0:
    df_encoded = pd.get_dummies(
        df_encoded,
        columns=one_hot_cols,
        drop_first=False,
        dtype=int
    )

print("最终数据形状:", df_encoded.shape)
# 保存处理后的 CSV
df_encoded.to_csv(output_path, index=False)
print("处理后的数据已保存至:", output_path)
