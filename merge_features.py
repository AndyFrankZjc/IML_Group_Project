import pandas as pd
from pathlib import Path

application_train_processed = pd.read_csv("processed_data/application_test_processed.csv")
installments = pd.read_csv("raw_data/installments_payments.csv")
bureau = pd.read_csv("raw_data/bureau.csv")

# =========================
# 1. installments_payments.csv 特征聚合
# =========================
# 逾期还款天数
installments["PAYMENT_DELAY"] = (
    installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]
)
# 是否足额还款
installments["PAYMENT_DIFF"] = (
    installments["AMT_INSTALMENT"] - installments["AMT_PAYMENT"]
)
# bool to int
installments["PAYMENT_DELAY_FLAG"] = (
    installments["PAYMENT_DELAY"] > 0
).astype("int8")
installments["PAYMENT_DIFF_FLAG"] = (
    installments["PAYMENT_DIFF"] > 0
).astype("int8")

installments_agg = (
    installments
    .groupby("SK_ID_CURR", as_index=False)
    .agg(
        PAYMENT_DELAY_COUNT=("PAYMENT_DELAY_FLAG", "sum"),
        PAYMENT_DIFF_COUNT=("PAYMENT_DIFF_FLAG", "sum")
    )
)

# =========================
# 2. bureau.csv 特征聚合
# =========================
bureau_agg = (
    bureau
    .groupby("SK_ID_CURR", as_index=False)
    .agg(
        CREDIT_DAY_OVERDUE_SUM=("CREDIT_DAY_OVERDUE", "sum")
    )
)

# =========================
# 3. 与 application_train_processed.csv 合并
# =========================
df = application_train_processed.copy()

df = df.merge(
    installments_agg,
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

df = df.merge(
    bureau_agg,
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

numerical_cols = ["PAYMENT_DELAY_COUNT", "PAYMENT_DIFF_COUNT", "CREDIT_DAY_OVERDUE_SUM"]
for col in numerical_cols:
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)
    

# 保存结果
df.to_csv("processed_data/application_test_processed_merge_features.csv", index=False)
df.head(1000).to_csv("view_data/application_test_processed_merge_features_1000.csv", index=False)