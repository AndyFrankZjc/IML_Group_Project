from typing import Optional

import numpy as np
import pandas as pd


def add_application_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
        df.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan

    eps = 1e-6

    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + eps)

    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + eps)

    if {"AMT_GOODS_PRICE", "AMT_CREDIT"}.issubset(df.columns):
        df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / (df["AMT_CREDIT"] + eps)

    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(df.columns):
        df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + eps)

    if {"DAYS_EMPLOYED", "DAYS_BIRTH"}.issubset(df.columns):
        df["EMPLOYED_BIRTH_RATIO"] = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + eps)

    return df


def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    bureau = bureau.copy()

    aggs = {
        "SK_ID_BUREAU": ["count"],
        "DAYS_CREDIT": ["mean", "min", "max"],
        "CREDIT_DAY_OVERDUE": ["mean", "max"],
        "DAYS_CREDIT_ENDDATE": ["mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["mean", "max"],
        "AMT_CREDIT_SUM": ["mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean", "sum"],
        "AMT_ANNUITY": ["mean", "sum"],
    }

    aggs = {col: funcs for col, funcs in aggs.items() if col in bureau.columns}

    bureau_num = bureau.groupby("SK_ID_CURR").agg(aggs)
    bureau_num.columns = [
        "BUREAU_" + "_".join(col).upper()
        for col in bureau_num.columns.to_flat_index()
    ]
    bureau_num = bureau_num.reset_index()

    if "CREDIT_ACTIVE" in bureau.columns:
        active_counts = pd.crosstab(
            bureau["SK_ID_CURR"],
            bureau["CREDIT_ACTIVE"],
            normalize="index"
        )
        active_counts.columns = [
            f"BUREAU_CREDIT_ACTIVE_RATE_{str(c).upper().replace(' ', '_')}"
            for c in active_counts.columns
        ]
        active_counts = active_counts.reset_index()
        bureau_num = bureau_num.merge(active_counts, on="SK_ID_CURR", how="left")

    return bureau_num


def aggregate_previous_application(prev: pd.DataFrame) -> pd.DataFrame:
    prev = prev.copy()

    aggs = {
        "SK_ID_PREV": ["count"],
        "AMT_ANNUITY": ["mean", "max"],
        "AMT_APPLICATION": ["mean", "sum"],
        "AMT_CREDIT": ["mean", "sum"],
        "AMT_DOWN_PAYMENT": ["mean"],
        "AMT_GOODS_PRICE": ["mean"],
        "HOUR_APPR_PROCESS_START": ["mean"],
        "DAYS_DECISION": ["mean", "min", "max"],
        "CNT_PAYMENT": ["mean", "max"],
    }

    aggs = {col: funcs for col, funcs in aggs.items() if col in prev.columns}

    prev_num = prev.groupby("SK_ID_CURR").agg(aggs)
    prev_num.columns = [
        "PREV_" + "_".join(col).upper()
        for col in prev_num.columns.to_flat_index()
    ]
    prev_num = prev_num.reset_index()

    if "NAME_CONTRACT_STATUS" in prev.columns:
        status_counts = pd.crosstab(
            prev["SK_ID_CURR"],
            prev["NAME_CONTRACT_STATUS"],
            normalize="index"
        )
        status_counts.columns = [
            f"PREV_STATUS_RATE_{str(c).upper().replace(' ', '_')}"
            for c in status_counts.columns
        ]
        status_counts = status_counts.reset_index()
        prev_num = prev_num.merge(status_counts, on="SK_ID_CURR", how="left")

    return prev_num


def aggregate_installments(installments: pd.DataFrame) -> pd.DataFrame:
    inst = installments.copy()
    eps = 1e-6

    if {"DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT"}.issubset(inst.columns):
        inst["PAYMENT_DELAY"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
        inst["LATE_PAYMENT"] = (inst["PAYMENT_DELAY"] > 0).astype(int)

    if {"AMT_PAYMENT", "AMT_INSTALMENT"}.issubset(inst.columns):
        inst["PAYMENT_RATIO"] = inst["AMT_PAYMENT"] / (inst["AMT_INSTALMENT"] + eps)

    aggs = {
        "SK_ID_PREV": ["count"],
        "NUM_INSTALMENT_VERSION": ["mean"],
        "NUM_INSTALMENT_NUMBER": ["max"],
        "DAYS_INSTALMENT": ["mean"],
        "DAYS_ENTRY_PAYMENT": ["mean"],
        "AMT_INSTALMENT": ["mean", "sum"],
        "AMT_PAYMENT": ["mean", "sum"],
        "PAYMENT_DELAY": ["mean", "max"],
        "LATE_PAYMENT": ["mean", "sum"],
        "PAYMENT_RATIO": ["mean", "min"],
    }

    aggs = {col: funcs for col, funcs in aggs.items() if col in inst.columns}

    inst_num = inst.groupby("SK_ID_CURR").agg(aggs)
    inst_num.columns = [
        "INST_" + "_".join(col).upper()
        for col in inst_num.columns.to_flat_index()
    ]
    inst_num = inst_num.reset_index()

    return inst_num


def merge_optional_tables(
    app: pd.DataFrame,
    bureau: Optional[pd.DataFrame] = None,
    previous: Optional[pd.DataFrame] = None,
    installments: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    df = app.copy()

    if bureau is not None:
        bureau_agg = aggregate_bureau(bureau)
        df = df.merge(bureau_agg, on="SK_ID_CURR", how="left")

    if previous is not None:
        prev_agg = aggregate_previous_application(previous)
        df = df.merge(prev_agg, on="SK_ID_CURR", how="left")

    if installments is not None:
        inst_agg = aggregate_installments(installments)
        df = df.merge(inst_agg, on="SK_ID_CURR", how="left")

    return df
