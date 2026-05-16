import argparse
import json
import re

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import VarianceThreshold

from config import (
    TRAIN_PROCESSED_PATH,
    TEST_PROCESSED_PATH,
    TRAIN_MERGED_PATH,
    TEST_MERGED_PATH,
    OUTPUT_DIR,
    RANDOM_STATE,
    TARGET_COL,
    ID_COL,
)
from model import build_mlp, get_callbacks
from evaluation import (
    evaluate_binary_classifier,
    save_metrics,
    plot_confusion_matrix,
    plot_training_history,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a neural network using processed Home Credit team features."
    )

    parser.add_argument(
        "--feature-set",
        choices=["application", "merged"],
        default="merged",
        help="Use application-only processed features or processed features merged with historical features."
    )
    parser.add_argument("--tune", action="store_true", help="Run quick hyperparameter tuning.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)

    return parser.parse_args()


def get_paths(feature_set):
    if feature_set == "application":
        return TRAIN_PROCESSED_PATH, TEST_PROCESSED_PATH
    if feature_set == "merged":
        return TRAIN_MERGED_PATH, TEST_MERGED_PATH
    raise ValueError(f"Unknown feature_set: {feature_set}")


def clean_column_names(df):
    """
    Make column names safer for downstream processing.
    This does not change feature meaning.
    """
    df = df.copy()
    cleaned = []
    for col in df.columns:
        new_col = str(col).strip()
        new_col = re.sub(r"\\s+", "_", new_col)
        cleaned.append(new_col)
    df.columns = cleaned
    return df


def load_processed_data(feature_set):
    train_path, test_path = get_paths(feature_set)

    if not train_path.exists():
        raise FileNotFoundError(f"Cannot find training file: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Cannot find test file: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df = clean_column_names(train_df)
    test_df = clean_column_names(test_df)

    if TARGET_COL not in train_df.columns:
        raise ValueError(f"{TARGET_COL} is missing from training file.")
    if ID_COL not in train_df.columns:
        raise ValueError(f"{ID_COL} is missing from training file.")
    if ID_COL not in test_df.columns:
        raise ValueError(f"{ID_COL} is missing from test file.")
    if TARGET_COL in test_df.columns:
        test_df = test_df.drop(columns=[TARGET_COL])

    return train_df, test_df


def check_duplicate_rows(train_df):
    n_before = len(train_df)
    train_df = train_df.drop_duplicates()
    n_after = len(train_df)
    return train_df, n_before - n_after


def align_train_test_features(train_df, kaggle_test_df):
    """
    Align columns between processed train and processed Kaggle test.

    The target exists only in train.
    SK_ID_CURR is kept separately and not used as a feature.
    """
    y = train_df[TARGET_COL].astype(int)
    train_ids = train_df[ID_COL].copy()
    kaggle_ids = kaggle_test_df[ID_COL].copy()

    X = train_df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
    X_kaggle = kaggle_test_df.drop(columns=[ID_COL], errors="ignore")

    # Add any missing columns to Kaggle test.
    for col in X.columns:
        if col not in X_kaggle.columns:
            X_kaggle[col] = np.nan

    # Drop extra columns in Kaggle test and enforce same order.
    X_kaggle = X_kaggle[X.columns]

    return X, y, train_ids, X_kaggle, kaggle_ids


def split_data(X, y, ids, args):
    X_temp, X_test, y_temp, y_test, ids_temp, ids_test = train_test_split(
        X,
        y,
        ids,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    val_relative = args.val_size / (1.0 - args.test_size)

    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_temp,
        y_temp,
        ids_temp,
        test_size=val_relative,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, ids_test


def preprocess_numeric_features(X_train, X_val, X_test, X_kaggle):
    """
    The team has already transformed categorical variables into numeric features.
    We therefore use a numeric-only NN preprocessing pipeline:
    1. median imputation as a safety check
    2. standardisation for stable neural network training
    3. variance threshold to remove constant columns
    """
    # Coerce all features to numeric. Any unexpected non-numeric value becomes NaN and is imputed.
    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_val = X_val.apply(pd.to_numeric, errors="coerce")
    X_test = X_test.apply(pd.to_numeric, errors="coerce")
    X_kaggle = X_kaggle.apply(pd.to_numeric, errors="coerce")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    selector = VarianceThreshold(threshold=0.0)

    X_train_i = imputer.fit_transform(X_train)
    X_val_i = imputer.transform(X_val)
    X_test_i = imputer.transform(X_test)
    X_kaggle_i = imputer.transform(X_kaggle)

    X_train_s = scaler.fit_transform(X_train_i)
    X_val_s = scaler.transform(X_val_i)
    X_test_s = scaler.transform(X_test_i)
    X_kaggle_s = scaler.transform(X_kaggle_i)

    X_train_p = selector.fit_transform(X_train_s)
    X_val_p = selector.transform(X_val_s)
    X_test_p = selector.transform(X_test_s)
    X_kaggle_p = selector.transform(X_kaggle_s)

    kept_mask = selector.get_support()
    kept_feature_names = X_train.columns[kept_mask].tolist()

    preprocessing_metadata = {
        "input_feature_count_before_variance_filter": int(X_train.shape[1]),
        "input_feature_count_after_variance_filter": int(X_train_p.shape[1]),
        "removed_constant_feature_count": int(X_train.shape[1] - X_train_p.shape[1]),
        "kept_feature_names_after_variance_filter": kept_feature_names,
        "preprocessing_note": (
            "Team-processed features were used as model input. "
            "This script only applied median imputation as a safety step, "
            "standardisation for neural network training, and constant-feature removal."
        )
    }

    return X_train_p, X_val_p, X_test_p, X_kaggle_p, preprocessing_metadata


def compute_weights(y_train):
    classes = np.array([0, 1])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    return {0: float(weights[0]), 1: float(weights[1])}


def train_one_model(
    X_train_p,
    X_val_p,
    y_train,
    y_val,
    input_dim,
    params,
    args,
    class_weight,
):
    tf.keras.backend.clear_session()

    model = build_mlp(
        input_dim=input_dim,
        hidden_layers=params["hidden_layers"],
        dropout_rate=params["dropout_rate"],
        learning_rate=params["learning_rate"],
    )

    history = model.fit(
        X_train_p,
        y_train,
        validation_data=(X_val_p, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=get_callbacks(),
        verbose=1,
    )

    val_prob = model.predict(X_val_p, verbose=0).ravel()
    val_auc = evaluate_binary_classifier(y_val, val_prob, threshold=args.threshold)[0]["roc_auc"]

    return model, history, val_auc


def get_search_space(tune):
    if tune:
        return [
            {"hidden_layers": (128, 64), "dropout_rate": 0.20, "learning_rate": 0.001},
            {"hidden_layers": (256, 128, 64), "dropout_rate": 0.30, "learning_rate": 0.001},
            {"hidden_layers": (512, 256, 128), "dropout_rate": 0.40, "learning_rate": 0.0005},
            {"hidden_layers": (256, 128), "dropout_rate": 0.50, "learning_rate": 0.001},
        ]

    return [
        {"hidden_layers": (256, 128, 64), "dropout_rate": 0.30, "learning_rate": 0.001}
    ]


def run(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    suffix = args.feature_set

    print(f"Loading team-processed feature set: {args.feature_set}")
    train_df, kaggle_test_df = load_processed_data(args.feature_set)

    train_df, duplicate_rows_removed = check_duplicate_rows(train_df)

    X, y, train_ids, X_kaggle, kaggle_ids = align_train_test_features(train_df, kaggle_test_df)

    print(f"Training rows after duplicate removal: {len(X)}")
    print(f"Duplicate rows removed: {duplicate_rows_removed}")
    print(f"Raw processed feature count: {X.shape[1]}")
    print(f"Kaggle test rows: {len(X_kaggle)}")

    X_train, X_val, X_internal_test, y_train, y_val, y_internal_test, ids_internal_test = split_data(
        X, y, train_ids, args
    )

    X_train_p, X_val_p, X_test_p, X_kaggle_p, preprocessing_metadata = preprocess_numeric_features(
        X_train, X_val, X_internal_test, X_kaggle
    )

    class_weight = compute_weights(y_train)

    print(f"Processed NN train shape: {X_train_p.shape}")
    print(f"Processed NN validation shape: {X_val_p.shape}")
    print(f"Processed NN internal test shape: {X_test_p.shape}")
    print(f"Processed NN Kaggle test shape: {X_kaggle_p.shape}")
    print(f"Class weights: {class_weight}")

    input_dim = X_train_p.shape[1]
    search_space = get_search_space(args.tune)

    tuning_rows = []
    best = {
        "val_auc": -1,
        "model": None,
        "history": None,
        "params": None,
    }

    for i, params in enumerate(search_space, start=1):
        print(f"\\nTraining model {i}/{len(search_space)} with params: {params}")
        model, history, val_auc = train_one_model(
            X_train_p=X_train_p,
            X_val_p=X_val_p,
            y_train=y_train,
            y_val=y_val,
            input_dim=input_dim,
            params=params,
            args=args,
            class_weight=class_weight,
        )

        row = {
            "model_id": i,
            "hidden_layers": str(params["hidden_layers"]),
            "dropout_rate": params["dropout_rate"],
            "learning_rate": params["learning_rate"],
            "batch_size": args.batch_size,
            "best_val_auc": val_auc,
        }
        tuning_rows.append(row)

        if val_auc > best["val_auc"]:
            best = {
                "val_auc": val_auc,
                "model": model,
                "history": history,
                "params": params,
            }

    pd.DataFrame(tuning_rows).to_csv(
        OUTPUT_DIR / f"tuning_results_processed_{suffix}.csv",
        index=False
    )

    best_model = best["model"]
    best_history = best["history"]

    print("\\nEvaluating best model on internal test set...")
    y_test_prob = best_model.predict(X_test_p, verbose=0).ravel()
    metrics, y_test_pred = evaluate_binary_classifier(
        y_internal_test,
        y_test_prob,
        threshold=args.threshold
    )

    metrics["best_validation_auc"] = float(best["val_auc"])
    metrics["best_hyperparameters"] = {
        "hidden_layers": str(best["params"]["hidden_layers"]),
        "dropout_rate": best["params"]["dropout_rate"],
        "learning_rate": best["params"]["learning_rate"],
        "batch_size": args.batch_size,
    }
    metrics["feature_set"] = args.feature_set
    metrics["duplicate_rows_removed"] = int(duplicate_rows_removed)
    metrics["class_weight"] = class_weight
    metrics["preprocessing_metadata"] = preprocessing_metadata
    metrics["team_preprocessing_assumptions"] = {
        "feature_screening": True,
        "duplicate_removal_before_modeling": True,
        "ambiguous_category_cleaning": True,
        "missing_indicator_for_missing_rate_ge_15_percent": True,
        "ext_source_aggregation": True,
        "six_ratio_features": True,
        "winsorization": True,
        "missing_value_imputation": True,
        "one_hot_and_frequency_encoding": True,
        "rare_category_grouping_under_2_percent": True,
        "merged_external_features_if_feature_set_is_merged": args.feature_set == "merged"
    }

    print(json.dumps(metrics, indent=2))

    save_metrics(metrics, OUTPUT_DIR / f"metrics_processed_{suffix}.json")

    plot_confusion_matrix(
        y_internal_test,
        y_test_pred,
        OUTPUT_DIR / f"confusion_matrix_processed_{suffix}.png"
    )

    plot_training_history(
        best_history,
        auc_path=OUTPUT_DIR / f"training_auc_processed_{suffix}.png",
        loss_path=OUTPUT_DIR / f"training_loss_processed_{suffix}.png"
    )

    pd.DataFrame({
        "SK_ID_CURR": ids_internal_test.values,
        "y_true": y_internal_test.values,
        "default_probability": y_test_prob,
        "y_pred_threshold": y_test_pred,
    }).to_csv(
        OUTPUT_DIR / f"nn_predictions_internal_test_processed_{suffix}.csv",
        index=False
    )

    kaggle_prob = best_model.predict(X_kaggle_p, verbose=0).ravel()

    pd.DataFrame({
        "SK_ID_CURR": kaggle_ids.values,
        "TARGET": kaggle_prob,
    }).to_csv(
        OUTPUT_DIR / f"kaggle_submission_nn_processed_{suffix}.csv",
        index=False
    )

    best_model.save(OUTPUT_DIR / f"best_mlp_model_processed_{suffix}.keras")

    print(f"\\nDone. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run(parse_args())
