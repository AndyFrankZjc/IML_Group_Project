import argparse
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from config import (
    APPLICATION_TRAIN_PATH,
    BUREAU_PATH,
    PREVIOUS_APPLICATION_PATH,
    INSTALLMENTS_PAYMENTS_PATH,
    OUTPUT_DIR,
    RANDOM_STATE,
    TARGET_COL,
    ID_COL,
)
from features import (
    add_application_features,
    merge_optional_tables,
)
from preprocessing import preprocess_for_nn
from model import build_mlp, get_callbacks
from evaluation import (
    evaluate_binary_classifier,
    save_metrics,
    plot_confusion_matrix,
    plot_training_history,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a neural network for Home Credit Default Risk."
    )
    parser.add_argument("--use-bureau", action="store_true", help="Use aggregated bureau.csv features.")
    parser.add_argument("--use-previous", action="store_true", help="Use aggregated previous_application.csv features.")
    parser.add_argument("--use-installments", action="store_true", help="Use aggregated installments_payments.csv features.")
    parser.add_argument("--tune", action="store_true", help="Run a small hyperparameter tuning experiment.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--missing-threshold", type=float, default=0.70)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    return parser.parse_args()


def load_data(args):
    if not APPLICATION_TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {APPLICATION_TRAIN_PATH}. "
            "Please place application_train.csv under data/raw/."
        )

    app = pd.read_csv(APPLICATION_TRAIN_PATH)
    app = add_application_features(app)

    bureau = None
    previous = None
    installments = None

    if args.use_bureau:
        if not BUREAU_PATH.exists():
            raise FileNotFoundError(f"--use-bureau was set, but {BUREAU_PATH} does not exist.")
        bureau = pd.read_csv(BUREAU_PATH)

    if args.use_previous:
        if not PREVIOUS_APPLICATION_PATH.exists():
            raise FileNotFoundError(f"--use-previous was set, but {PREVIOUS_APPLICATION_PATH} does not exist.")
        previous = pd.read_csv(PREVIOUS_APPLICATION_PATH)

    if args.use_installments:
        if not INSTALLMENTS_PAYMENTS_PATH.exists():
            raise FileNotFoundError(f"--use-installments was set, but {INSTALLMENTS_PAYMENTS_PATH} does not exist.")
        installments = pd.read_csv(INSTALLMENTS_PAYMENTS_PATH)

    df = merge_optional_tables(
        app=app,
        bureau=bureau,
        previous=previous,
        installments=installments,
    )

    return df


def split_data(df, args):
    y = df[TARGET_COL].astype(int)
    ids = df[ID_COL]

    X = df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")

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


def compute_weights(y_train):
    classes = np.array([0, 1])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    return {0: float(weights[0]), 1: float(weights[1])}


def train_one_model(
    X_train_processed,
    X_val_processed,
    y_train,
    y_val,
    input_dim,
    hidden_layers,
    dropout_rate,
    learning_rate,
    batch_size,
    epochs,
    class_weight,
):
    tf.keras.backend.clear_session()

    model = build_mlp(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
    )

    history = model.fit(
        X_train_processed,
        y_train,
        validation_data=(X_val_processed, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=get_callbacks(),
        verbose=1,
    )

    val_prob = model.predict(X_val_processed, verbose=0).ravel()
    val_auc = evaluate_binary_classifier(y_val, val_prob)[0]["roc_auc"]

    return model, history, val_auc


def run_training(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data(args)

    print(f"Final applicant-level table shape: {df.shape}")
    print("Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test, ids_test = split_data(df, args)

    print("Preprocessing data for neural network...")
    X_train_p, X_val_p, X_test_p, preprocessor, selector, prep_metadata = preprocess_for_nn(
        X_train,
        X_val,
        X_test,
        missing_threshold=args.missing_threshold,
    )

    print(f"Processed train shape: {X_train_p.shape}")
    print(f"Processed validation shape: {X_val_p.shape}")
    print(f"Processed test shape: {X_test_p.shape}")

    class_weight = compute_weights(y_train)
    print(f"Class weights: {class_weight}")

    input_dim = X_train_p.shape[1]

    if args.tune:
        search_space = [
            {"hidden_layers": (128, 64), "dropout_rate": 0.20, "learning_rate": 0.001},
            {"hidden_layers": (256, 128, 64), "dropout_rate": 0.30, "learning_rate": 0.001},
            {"hidden_layers": (512, 256, 128), "dropout_rate": 0.40, "learning_rate": 0.0005},
            {"hidden_layers": (256, 128), "dropout_rate": 0.50, "learning_rate": 0.001},
        ]
    else:
        search_space = [
            {"hidden_layers": (256, 128, 64), "dropout_rate": 0.30, "learning_rate": 0.001}
        ]

    tuning_rows = []
    best = {
        "val_auc": -1,
        "model": None,
        "history": None,
        "params": None,
    }

    for i, params in enumerate(search_space, start=1):
        print(f"\nTraining model {i}/{len(search_space)} with params: {params}")

        model, history, val_auc = train_one_model(
            X_train_p,
            X_val_p,
            y_train,
            y_val,
            input_dim=input_dim,
            hidden_layers=params["hidden_layers"],
            dropout_rate=params["dropout_rate"],
            learning_rate=params["learning_rate"],
            batch_size=args.batch_size,
            epochs=args.epochs,
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

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df.to_csv(OUTPUT_DIR / "tuning_results.csv", index=False)

    best_model = best["model"]
    best_history = best["history"]

    print("\nEvaluating best model on test set...")
    y_test_prob = best_model.predict(X_test_p, verbose=0).ravel()
    metrics, y_test_pred = evaluate_binary_classifier(y_test, y_test_prob, threshold=0.5)

    metrics["best_validation_auc"] = float(best["val_auc"])
    metrics["best_hyperparameters"] = {
        "hidden_layers": str(best["params"]["hidden_layers"]),
        "dropout_rate": best["params"]["dropout_rate"],
        "learning_rate": best["params"]["learning_rate"],
        "batch_size": args.batch_size,
    }
    metrics["preprocessing_metadata"] = prep_metadata
    metrics["used_optional_tables"] = {
        "bureau": args.use_bureau,
        "previous_application": args.use_previous,
        "installments_payments": args.use_installments,
    }

    print(json.dumps(metrics, indent=2))

    save_metrics(metrics, OUTPUT_DIR)
    plot_confusion_matrix(y_test, y_test_pred, OUTPUT_DIR)
    plot_training_history(best_history, OUTPUT_DIR)

    pred_df = pd.DataFrame({
        "SK_ID_CURR": ids_test.values,
        "y_true": y_test.values,
        "default_probability": y_test_prob,
        "y_pred_threshold_0_5": y_test_pred,
    })
    pred_df.to_csv(OUTPUT_DIR / "nn_predictions_test.csv", index=False)

    best_model.save(OUTPUT_DIR / "best_mlp_model.keras")

    print(f"\nDone. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
