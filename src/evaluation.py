import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


def evaluate_binary_classifier(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }

    return metrics, y_pred


def save_metrics(metrics: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def plot_confusion_matrix(y_true, y_pred, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Non-default", "Default"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, values_format="d")
    ax.set_title("Neural Network Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)


def plot_training_history(history, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    hist = pd.DataFrame(history.history)

    if {"auc", "val_auc"}.issubset(hist.columns):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(hist["auc"], label="Train AUC")
        ax.plot(hist["val_auc"], label="Validation AUC")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AUC")
        ax.set_title("Training and Validation AUC")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "training_auc.png", dpi=200)
        plt.close(fig)

    if {"loss", "val_loss"}.issubset(hist.columns):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(hist["loss"], label="Train Loss")
        ax.plot(hist["val_loss"], label="Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Binary Crossentropy Loss")
        ax.set_title("Training and Validation Loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "training_loss.png", dpi=200)
        plt.close(fig)
