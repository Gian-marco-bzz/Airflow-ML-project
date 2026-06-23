"""
Model evaluation and metrics.

Computes classification metrics and diagnostic plots for binary classification
models trained on imbalanced network utilization data.

Metrics prioritize:
- Recall (minimize false negatives = missed network caídas)
- Precision (minimize false alerts to Ops)
- PR-AUC (robust to imbalance)
"""

import logging
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)


def proba_to_class(y_prob: np.ndarray, threshold: float) -> np.ndarray:
    """
    Convert predicted probabilities to binary class predictions.

    Args:
        y_prob: Array of predicted probabilities [0, 1]
        threshold: Decision boundary (e.g., 0.70)

    Returns:
        Binary predictions [0, 1]

    Raises:
        ValueError: If threshold not in [0, 1]
    """
    if not (0 <= threshold <= 1):
        raise ValueError(f"threshold={threshold} must be in [0, 1]")
    return (y_prob >= threshold).astype(int)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.70
) -> Dict[str, float]:
    """
    Compute classification metrics for imbalanced data.

    Args:
        y_true: True labels (0/1)
        y_prob: Predicted probabilities [0, 1]
        threshold: Classification threshold (default: 0.70)

    Returns:
        Dict with Recall, Precision, F1, PR_AUC
    """
    y_pred = proba_to_class(y_prob, threshold)

    metrics = {
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "PR_AUC": float(average_precision_score(y_true, y_prob)),
    }

    return metrics


def pr_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract Precision-Recall curve data.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities

    Returns:
        Tuple (precision, recall, thresholds)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return precision, recall, thresholds


def roc_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extract ROC curve data with AUC.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities

    Returns:
        Tuple (fpr, tpr, roc_auc)
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = float(auc(fpr, tpr))
    return fpr, tpr, roc_auc


def classification_report_text(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.70
) -> str:
    """
    Generate sklearn classification report as text.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Formatted classification report string
    """
    y_pred = proba_to_class(y_prob, threshold)

    return classification_report(y_true, y_pred, zero_division=0)


def plot_confusion_matrix(
    y_true,
    y_prob,
    threshold: float,
    out_path: str,
    labels=("No riesgo", "Riesgo"),
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    y_pred = proba_to_class(y_prob, threshold)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Matriz de confusión")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pr_curve(y_true, y_prob, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    ax.set_title("Curva Precision-Recall")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_roc_curve(y_true, y_prob, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("Curva ROC")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path