"""Performance metrics for model evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_performance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Compute standard performance metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_pred : np.ndarray
        Binary predictions.
    y_proba : np.ndarray, optional
        Predicted probabilities for positive class.

    Returns
    -------
    dict[str, Any]
        Dictionary containing accuracy, precision, recall, f1,
        and optionally ROC-AUC, PR-AUC, and Brier score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    if cm.shape == (2, 2):
        metrics["true_negatives"] = int(cm[0, 0])
        metrics["false_positives"] = int(cm[0, 1])
        metrics["false_negatives"] = int(cm[1, 0])
        metrics["true_positives"] = int(cm[1, 1])

    # Probability-based metrics
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = None

        try:
            metrics["pr_auc"] = average_precision_score(y_true, y_proba)
        except ValueError:
            metrics["pr_auc"] = None

        metrics["brier_score"] = brier_score_loss(y_true, y_proba)

    return metrics


def print_performance_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Print and return performance metrics for a model.

    Parameters
    ----------
    model_name : str
        Name of the model for display.
    y_true : np.ndarray
        Ground truth binary labels.
    y_pred : np.ndarray
        Binary predictions.
    y_proba : np.ndarray, optional
        Predicted probabilities.

    Returns
    -------
    dict[str, Any]
        Dictionary of computed metrics.
    """
    metrics = compute_performance_metrics(y_true, y_pred, y_proba)

    print(f"\n {model_name} PERFORMANCE ")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")

    if "roc_auc" in metrics and metrics["roc_auc"] is not None:
        print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    if "pr_auc" in metrics and metrics["pr_auc"] is not None:
        print(f"PR-AUC   : {metrics['pr_auc']:.4f}")
    if "brier_score" in metrics:
        print(f"Brier    : {metrics['brier_score']:.4f}")

    return metrics
