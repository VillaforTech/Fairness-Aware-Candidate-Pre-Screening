"""Evaluation functions combining performance and fairness metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


def compute_tpr_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    privileged_value: Any,
) -> dict[str, float]:
    """
    Compute TPR by sensitive group.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Binary predictions.
    sensitive : np.ndarray
        Sensitive attribute values.
    privileged_value : Any
        Value identifying the privileged group.

    Returns
    -------
    dict[str, float]
        Dictionary with TPR for each group and the gap.
    """
    from fairness_project.metrics.fairness import true_positive_rate

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    priv_mask = sensitive == privileged_value
    unpriv_mask = ~priv_mask

    tpr_priv = true_positive_rate(y_true[priv_mask], y_pred[priv_mask])
    tpr_unpriv = true_positive_rate(y_true[unpriv_mask], y_pred[unpriv_mask])

    return {
        "TPR_priv": tpr_priv,
        "TPR_unpriv": tpr_unpriv,
        "TPR_gap": tpr_priv - tpr_unpriv,
    }


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    privileged_group: Any,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Compute all performance and fairness metrics for predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Binary predictions.
    sensitive : np.ndarray
        Sensitive attribute values.
    privileged_group : Any
        Value identifying the privileged group.
    y_proba : np.ndarray, optional
        Predicted probabilities.

    Returns
    -------
    dict[str, Any]
        Dictionary with all metrics.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    from fairness_project.metrics.fairness import compute_fairness_metrics

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Performance metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # Probability-based metrics
    if y_proba is not None:
        from sklearn.metrics import (
            average_precision_score,
            brier_score_loss,
            roc_auc_score,
        )

        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = None
        try:
            metrics["pr_auc"] = average_precision_score(y_true, y_proba)
        except ValueError:
            metrics["pr_auc"] = None
        metrics["brier_score"] = brier_score_loss(y_true, y_proba)

    # Fairness metrics
    fair = compute_fairness_metrics(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive,
        privileged_group=privileged_group,
    )
    metrics["SPD"] = fair["SPD"]
    metrics["DI"] = fair["DI"]

    # TPR by group
    tpr_metrics = compute_tpr_by_group(y_true, y_pred, sensitive, privileged_group)
    metrics.update(tpr_metrics)

    return metrics


def print_evaluation_summary(
    metrics: dict[str, Any],
    label: str = "Model",
) -> None:
    """
    Print a formatted evaluation summary.

    Parameters
    ----------
    metrics : dict[str, Any]
        Dictionary of metrics.
    label : str
        Label for the model/condition.
    """
    print(f"\n {'=' * 20} {label} {'=' * 20}")

    print("\nPerformance:")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-score : {metrics['f1']:.4f}")

    if "roc_auc" in metrics and metrics["roc_auc"] is not None:
        print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")

    print("\nFairness:")
    print(f"  SPD      : {metrics['SPD']:.4f}")
    print(f"  DI       : {metrics['DI']:.4f}")
    print(f"  TPR Gap  : {metrics['TPR_gap']:.4f}")

    print("\nTPR by Group:")
    print(f"  Privileged   : {metrics['TPR_priv']:.4f}")
    print(f"  Unprivileged : {metrics['TPR_unpriv']:.4f}")


def compare_baseline_vs_eo(
    baseline_metrics: dict[str, Any],
    eo_metrics: dict[str, Any],
) -> pd.DataFrame:
    """
    Create a comparison table of baseline vs EO-adjusted metrics.

    Parameters
    ----------
    baseline_metrics : dict[str, Any]
        Metrics from baseline predictions.
    eo_metrics : dict[str, Any]
        Metrics from EO-adjusted predictions.

    Returns
    -------
    pd.DataFrame
        Comparison table.
    """
    metrics_to_compare = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "SPD",
        "DI",
        "TPR_gap",
    ]

    rows = []
    for metric in metrics_to_compare:
        if metric in baseline_metrics and metric in eo_metrics:
            baseline_val = baseline_metrics[metric]
            eo_val = eo_metrics[metric]
            change = eo_val - baseline_val if baseline_val is not None else None

            rows.append(
                {
                    "Metric": metric,
                    "Baseline": baseline_val,
                    "EO-Adjusted": eo_val,
                    "Change": change,
                }
            )

    return pd.DataFrame(rows)
