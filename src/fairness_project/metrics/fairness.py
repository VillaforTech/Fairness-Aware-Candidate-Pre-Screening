"""Fairness metrics for evaluating algorithmic fairness."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def demographic_parity(y: np.ndarray, sensitive: np.ndarray) -> dict[Any, float]:
    """
    Compute P(Y=1 | group) for each group.

    Parameters
    ----------
    y : np.ndarray
        Binary predictions or labels.
    sensitive : np.ndarray
        Sensitive attribute values.

    Returns
    -------
    dict[Any, float]
        Dictionary mapping group values to positive prediction rates.
    """
    df = pd.DataFrame({"y": y, "sensitive": sensitive})
    return df.groupby("sensitive")["y"].mean().to_dict()


def statistical_parity_difference(
    y: np.ndarray,
    sensitive: np.ndarray,
    privileged_group: Any,
) -> float:
    """
    Compute Statistical Parity Difference (SPD).

    SPD = P(Y=1 | privileged) - P(Y=1 | unprivileged)

    Parameters
    ----------
    y : np.ndarray
        Binary predictions.
    sensitive : np.ndarray
        Sensitive attribute values.
    privileged_group : Any
        Value identifying the privileged group.

    Returns
    -------
    float
        Statistical parity difference. Ideal value is 0.
    """
    dp = demographic_parity(y, sensitive)
    privileged_rate = dp[privileged_group]

    # Get unprivileged group(s) - average rate for multi-group case
    unprivileged_rates = [rate for g, rate in dp.items() if g != privileged_group]
    if not unprivileged_rates:
        return 0.0
    unprivileged_rate = np.mean(unprivileged_rates)

    return privileged_rate - unprivileged_rate


def disparate_impact(
    y: np.ndarray,
    sensitive: np.ndarray,
    privileged_group: Any,
) -> float:
    """
    Compute Disparate Impact ratio.

    DI = P(Y=1 | unprivileged) / P(Y=1 | privileged)

    Parameters
    ----------
    y : np.ndarray
        Binary predictions.
    sensitive : np.ndarray
        Sensitive attribute values.
    privileged_group : Any
        Value identifying the privileged group.

    Returns
    -------
    float
        Disparate impact ratio. Ideal value is 1.0.
        Returns NaN if privileged rate is 0.
    """
    dp = demographic_parity(y, sensitive)
    privileged_rate = dp[privileged_group]

    if privileged_rate == 0:
        return np.nan

    # Get unprivileged group(s) - average rate for multi-group case
    unprivileged_rates = [rate for g, rate in dp.items() if g != privileged_group]
    if not unprivileged_rates:
        return 1.0
    unprivileged_rate = np.mean(unprivileged_rates)

    return unprivileged_rate / privileged_rate


def true_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute True Positive Rate (TPR).

    TPR = TP / (TP + FN)

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_pred : np.ndarray
        Binary predictions.

    Returns
    -------
    float
        True positive rate. Returns 0.0 if no positive examples.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask_pos = y_true == 1
    if mask_pos.sum() == 0:
        return 0.0

    return (y_pred[mask_pos] == 1).mean()


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute False Positive Rate (FPR).

    FPR = FP / (FP + TN)

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_pred : np.ndarray
        Binary predictions.

    Returns
    -------
    float
        False positive rate. Returns 0.0 if no negative examples.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask_neg = y_true == 0
    if mask_neg.sum() == 0:
        return 0.0

    return (y_pred[mask_neg] == 1).mean()


def equalized_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    privileged_group: Any,
) -> dict[str, float]:
    """
    Compute Equalized Odds difference (TPR gap and FPR gap).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_pred : np.ndarray
        Binary predictions.
    sensitive : np.ndarray
        Sensitive attribute values.
    privileged_group : Any
        Value identifying the privileged group.

    Returns
    -------
    dict[str, float]
        Dictionary with TPR_gap and FPR_gap.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    priv_mask = sensitive == privileged_group
    unpriv_mask = ~priv_mask

    tpr_priv = true_positive_rate(y_true[priv_mask], y_pred[priv_mask])
    tpr_unpriv = true_positive_rate(y_true[unpriv_mask], y_pred[unpriv_mask])

    fpr_priv = false_positive_rate(y_true[priv_mask], y_pred[priv_mask])
    fpr_unpriv = false_positive_rate(y_true[unpriv_mask], y_pred[unpriv_mask])

    return {
        "TPR_gap": tpr_priv - tpr_unpriv,
        "FPR_gap": fpr_priv - fpr_unpriv,
        "TPR_priv": tpr_priv,
        "TPR_unpriv": tpr_unpriv,
        "FPR_priv": fpr_priv,
        "FPR_unpriv": fpr_unpriv,
    }


def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    privileged_group: Any,
) -> dict[str, Any]:
    """
    Compute all fairness metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_pred : np.ndarray
        Binary predictions.
    sensitive : np.ndarray
        Sensitive attribute values.
    privileged_group : Any
        Value identifying the privileged group.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - DP: Demographic parity rates per group
        - SPD: Statistical parity difference
        - DI: Disparate impact
        - EO: Equalized odds metrics (TPR gap, FPR gap, etc.)
    """
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    results = {
        "DP": demographic_parity(y_pred, sensitive),
        "SPD": statistical_parity_difference(y_pred, sensitive, privileged_group),
        "DI": disparate_impact(y_pred, sensitive, privileged_group),
    }

    # Add equalized odds if y_true is provided
    if y_true is not None:
        y_true = np.asarray(y_true)
        eo_metrics = equalized_odds_difference(y_true, y_pred, sensitive, privileged_group)
        results["EO"] = eo_metrics
        results["TPR_gap"] = eo_metrics["TPR_gap"]

    return results
