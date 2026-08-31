"""Fairness metrics for evaluating algorithmic fairness."""

from __future__ import annotations

from typing import Any, cast

import numpy as np


def _validate_equal_length(**arrays: np.ndarray) -> None:
    lengths = {name: len(np.asarray(values)) for name, values in arrays.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"Inputs must have equal lengths, got {lengths}")


def _validated_weights(sample_weight: np.ndarray | None, length: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(length, dtype=float)
    weights = np.asarray(sample_weight, dtype=float)
    if weights.ndim != 1 or len(weights) != length:
        raise ValueError("sample_weight must be one-dimensional and match the input length")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("sample_weight must contain finite non-negative values")
    if float(weights.sum()) <= 0:
        raise ValueError("sample_weight must have positive total weight")
    return weights


def _weighted_rate(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0 or float(weights.sum()) <= 0:
        return float("nan")
    return float(np.average(values.astype(float), weights=weights))


def demographic_parity(
    y: np.ndarray,
    sensitive: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[Any, float]:
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
    _validate_equal_length(y=y, sensitive=sensitive)
    outcomes = np.asarray(y)
    groups = np.asarray(sensitive)
    weights = _validated_weights(sample_weight, len(outcomes))
    rates = {
        value: _weighted_rate(outcomes[groups == value], weights[groups == value])
        for value in np.unique(groups)
    }
    return cast(dict[Any, float], rates)


def statistical_parity_difference(
    y: np.ndarray,
    sensitive: np.ndarray,
    privileged_group: Any,
    sample_weight: np.ndarray | None = None,
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
    dp = demographic_parity(y, sensitive, sample_weight)
    if privileged_group not in dp:
        raise ValueError(f"Privileged group {privileged_group!r} is absent")
    privileged_rate = dp[privileged_group]

    # Get unprivileged group(s) - average rate for multi-group case
    unprivileged_rates = [rate for g, rate in dp.items() if g != privileged_group]
    if not unprivileged_rates:
        raise ValueError("At least one unprivileged group is required")
    unprivileged_rate = np.mean(unprivileged_rates)

    return float(privileged_rate - unprivileged_rate)


def disparate_impact(
    y: np.ndarray,
    sensitive: np.ndarray,
    privileged_group: Any,
    sample_weight: np.ndarray | None = None,
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
    dp = demographic_parity(y, sensitive, sample_weight)
    if privileged_group not in dp:
        raise ValueError(f"Privileged group {privileged_group!r} is absent")
    privileged_rate = dp[privileged_group]

    if privileged_rate == 0:
        return np.nan

    # Get unprivileged group(s) - average rate for multi-group case
    unprivileged_rates = [rate for g, rate in dp.items() if g != privileged_group]
    if not unprivileged_rates:
        raise ValueError("At least one unprivileged group is required")
    unprivileged_rate = np.mean(unprivileged_rates)

    return float(unprivileged_rate / privileged_rate)


def true_positive_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
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
        True positive rate. Returns NaN if no positive examples.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    _validate_equal_length(y_true=y_true, y_pred=y_pred)
    weights = _validated_weights(sample_weight, len(y_true))
    mask_pos = y_true == 1
    if mask_pos.sum() == 0:
        return float("nan")

    return _weighted_rate(y_pred[mask_pos] == 1, weights[mask_pos])


def false_positive_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
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
        False positive rate. Returns NaN if no negative examples.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    _validate_equal_length(y_true=y_true, y_pred=y_pred)
    weights = _validated_weights(sample_weight, len(y_true))
    mask_neg = y_true == 0
    if mask_neg.sum() == 0:
        return float("nan")

    return _weighted_rate(y_pred[mask_neg] == 1, weights[mask_neg])


def equalized_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    privileged_group: Any,
    sample_weight: np.ndarray | None = None,
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
    _validate_equal_length(y_true=y_true, y_pred=y_pred, sensitive=sensitive)
    weights = _validated_weights(sample_weight, len(y_true))

    groups = set(np.unique(sensitive))
    if privileged_group not in groups:
        raise ValueError(f"Privileged group {privileged_group!r} is absent")
    if len(groups) != 2:
        raise ValueError(
            f"Binary fairness metrics require exactly two groups, got {sorted(groups)}"
        )

    priv_mask = sensitive == privileged_group
    unpriv_mask = ~priv_mask

    tpr_priv = true_positive_rate(y_true[priv_mask], y_pred[priv_mask], weights[priv_mask])
    tpr_unpriv = true_positive_rate(y_true[unpriv_mask], y_pred[unpriv_mask], weights[unpriv_mask])

    fpr_priv = false_positive_rate(y_true[priv_mask], y_pred[priv_mask], weights[priv_mask])
    fpr_unpriv = false_positive_rate(y_true[unpriv_mask], y_pred[unpriv_mask], weights[unpriv_mask])

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
    sample_weight: np.ndarray | None = None,
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
        "DP": demographic_parity(y_pred, sensitive, sample_weight),
        "SPD": statistical_parity_difference(y_pred, sensitive, privileged_group, sample_weight),
        "DI": disparate_impact(y_pred, sensitive, privileged_group, sample_weight),
    }

    # Add equalized odds if y_true is provided
    if y_true is not None:
        y_true = np.asarray(y_true)
        eo_metrics = equalized_odds_difference(
            y_true, y_pred, sensitive, privileged_group, sample_weight
        )
        results["EO"] = eo_metrics
        results["TPR_gap"] = eo_metrics["TPR_gap"]

    return results
