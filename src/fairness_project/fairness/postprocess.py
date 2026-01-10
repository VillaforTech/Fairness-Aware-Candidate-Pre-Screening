"""Equal Opportunity post-processing for bias mitigation."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_tpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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

    return float((y_pred[mask_pos] == 1).mean())


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_tpr: float,
    n_thresholds: int = 101,
    search_range: tuple[float, float] = (0.0, 0.5),
) -> tuple[float, float]:
    """
    Find optimal threshold to achieve target TPR.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_proba : np.ndarray
        Predicted probabilities.
    target_tpr : float
        Target TPR to match.
    n_thresholds : int
        Number of thresholds to search.
    search_range : tuple[float, float]
        Range of thresholds to search.

    Returns
    -------
    tuple[float, float]
        Best threshold and achieved TPR.
    """
    thresholds = np.linspace(search_range[0], search_range[1], n_thresholds)
    best_threshold = search_range[1]
    best_diff = float("inf")
    best_tpr = 0.0

    for th in thresholds:
        preds = (y_proba >= th).astype(int)
        tpr = compute_tpr(y_true, preds)
        diff = abs(tpr - target_tpr)

        if diff < best_diff:
            best_diff = diff
            best_threshold = th
            best_tpr = tpr

    return best_threshold, best_tpr


def equal_opportunity_postprocessing(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    sensitive_attr: np.ndarray,
    privileged_value: Any = "Male",
    unprivileged_value: Any = "Female",
    base_threshold: float = 0.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Equal Opportunity post-processing for a binary classifier.

    This technique equalizes TPR between privileged and unprivileged groups
    by adjusting the decision threshold for the unprivileged group.

    WARNING: This function requires ground truth labels for threshold optimization.
    To avoid test set leakage, ensure this is called ONLY on validation data
    for threshold tuning, then apply_thresholds() on test data.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels (for threshold optimization).
    y_pred_proba : np.ndarray
        Predicted probabilities for positive class.
    sensitive_attr : np.ndarray
        Sensitive attribute values.
    privileged_value : Any
        Value identifying the privileged group.
    unprivileged_value : Any
        Value identifying the unprivileged group.
    base_threshold : float
        Base decision threshold for privileged group.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        - Adjusted binary predictions
        - Info dictionary with TPRs and thresholds
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    sensitive_attr = np.asarray(sensitive_attr)

    priv_mask = sensitive_attr == privileged_value
    unpriv_mask = sensitive_attr == unprivileged_value

    # Sanity checks
    if priv_mask.sum() == 0:
        raise ValueError(f"No samples with privileged value '{privileged_value}' found.")
    if unpriv_mask.sum() == 0:
        raise ValueError(f"No samples with unprivileged value '{unprivileged_value}' found.")

    # Baseline predictions with common threshold
    y_pred_base = (y_pred_proba >= base_threshold).astype(int)

    # Baseline TPRs
    tpr_priv = compute_tpr(y_true[priv_mask], y_pred_base[priv_mask])
    tpr_unpriv_before = compute_tpr(y_true[unpriv_mask], y_pred_base[unpriv_mask])

    # If unprivileged TPR is already >= privileged TPR, no adjustment needed
    if tpr_unpriv_before >= tpr_priv:
        info = {
            "tpr_priv": tpr_priv,
            "tpr_unpriv_before": tpr_unpriv_before,
            "tpr_unpriv_after": tpr_unpriv_before,
            "threshold_priv": base_threshold,
            "threshold_unpriv": base_threshold,
            "note": "No adjustment: unprivileged TPR already >= privileged TPR.",
        }
        return y_pred_base, info

    # Find optimal threshold for unprivileged group
    best_threshold, tpr_unpriv_after = find_optimal_threshold(
        y_true=y_true[unpriv_mask],
        y_proba=y_pred_proba[unpriv_mask],
        target_tpr=tpr_priv,
    )

    # Construct adjusted predictions
    y_pred_adj = y_pred_base.copy()
    y_pred_adj[unpriv_mask] = (y_pred_proba[unpriv_mask] >= best_threshold).astype(int)

    info = {
        "tpr_priv": tpr_priv,
        "tpr_unpriv_before": tpr_unpriv_before,
        "tpr_unpriv_after": tpr_unpriv_after,
        "threshold_priv": base_threshold,
        "threshold_unpriv": best_threshold,
        "note": "EO post-processing applied (unprivileged threshold lowered).",
    }
    return y_pred_adj, info


def apply_thresholds(
    y_pred_proba: np.ndarray,
    sensitive_attr: np.ndarray,
    threshold_priv: float,
    threshold_unpriv: float,
    privileged_value: Any = "Male",
    unprivileged_value: Any = "Female",
) -> np.ndarray:
    """
    Apply group-specific thresholds to probabilities.

    Use this function to apply thresholds learned on validation data
    to test data WITHOUT using test labels.

    Parameters
    ----------
    y_pred_proba : np.ndarray
        Predicted probabilities.
    sensitive_attr : np.ndarray
        Sensitive attribute values.
    threshold_priv : float
        Threshold for privileged group.
    threshold_unpriv : float
        Threshold for unprivileged group.
    privileged_value : Any
        Value identifying the privileged group.
    unprivileged_value : Any
        Value identifying the unprivileged group.

    Returns
    -------
    np.ndarray
        Binary predictions with group-specific thresholds applied.
    """
    y_pred_proba = np.asarray(y_pred_proba)
    sensitive_attr = np.asarray(sensitive_attr)

    priv_mask = sensitive_attr == privileged_value
    unpriv_mask = sensitive_attr == unprivileged_value

    y_pred = np.zeros(len(y_pred_proba), dtype=int)
    y_pred[priv_mask] = (y_pred_proba[priv_mask] >= threshold_priv).astype(int)
    y_pred[unpriv_mask] = (y_pred_proba[unpriv_mask] >= threshold_unpriv).astype(int)

    # Handle any remaining groups with base threshold
    other_mask = ~(priv_mask | unpriv_mask)
    if other_mask.any():
        y_pred[other_mask] = (y_pred_proba[other_mask] >= threshold_priv).astype(int)

    return y_pred


def tune_equal_opportunity(
    y_val: np.ndarray,
    y_proba_val: np.ndarray,
    sensitive_val: np.ndarray,
    privileged_value: Any = "Male",
    unprivileged_value: Any = "Female",
    base_threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Tune EO thresholds on validation data only.

    This function is designed to be used in a leakage-free evaluation protocol:
    1. Call tune_equal_opportunity() on validation data to get thresholds
    2. Call apply_thresholds() on test data using learned thresholds

    Parameters
    ----------
    y_val : np.ndarray
        Validation ground truth labels.
    y_proba_val : np.ndarray
        Validation predicted probabilities.
    sensitive_val : np.ndarray
        Validation sensitive attribute values.
    privileged_value : Any
        Value identifying the privileged group.
    unprivileged_value : Any
        Value identifying the unprivileged group.
    base_threshold : float
        Base decision threshold.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - threshold_priv: Threshold for privileged group
        - threshold_unpriv: Threshold for unprivileged group
        - tpr_priv_val: Privileged TPR on validation
        - tpr_unpriv_before_val: Unprivileged TPR before adjustment on validation
        - tpr_unpriv_after_val: Unprivileged TPR after adjustment on validation
    """
    _, info = equal_opportunity_postprocessing(
        y_true=y_val,
        y_pred_proba=y_proba_val,
        sensitive_attr=sensitive_val,
        privileged_value=privileged_value,
        unprivileged_value=unprivileged_value,
        base_threshold=base_threshold,
    )

    return {
        "threshold_priv": info["threshold_priv"],
        "threshold_unpriv": info["threshold_unpriv"],
        "tpr_priv_val": info["tpr_priv"],
        "tpr_unpriv_before_val": info["tpr_unpriv_before"],
        "tpr_unpriv_after_val": info["tpr_unpriv_after"],
    }
