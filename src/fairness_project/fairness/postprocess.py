"""Equal Opportunity post-processing for bias mitigation."""

from __future__ import annotations

from typing import Any

import numpy as np


def _validate_same_length(**arrays: np.ndarray) -> None:
    lengths = {name: len(np.asarray(value)) for name, value in arrays.items()}
    if not lengths or len(set(lengths.values())) == 1:
        return
    raise ValueError(f"Inputs must have equal lengths, got {lengths}")


def _validate_probabilities(values: np.ndarray, name: str) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{name} must contain only finite values")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"{name} must contain values between 0 and 1")
    return probabilities


def _validate_threshold(value: float, name: str) -> float:
    threshold = float(value)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(f"{name} must be a finite value between 0 and 1")
    return threshold


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
        True positive rate. Returns NaN if no positive examples exist.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    _validate_same_length(y_true=y_true, y_pred=y_pred)
    mask_pos = y_true == 1
    if mask_pos.sum() == 0:
        return float("nan")

    return float((y_pred[mask_pos] == 1).mean())


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_tpr: float,
    n_thresholds: int = 101,
    search_range: tuple[float, float] = (0.0, 0.5),
    reference_threshold: float = 0.5,
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
    _validate_same_length(y_true=y_true, y_proba=y_proba)
    y_proba = _validate_probabilities(y_proba, "y_proba")
    if not np.isfinite(target_tpr) or not 0 <= target_tpr <= 1:
        raise ValueError("target_tpr must be a finite value between 0 and 1")
    if n_thresholds < 2:
        raise ValueError("n_thresholds must be at least 2")
    search_min = _validate_threshold(search_range[0], "search_range minimum")
    search_max = _validate_threshold(search_range[1], "search_range maximum")
    if search_min > search_max:
        raise ValueError("search_range minimum cannot exceed its maximum")
    reference_threshold = _validate_threshold(reference_threshold, "reference_threshold")
    if not (y_true == 1).any():
        raise ValueError("TPR is undefined because this group has no positive labels")

    thresholds = np.linspace(search_min, search_max, n_thresholds)
    best_threshold = search_range[1]
    best_diff = float("inf")
    best_tpr = 0.0

    for th in thresholds:
        preds = (y_proba >= th).astype(int)
        tpr = compute_tpr(y_true, preds)
        diff = abs(tpr - target_tpr)

        is_better_fit = diff < best_diff
        is_smaller_change = np.isclose(diff, best_diff) and abs(th - reference_threshold) < abs(
            best_threshold - reference_threshold
        )
        if is_better_fit or is_smaller_change:
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
    n_thresholds: int = 101,
    search_range: tuple[float, float] = (0.0, 0.5),
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
    y_pred_proba = _validate_probabilities(y_pred_proba, "y_pred_proba")
    sensitive_attr = np.asarray(sensitive_attr)
    _validate_same_length(
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        sensitive_attr=sensitive_attr,
    )
    base_threshold = _validate_threshold(base_threshold, "base_threshold")

    priv_mask = sensitive_attr == privileged_value
    unpriv_mask = sensitive_attr == unprivileged_value

    # Sanity checks
    if priv_mask.sum() == 0:
        raise ValueError(f"No samples with privileged value '{privileged_value}' found.")
    if unpriv_mask.sum() == 0:
        raise ValueError(f"No samples with unprivileged value '{unprivileged_value}' found.")
    unknown_values = set(np.unique(sensitive_attr)) - {privileged_value, unprivileged_value}
    if unknown_values:
        raise ValueError(f"Unexpected sensitive attribute values: {sorted(unknown_values)}")
    if not (y_true[priv_mask] == 1).any():
        raise ValueError(f"TPR is undefined for '{privileged_value}': no positive labels")
    if not (y_true[unpriv_mask] == 1).any():
        raise ValueError(f"TPR is undefined for '{unprivileged_value}': no positive labels")

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
        n_thresholds=n_thresholds,
        search_range=search_range,
        reference_threshold=base_threshold,
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
    y_pred_proba = _validate_probabilities(y_pred_proba, "y_pred_proba")
    sensitive_attr = np.asarray(sensitive_attr)
    _validate_same_length(y_pred_proba=y_pred_proba, sensitive_attr=sensitive_attr)
    threshold_priv = _validate_threshold(threshold_priv, "threshold_priv")
    threshold_unpriv = _validate_threshold(threshold_unpriv, "threshold_unpriv")

    priv_mask = sensitive_attr == privileged_value
    unpriv_mask = sensitive_attr == unprivileged_value

    unknown_values = set(np.unique(sensitive_attr)) - {privileged_value, unprivileged_value}
    if unknown_values:
        raise ValueError(f"Unexpected sensitive attribute values: {sorted(unknown_values)}")

    y_pred = np.zeros(len(y_pred_proba), dtype=int)
    y_pred[priv_mask] = (y_pred_proba[priv_mask] >= threshold_priv).astype(int)
    y_pred[unpriv_mask] = (y_pred_proba[unpriv_mask] >= threshold_unpriv).astype(int)

    return y_pred


def tune_equal_opportunity(
    y_val: np.ndarray,
    y_proba_val: np.ndarray,
    sensitive_val: np.ndarray,
    privileged_value: Any = "Male",
    unprivileged_value: Any = "Female",
    base_threshold: float = 0.5,
    n_thresholds: int = 101,
    search_range: tuple[float, float] = (0.0, 0.5),
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
        n_thresholds=n_thresholds,
        search_range=search_range,
    )

    return {
        "threshold_priv": info["threshold_priv"],
        "threshold_unpriv": info["threshold_unpriv"],
        "tpr_priv_val": info["tpr_priv"],
        "tpr_unpriv_before_val": info["tpr_unpriv_before"],
        "tpr_unpriv_after_val": info["tpr_unpriv_after"],
    }
