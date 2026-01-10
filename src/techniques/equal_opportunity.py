# src/techniques/equal_opportunity.py

import numpy as np


def compute_tpr(y_true, y_pred):
    """
    True Positive Rate = TP / (TP + FN).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask_pos = y_true == 1
    if mask_pos.sum() == 0:
        return 0.0

    return (y_pred[mask_pos] == 1).mean()


def equal_opportunity_postprocessing(y_true, y_pred_proba, sensitive_attr):
    """
    Equal Opportunity post-processing for a binary classifier.

    Idea:
      - Base model uses a global threshold 0.5.
      - We keep threshold 0.5 for the PRIVILEGED group ("Male").
      - For the UNPRIVILEGED group ("Female"), we LOWER the threshold
        until their TPR is as close as possible to male TPR.

    Inputs:
        y_true:          1D array-like of true labels (0/1).
        y_pred_proba:    1D array-like of predicted probabilities for class 1.
        sensitive_attr:  1D array-like of sensitive attribute values, expected
                         to contain "Male" / "Female".

    Returns:
        y_pred_adj: 1D numpy array of EO-adjusted binary predictions.
        info:       dict with TPRs before/after and chosen threshold.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    sensitive_attr = np.asarray(sensitive_attr)

    priv_mask = sensitive_attr == "Male"
    unpriv_mask = sensitive_attr == "Female"

    # Sanity checks
    if priv_mask.sum() == 0 or unpriv_mask.sum() == 0:
        raise ValueError(
            "Both privileged ('Male') and unprivileged ('Female') groups "
            "must be present in sensitive_attr."
        )

    # Baseline predictions with common threshold 0.5
    y_pred_base = (y_pred_proba >= 0.5).astype(int)

    # Baseline TPRs
    tpr_priv = compute_tpr(y_true[priv_mask], y_pred_base[priv_mask])
    tpr_unpriv_before = compute_tpr(y_true[unpriv_mask], y_pred_base[unpriv_mask])

    # If unprivileged TPR is already >= privileged TPR, we leave predictions unchanged
    if tpr_unpriv_before >= tpr_priv:
        info = {
            "tpr_priv": float(tpr_priv),
            "tpr_unpriv_before": float(tpr_unpriv_before),
            "tpr_unpriv_after": float(tpr_unpriv_before),
            "threshold_used": 0.5,
            "note": "No adjustment: unprivileged TPR already >= privileged TPR.",
        }
        return y_pred_base, info

    # Otherwise: search for a lower threshold for the unprivileged group
    thresholds = np.linspace(0.0, 0.5, 101)
    best_threshold = 0.5
    best_diff = float("inf")

    for th in thresholds:
        preds_unpriv = (y_pred_proba[unpriv_mask] >= th).astype(int)
        tpr_unpriv_candidate = compute_tpr(y_true[unpriv_mask], preds_unpriv)
        diff = abs(tpr_unpriv_candidate - tpr_priv)

        if diff < best_diff:
            best_diff = diff
            best_threshold = th

    # Construct final adjusted predictions
    y_pred_adj = y_pred_base.copy()
    y_pred_adj[unpriv_mask] = (y_pred_proba[unpriv_mask] >= best_threshold).astype(int)

    tpr_unpriv_after = compute_tpr(y_true[unpriv_mask], y_pred_adj[unpriv_mask])

    info = {
        "tpr_priv": float(tpr_priv),
        "tpr_unpriv_before": float(tpr_unpriv_before),
        "tpr_unpriv_after": float(tpr_unpriv_after),
        "threshold_used": float(best_threshold),
        "note": "EO post-processing applied (female threshold lowered).",
    }
    return y_pred_adj, info
