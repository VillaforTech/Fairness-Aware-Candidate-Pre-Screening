import numpy as np
import pandas as pd


def demographic_parity(y, sensitive):
    """
    Computes P(Y=1 | group) for each group.
    """
    df = pd.DataFrame({"y": y, "sensitive": sensitive})
    return df.groupby("sensitive")["y"].mean().to_dict()


def statistical_parity_difference(y, sensitive, privileged_group):
    """
    SPD = P(Y=1 | privileged) - P(Y=1 | unprivileged)
    """
    dp = demographic_parity(y, sensitive)

    privileged_rate = dp[privileged_group]
    # unprivileged = the other group
    unprivileged_group = [g for g in dp.keys() if g != privileged_group][0]
    unprivileged_rate = dp[unprivileged_group]

    return privileged_rate - unprivileged_rate


def disparate_impact(y, sensitive, privileged_group):
    """
    DI = P(Y=1 | unprivileged) / P(Y=1 | privileged)
    """
    dp = demographic_parity(y, sensitive)

    privileged_rate = dp[privileged_group]
    unprivileged_group = [g for g in dp.keys() if g != privileged_group][0]
    unprivaged_rate = dp[unprivileged_group]

    # avoid division-by-zero
    if privileged_rate == 0:
        return np.nan

    return unprivaged_rate / privileged_rate


def true_positive_rate(y_true, y_pred):
    """
    Computes True Positive Rate (TPR) = TP / (TP + FN).

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    float
        True positive rate (recall).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    positives = y_true == 1
    n_positives = positives.sum()

    if n_positives == 0:
        return 0.0

    true_positives = ((y_true == 1) & (y_pred == 1)).sum()
    return true_positives / n_positives


def compute_fairness_metrics(y_true, y_pred, sensitive, privileged_group):
    results = {}

    # DP (based on predictions)
    results["DP"] = demographic_parity(y_pred, sensitive)

    # SPD (predictions vs sensitive attribute)
    results["SPD"] = statistical_parity_difference(y_pred, sensitive, privileged_group)

    # DI
    results["DI"] = disparate_impact(y_pred, sensitive, privileged_group)

    return results
