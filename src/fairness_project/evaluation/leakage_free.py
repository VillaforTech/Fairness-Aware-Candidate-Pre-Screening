"""
Leakage-free evaluation protocol for Equal Opportunity post-processing.

The key principle: EO threshold tuning must NOT use test set labels.

Protocol:
1. Split data into train/val/test
2. Train model on train set
3. Tune EO thresholds using validation set labels
4. Apply learned thresholds to test set (without using test labels)
5. Report final metrics on test set

This module provides utilities to implement this protocol correctly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def create_train_val_test_split(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Create train/val/test split from existing train/test split.

    Takes a portion of training data for validation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'split' column containing 'train' and 'test'.
    val_ratio : float
        Ratio of training data to use for validation.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with updated 'split' column ('train', 'val', 'test').
    """
    df = df.copy()
    np.random.seed(random_state)

    train_mask = df["split"] == "train"
    train_indices = df[train_mask].index.tolist()

    n_val = int(len(train_indices) * val_ratio)
    val_indices = np.random.choice(train_indices, size=n_val, replace=False)

    df.loc[val_indices, "split"] = "val"

    return df


class LeakageFreeEvaluator:
    """
    Evaluator that ensures no test set leakage during EO threshold tuning.

    This class enforces the correct protocol:
    - Thresholds are tuned on validation data only
    - Test data is never seen during tuning
    - Final metrics are computed on test data after tuning
    """

    def __init__(
        self,
        sensitive_col: str = "sex",
        privileged_value: str = "Male",
        unprivileged_value: str = "Female",
        base_threshold: float = 0.5,
    ):
        """
        Initialize the evaluator.

        Parameters
        ----------
        sensitive_col : str
            Name of sensitive attribute column.
        privileged_value : str
            Value identifying the privileged group.
        unprivileged_value : str
            Value identifying the unprivileged group.
        base_threshold : float
            Base decision threshold.
        """
        self.sensitive_col = sensitive_col
        self.privileged_value = privileged_value
        self.unprivileged_value = unprivileged_value
        self.base_threshold = base_threshold

        # Learned thresholds (set during tuning)
        self.threshold_priv: float | None = None
        self.threshold_unpriv: float | None = None
        self.tuning_info: dict[str, Any] = {}

    def tune_thresholds_on_validation(
        self,
        y_val: np.ndarray,
        y_proba_val: np.ndarray,
        sensitive_val: np.ndarray,
    ) -> dict[str, Any]:
        """
        Tune EO thresholds using validation data ONLY.

        This is the only method that should use ground truth labels
        for threshold selection.

        Parameters
        ----------
        y_val : np.ndarray
            Validation set true labels.
        y_proba_val : np.ndarray
            Validation set predicted probabilities.
        sensitive_val : np.ndarray
            Validation set sensitive attribute values.

        Returns
        -------
        dict[str, Any]
            Tuning information including learned thresholds.
        """
        from fairness_project.fairness.postprocess import tune_equal_opportunity

        info = tune_equal_opportunity(
            y_val=y_val,
            y_proba_val=y_proba_val,
            sensitive_val=sensitive_val,
            privileged_value=self.privileged_value,
            unprivileged_value=self.unprivileged_value,
            base_threshold=self.base_threshold,
        )

        self.threshold_priv = info["threshold_priv"]
        self.threshold_unpriv = info["threshold_unpriv"]
        self.tuning_info = info

        return info

    def apply_to_test(
        self,
        y_proba_test: np.ndarray,
        sensitive_test: np.ndarray,
    ) -> np.ndarray:
        """
        Apply learned thresholds to test data WITHOUT using test labels.

        Parameters
        ----------
        y_proba_test : np.ndarray
            Test set predicted probabilities.
        sensitive_test : np.ndarray
            Test set sensitive attribute values.

        Returns
        -------
        np.ndarray
            Adjusted binary predictions for test set.

        Raises
        ------
        RuntimeError
            If thresholds haven't been tuned yet.
        """
        if self.threshold_priv is None or self.threshold_unpriv is None:
            raise RuntimeError("Thresholds not tuned. Call tune_thresholds_on_validation first.")

        from fairness_project.fairness.postprocess import apply_thresholds

        return apply_thresholds(
            y_pred_proba=y_proba_test,
            sensitive_attr=sensitive_test,
            threshold_priv=self.threshold_priv,
            threshold_unpriv=self.threshold_unpriv,
            privileged_value=self.privileged_value,
            unprivileged_value=self.unprivileged_value,
        )

    def evaluate_on_test(
        self,
        y_test: np.ndarray,
        y_pred_test: np.ndarray,
        sensitive_test: np.ndarray,
    ) -> dict[str, Any]:
        """
        Evaluate predictions on test set.

        This should be called AFTER apply_to_test to compute final metrics.

        Parameters
        ----------
        y_test : np.ndarray
            Test set true labels.
        y_pred_test : np.ndarray
            Test set predictions (from apply_to_test).
        sensitive_test : np.ndarray
            Test set sensitive attribute values.

        Returns
        -------
        dict[str, Any]
            Evaluation metrics.
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        from fairness_project.metrics.fairness import (
            compute_fairness_metrics,
            true_positive_rate,
        )

        # Performance metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred_test),
            "precision": precision_score(y_test, y_pred_test, zero_division=0),
            "recall": recall_score(y_test, y_pred_test, zero_division=0),
            "f1": f1_score(y_test, y_pred_test, zero_division=0),
        }

        # Fairness metrics
        fair = compute_fairness_metrics(
            y_true=y_test,
            y_pred=y_pred_test,
            sensitive=sensitive_test,
            privileged_group=self.privileged_value,
        )
        metrics["SPD"] = fair["SPD"]
        metrics["DI"] = fair["DI"]

        # TPR by group
        priv_mask = sensitive_test == self.privileged_value
        unpriv_mask = sensitive_test == self.unprivileged_value

        tpr_priv = true_positive_rate(y_test[priv_mask], y_pred_test[priv_mask])
        tpr_unpriv = true_positive_rate(y_test[unpriv_mask], y_pred_test[unpriv_mask])

        metrics["TPR_priv"] = tpr_priv
        metrics["TPR_unpriv"] = tpr_unpriv
        metrics["TPR_gap"] = tpr_priv - tpr_unpriv

        return metrics


def run_leakage_free_evaluation(
    model,
    df: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_val: np.ndarray,
    y_test: np.ndarray,
    sensitive_col: str = "sex",
    privileged_value: str = "Male",
    unprivileged_value: str = "Female",
) -> dict[str, Any]:
    """
    Run complete leakage-free evaluation for a trained model.

    Parameters
    ----------
    model : sklearn.Pipeline
        Trained model with predict and predict_proba methods.
    df : pd.DataFrame
        Full dataframe with sensitive attributes.
    X_val : pd.DataFrame
        Validation features.
    X_test : pd.DataFrame
        Test features.
    y_val : np.ndarray
        Validation labels.
    y_test : np.ndarray
        Test labels.
    sensitive_col : str
        Sensitive attribute column name.
    privileged_value : str
        Privileged group value.
    unprivileged_value : str
        Unprivileged group value.

    Returns
    -------
    dict[str, Any]
        Results including baseline metrics, EO metrics, and thresholds.
    """
    # Get sensitive attributes for val and test
    df_val = df[df["split"] == "val"]
    df_test = df[df["split"] == "test"]

    sensitive_val = df_val[sensitive_col].values
    sensitive_test = df_test[sensitive_col].values

    # Get probabilities
    y_proba_val = model.predict_proba(X_val)[:, 1]
    y_proba_test = model.predict_proba(X_test)[:, 1]

    # Baseline predictions (threshold 0.5)
    y_pred_base_test = (y_proba_test >= 0.5).astype(int)

    # Initialize evaluator
    evaluator = LeakageFreeEvaluator(
        sensitive_col=sensitive_col,
        privileged_value=privileged_value,
        unprivileged_value=unprivileged_value,
    )

    # Tune thresholds on VALIDATION (not test)
    tuning_info = evaluator.tune_thresholds_on_validation(
        y_val=y_val,
        y_proba_val=y_proba_val,
        sensitive_val=sensitive_val,
    )

    # Apply to TEST (without using test labels)
    y_pred_eo_test = evaluator.apply_to_test(
        y_proba_test=y_proba_test,
        sensitive_test=sensitive_test,
    )

    # Evaluate baseline on test
    baseline_metrics = evaluator.evaluate_on_test(
        y_test=y_test,
        y_pred_test=y_pred_base_test,
        sensitive_test=sensitive_test,
    )

    # Evaluate EO-adjusted on test
    eo_metrics = evaluator.evaluate_on_test(
        y_test=y_test,
        y_pred_test=y_pred_eo_test,
        sensitive_test=sensitive_test,
    )

    return {
        "baseline": baseline_metrics,
        "eo_adjusted": eo_metrics,
        "thresholds": {
            "privileged": evaluator.threshold_priv,
            "unprivileged": evaluator.threshold_unpriv,
        },
        "tuning_info": tuning_info,
        "predictions": {
            "baseline": y_pred_base_test,
            "eo_adjusted": y_pred_eo_test,
        },
    }
