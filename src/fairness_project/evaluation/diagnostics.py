"""Subgroup diagnostics and paired uncertainty estimates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fairness_project.evaluation.evaluate import evaluate_predictions
from fairness_project.metrics.fairness import false_positive_rate, true_positive_rate


def group_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: pd.Series,
) -> dict[str, dict[str, float | int | None]]:
    """Return auditable cell counts and rates for every observed group."""
    truth = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    group_values = groups.astype(str).to_numpy()
    if not (len(truth) == len(predictions) == len(group_values)):
        raise ValueError("y_true, y_pred, and groups must have equal lengths")

    diagnostics: dict[str, dict[str, float | int | None]] = {}
    for value in sorted(np.unique(group_values)):
        mask = group_values == value
        positives = int((truth[mask] == 1).sum())
        negatives = int((truth[mask] == 0).sum())
        tpr = true_positive_rate(truth[mask], predictions[mask])
        fpr = false_positive_rate(truth[mask], predictions[mask])
        diagnostics[value] = {
            "n": int(mask.sum()),
            "positive_labels": positives,
            "negative_labels": negatives,
            "predicted_positive_rate": float(predictions[mask].mean()),
            "tpr": float(tpr) if np.isfinite(tpr) else None,
            "fpr": float(fpr) if np.isfinite(fpr) else None,
        }
    return diagnostics


def paired_bootstrap_intervals(
    *,
    y_true: np.ndarray,
    baseline_pred: np.ndarray,
    adjusted_pred: np.ndarray,
    sensitive: np.ndarray,
    privileged_group: Any,
    samples: int = 500,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, Any]:
    """Estimate paired test-set intervals while preserving label/group cells."""
    if samples < 1:
        raise ValueError("samples must be at least 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    truth = np.asarray(y_true)
    baseline = np.asarray(baseline_pred)
    adjusted = np.asarray(adjusted_pred)
    group = np.asarray(sensitive)
    if len({len(truth), len(baseline), len(adjusted), len(group)}) != 1:
        raise ValueError("Bootstrap inputs must have equal lengths")

    strata = pd.Series(group.astype(str)) + "|" + pd.Series(truth.astype(str))
    stratum_indices = [indices.to_numpy() for _, indices in strata.groupby(strata).groups.items()]
    rng = np.random.default_rng(random_state)
    metric_names = ("accuracy", "SPD", "DI", "TPR_gap", "FPR_gap")
    values: dict[str, dict[str, list[float]]] = {
        condition: {metric: [] for metric in metric_names}
        for condition in ("baseline", "adjusted", "change")
    }

    for _ in range(samples):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in stratum_indices]
        )
        base_metrics = evaluate_predictions(
            truth[sampled], baseline[sampled], group[sampled], privileged_group
        )
        adjusted_metrics = evaluate_predictions(
            truth[sampled], adjusted[sampled], group[sampled], privileged_group
        )
        for metric in metric_names:
            before = float(base_metrics[metric])
            after = float(adjusted_metrics[metric])
            if np.isfinite(before) and np.isfinite(after):
                values["baseline"][metric].append(before)
                values["adjusted"][metric].append(after)
                values["change"][metric].append(after - before)

    alpha = (1 - confidence) / 2
    intervals: dict[str, dict[str, dict[str, float] | None]] = {}
    for condition, metrics in values.items():
        intervals[condition] = {}
        for metric, observations in metrics.items():
            if not observations:
                intervals[condition][metric] = None
                continue
            lower, median, upper = np.quantile(observations, [alpha, 0.5, 1 - alpha])
            intervals[condition][metric] = {
                "lower": float(lower),
                "median": float(median),
                "upper": float(upper),
            }

    return {
        "method": "paired stratified bootstrap over test rows",
        "samples": samples,
        "confidence": confidence,
        "random_state": random_state,
        "intervals": intervals,
    }
