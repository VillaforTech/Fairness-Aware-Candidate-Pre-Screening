"""Uncertainty-aware diagnostics for observed intersectional groups.

The functions in this module are descriptive audit tools. They do not infer a
causal effect, establish fairness, or turn small cells into reliable evidence.
When sampling weights are supplied, rates are weighted and Wilson intervals use
Kish's effective sample size as an explicitly labelled sensitivity estimate.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _one_dimensional(values: ArrayLike, name: str) -> NDArray[Any]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if len(array) == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _binary(values: ArrayLike, name: str) -> NDArray[np.int_]:
    array = _one_dimensional(values, name)
    try:
        numeric = array.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only binary values 0 and 1") from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, [0.0, 1.0]).all():
        raise ValueError(f"{name} must contain only binary values 0 and 1")
    return numeric.astype(int)


def _probabilities(values: ArrayLike) -> NDArray[np.float64]:
    array = _one_dimensional(values, "y_proba").astype(float)
    if not np.isfinite(array).all():
        raise ValueError("y_proba must contain only finite values")
    if ((array < 0) | (array > 1)).any():
        raise ValueError("y_proba must contain values between 0 and 1")
    return array


def _weights(values: ArrayLike | None, length: int) -> tuple[NDArray[np.float64], bool]:
    if values is None:
        return np.ones(length, dtype=float), False
    array = _one_dimensional(values, "sample_weight").astype(float)
    if len(array) != length:
        raise ValueError("sample_weight must have the same length as the predictions")
    if not np.isfinite(array).all():
        raise ValueError("sample_weight must contain only finite values")
    if (array < 0).any():
        raise ValueError("sample_weight must be nonnegative")
    if float(array.sum()) <= 0:
        raise ValueError("sample_weight must have positive total weight")
    return array, True


def _json_scalar(value: Any, name: str) -> str | int | float | bool:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        raise ValueError(f"{name} must not contain missing values")
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"{name} must not contain missing or non-finite values")
    if not isinstance(value, (str, int, float, bool)):
        raise ValueError(f"{name} values must be scalar strings, numbers, or booleans")
    return value


def _group_dimensions(
    groups: Mapping[str, ArrayLike] | ArrayLike,
    length: int,
) -> tuple[list[str], list[tuple[str | int | float | bool, ...]]]:
    if isinstance(groups, Mapping):
        if not groups:
            raise ValueError("groups must contain at least one dimension")
        dimensions = list(groups)
        if any(not isinstance(name, str) or not name.strip() for name in dimensions):
            raise ValueError("group dimension names must be non-empty strings")
        columns: list[list[str | int | float | bool]] = []
        for name in dimensions:
            column = _one_dimensional(groups[name], f"groups[{name!r}]")
            if len(column) != length:
                raise ValueError("Every group dimension must have the same length as y_true")
            columns.append([_json_scalar(value, f"groups[{name!r}]") for value in column])
        rows = list(zip(*columns, strict=True))
        return dimensions, rows

    column = _one_dimensional(groups, "groups")
    if len(column) != length:
        raise ValueError("groups must have the same length as y_true")
    return ["group"], [(_json_scalar(value, "groups"),) for value in column]


def _format_group_id(dimensions: Sequence[str], values: Sequence[str | int | float | bool]) -> str:
    parts = []
    for dimension, value in zip(dimensions, values, strict=True):
        rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        parts.append(f"{dimension}={rendered}")
    return " | ".join(parts)


def _effective_sample_size(weights: NDArray[np.float64]) -> float:
    weight_sum = float(weights.sum())
    squared_sum = float(np.square(weights).sum())
    if weight_sum <= 0 or squared_sum <= 0:
        return 0.0
    return weight_sum**2 / squared_sum


def _wilson_interval(rate: float, effective_n: float, confidence: float) -> tuple[float, float]:
    if effective_n <= 0:
        raise ValueError("effective_n must be positive for a Wilson interval")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    z_squared = z**2
    denominator = 1 + z_squared / effective_n
    centre = (rate + z_squared / (2 * effective_n)) / denominator
    half_width = (
        z
        * np.sqrt(rate * (1 - rate) / effective_n + z_squared / (4 * effective_n**2))
        / denominator
    )
    return max(0.0, float(centre - half_width)), min(1.0, float(centre + half_width))


def _rate_metric(
    *,
    numerator: NDArray[np.bool_],
    denominator: NDArray[np.bool_],
    weights: NDArray[np.float64],
    weighted: bool,
    confidence: float,
    evidence_reasons: list[str],
) -> dict[str, Any]:
    denominator_weights = weights[denominator]
    denominator_weight = float(denominator_weights.sum())
    denominator_count = int(denominator.sum())
    if denominator_count == 0 or denominator_weight <= 0:
        reasons = [*evidence_reasons, "zero_metric_denominator"]
        return {
            "estimate": None,
            "interval": None,
            "denominator_count": denominator_count,
            "denominator_weight": denominator_weight,
            "effective_n": 0.0,
            "evidence_status": "not_estimable",
            "evidence_reasons": sorted(set(reasons)),
        }

    numerator_weight = float(weights[numerator & denominator].sum())
    estimate = numerator_weight / denominator_weight
    effective_n = (
        _effective_sample_size(denominator_weights) if weighted else float(denominator_count)
    )
    lower, upper = _wilson_interval(estimate, effective_n, confidence)
    interval_method = (
        "weighted_wilson_kish_effective_sample_size_sensitivity" if weighted else "wilson_score"
    )
    return {
        "estimate": float(estimate),
        "interval": {
            "lower": lower,
            "upper": upper,
            "confidence": confidence,
            "method": interval_method,
        },
        "denominator_count": denominator_count,
        "denominator_weight": denominator_weight,
        "effective_n": float(effective_n),
        "evidence_status": "limited" if evidence_reasons else "sufficient",
        "evidence_reasons": sorted(set(evidence_reasons)),
    }


def _calibration(
    y_true: NDArray[np.int_],
    y_proba: NDArray[np.float64],
    weights: NDArray[np.float64],
    *,
    ece_bins: int,
    evidence_reasons: list[str],
) -> dict[str, Any]:
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return {
            "brier_score": None,
            "ece": None,
            "bins": ece_bins,
            "evidence_status": "not_estimable",
            "evidence_reasons": sorted({*evidence_reasons, "zero_group_weight"}),
        }

    brier = float(np.average(np.square(y_proba - y_true), weights=weights))
    bin_indices = np.minimum((y_proba * ece_bins).astype(int), ece_bins - 1)
    ece = 0.0
    for index in range(ece_bins):
        mask = bin_indices == index
        bin_weight = float(weights[mask].sum())
        if bin_weight <= 0:
            continue
        observed = float(np.average(y_true[mask], weights=weights[mask]))
        predicted = float(np.average(y_proba[mask], weights=weights[mask]))
        ece += (bin_weight / weight_sum) * abs(observed - predicted)
    return {
        "brier_score": brier,
        "ece": float(ece),
        "bins": ece_bins,
        "evidence_status": "limited" if evidence_reasons else "sufficient",
        "evidence_reasons": sorted(set(evidence_reasons)),
    }


def _span(
    group_rows: Sequence[dict[str, Any]],
    metric_name: str,
) -> dict[str, Any]:
    observations: list[tuple[str, float]] = []
    for row in group_rows:
        if metric_name in {"selection_rate", "tpr", "fpr"}:
            metric = row[metric_name]
            estimate = metric.get("estimate")
        else:
            metric = row["calibration"]
            estimate = metric.get(metric_name)
        if metric.get("evidence_status") != "sufficient":
            continue
        if estimate is not None and np.isfinite(float(estimate)):
            observations.append((str(row["group_id"]), float(estimate)))

    if len(observations) < 2:
        return {
            "minimum": None,
            "maximum": None,
            "absolute_span": None,
            "minimum_group": None,
            "maximum_group": None,
            "eligible_group_count": len(observations),
            "evidence_status": "insufficient_groups",
        }
    minimum_group, minimum = min(observations, key=lambda item: (item[1], item[0]))
    maximum_group, maximum = max(observations, key=lambda item: (item[1], item[0]))
    return {
        "minimum": minimum,
        "maximum": maximum,
        "absolute_span": maximum - minimum,
        "minimum_group": minimum_group,
        "maximum_group": maximum_group,
        "eligible_group_count": len(observations),
        "evidence_status": "sufficient",
    }


def intersectional_diagnostics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_proba: ArrayLike,
    groups: Mapping[str, ArrayLike] | ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
    confidence: float = 0.95,
    ece_bins: int = 10,
    min_support: int = 30,
    min_positive: int = 10,
    min_negative: int = 10,
) -> dict[str, Any]:
    """Describe observed group performance with uncertainty and evidence limits.

    ``groups`` can be a single one-dimensional vector or a mapping of dimension
    names to vectors. A mapping creates intersectional cells from the observed
    combinations. Support thresholds affect evidence status and worst-group
    comparisons; they never hide the underlying descriptive estimates.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if not isinstance(ece_bins, int) or isinstance(ece_bins, bool) or ece_bins < 2:
        raise ValueError("ece_bins must be an integer of at least 2")
    thresholds = {
        "min_support": min_support,
        "min_positive": min_positive,
        "min_negative": min_negative,
    }
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in thresholds.values()
    ):
        raise ValueError("Minimum evidence thresholds must be positive integers")

    truth = _binary(y_true, "y_true")
    predictions = _binary(y_pred, "y_pred")
    probabilities = _probabilities(y_proba)
    if len({len(truth), len(predictions), len(probabilities)}) != 1:
        raise ValueError("y_true, y_pred, and y_proba must have equal lengths")
    weights, weighted = _weights(sample_weight, len(truth))
    dimensions, group_rows = _group_dimensions(groups, len(truth))

    cell_indices: dict[str, list[int]] = {}
    cell_values: dict[str, tuple[str | int | float | bool, ...]] = {}
    for index, cell in enumerate(group_rows):
        group_id = _format_group_id(dimensions, cell)
        cell_indices.setdefault(group_id, []).append(index)
        cell_values[group_id] = cell

    rows: list[dict[str, Any]] = []
    for group_id in sorted(cell_indices):
        cell = cell_values[group_id]
        indices = np.asarray(cell_indices[group_id], dtype=int)
        group_truth = truth[indices]
        group_pred = predictions[indices]
        group_proba = probabilities[indices]
        group_weights = weights[indices]
        support = len(indices)
        positive_labels = int((group_truth == 1).sum())
        negative_labels = int((group_truth == 0).sum())

        support_reasons = ["support_below_minimum"] if support < min_support else []
        positive_reasons = list(support_reasons)
        if positive_labels < min_positive:
            positive_reasons.append("positive_labels_below_minimum")
        negative_reasons = list(support_reasons)
        if negative_labels < min_negative:
            negative_reasons.append("negative_labels_below_minimum")

        selection_rate = _rate_metric(
            numerator=group_pred == 1,
            denominator=np.ones(support, dtype=bool),
            weights=group_weights,
            weighted=weighted,
            confidence=confidence,
            evidence_reasons=support_reasons,
        )
        tpr = _rate_metric(
            numerator=group_pred == 1,
            denominator=group_truth == 1,
            weights=group_weights,
            weighted=weighted,
            confidence=confidence,
            evidence_reasons=positive_reasons,
        )
        fpr = _rate_metric(
            numerator=group_pred == 1,
            denominator=group_truth == 0,
            weights=group_weights,
            weighted=weighted,
            confidence=confidence,
            evidence_reasons=negative_reasons,
        )
        calibration = _calibration(
            group_truth,
            group_proba,
            group_weights,
            ece_bins=ece_bins,
            evidence_reasons=support_reasons,
        )

        true_positive = (group_truth == 1) & (group_pred == 1)
        false_positive = (group_truth == 0) & (group_pred == 1)
        true_negative = (group_truth == 0) & (group_pred == 0)
        false_negative = (group_truth == 1) & (group_pred == 0)
        all_reasons = sorted(
            set(
                support_reasons
                + positive_reasons
                + negative_reasons
                + ([] if float(group_weights.sum()) > 0 else ["zero_group_weight"])
            )
        )
        rows.append(
            {
                "group_id": group_id,
                "attributes": dict(zip(dimensions, cell, strict=True)),
                "support": {
                    "n": support,
                    "weight": float(group_weights.sum()),
                    "effective_n": float(_effective_sample_size(group_weights)),
                    "positive_labels": positive_labels,
                    "negative_labels": negative_labels,
                    "positive_weight": float(group_weights[group_truth == 1].sum()),
                    "negative_weight": float(group_weights[group_truth == 0].sum()),
                },
                "confusion": {
                    "true_positive": int(true_positive.sum()),
                    "false_positive": int(false_positive.sum()),
                    "true_negative": int(true_negative.sum()),
                    "false_negative": int(false_negative.sum()),
                    "weighted_true_positive": float(group_weights[true_positive].sum()),
                    "weighted_false_positive": float(group_weights[false_positive].sum()),
                    "weighted_true_negative": float(group_weights[true_negative].sum()),
                    "weighted_false_negative": float(group_weights[false_negative].sum()),
                },
                "selection_rate": selection_rate,
                "tpr": tpr,
                "fpr": fpr,
                "calibration": calibration,
                "evidence_status": "limited" if all_reasons else "sufficient",
                "evidence_reasons": all_reasons,
            }
        )

    return {
        "schema_version": "1.0",
        "n": len(truth),
        "weighted": weighted,
        "dimensions": dimensions,
        "configuration": {
            "confidence": confidence,
            "ece_bins": ece_bins,
            **thresholds,
        },
        "methodology": {
            "rate_interval": (
                "weighted_wilson_kish_effective_sample_size_sensitivity"
                if weighted
                else "wilson_score"
            ),
            "calibration": "weighted_brier_and_fixed_width_expected_calibration_error",
            "scope": "descriptive_observed_groups",
        },
        "groups": rows,
        "worst_group_spans": {
            "selection_rate": _span(rows, "selection_rate"),
            "tpr": _span(rows, "tpr"),
            "fpr": _span(rows, "fpr"),
            "brier_score": _span(rows, "brier_score"),
            "ece": _span(rows, "ece"),
        },
    }
