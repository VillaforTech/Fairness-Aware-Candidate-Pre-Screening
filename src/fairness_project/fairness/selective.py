"""Global selective-classification policy with an explicit human-review band.

The policy is selected using validation labels only. Per-row decisions depend
solely on the model probability and a single global band around the base
threshold. Group labels are accepted only by the held-out evaluation function
and therefore cannot affect an individual decision.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

AUTO_NEGATIVE = "auto_negative"
AUTO_POSITIVE = "auto_positive"
REVIEW = "review"


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


def _probabilities(values: ArrayLike, name: str = "y_proba") -> NDArray[np.float64]:
    array = _one_dimensional(values, name).astype(float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if ((array < 0) | (array > 1)).any():
        raise ValueError(f"{name} must contain values between 0 and 1")
    return array


def _weights(values: ArrayLike | None, length: int) -> tuple[NDArray[np.float64], bool]:
    if values is None:
        return np.ones(length, dtype=float), False
    weights = _one_dimensional(values, "sample_weight").astype(float)
    if len(weights) != length:
        raise ValueError("sample_weight must have the same length as y_true")
    if not np.isfinite(weights).all():
        raise ValueError("sample_weight must contain only finite values")
    if (weights < 0).any():
        raise ValueError("sample_weight must be nonnegative")
    if float(weights.sum()) <= 0:
        raise ValueError("sample_weight must have positive total weight")
    return weights, True


def _unit_interval(value: float, name: str) -> float:
    converted = float(value)
    if not np.isfinite(converted) or not 0 <= converted <= 1:
        raise ValueError(f"{name} must be a finite value between 0 and 1")
    return converted


@dataclass(frozen=True)
class ReviewBandCandidate:
    """Auditable validation result for one candidate half-width."""

    half_width: float
    lower_threshold: float
    upper_threshold: float
    automated_n: int
    review_n: int
    automated_weight: float
    review_weight: float
    automation_coverage: float
    automated_error_rate: float | None
    feasible: bool
    rejection_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable candidate record."""
        return {
            "half_width": self.half_width,
            "lower_threshold": self.lower_threshold,
            "upper_threshold": self.upper_threshold,
            "automated_n": self.automated_n,
            "review_n": self.review_n,
            "automated_weight": self.automated_weight,
            "review_weight": self.review_weight,
            "automation_coverage": self.automation_coverage,
            "automated_error_rate": self.automated_error_rate,
            "feasible": self.feasible,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class ReviewBandPolicy:
    """Frozen global review-band policy selected on validation data."""

    base_threshold: float
    half_width: float
    lower_threshold: float
    upper_threshold: float
    max_automated_error_rate: float
    min_automated_samples: int
    validation_n: int
    validation_automated_n: int
    validation_review_n: int
    validation_automated_weight: float
    validation_review_weight: float
    validation_automation_coverage: float
    validation_automated_error_rate: float
    weighted_selection: bool
    candidates: tuple[ReviewBandCandidate, ...]
    schema_version: str = "1.0"
    selection_method: str = "maximize_automation_coverage_under_validation_error_constraint"

    def __post_init__(self) -> None:
        base = _unit_interval(self.base_threshold, "base_threshold")
        lower = _unit_interval(self.lower_threshold, "lower_threshold")
        upper = _unit_interval(self.upper_threshold, "upper_threshold")
        if not lower <= base <= upper:
            raise ValueError("Review-band thresholds must contain the base threshold")
        if not np.isfinite(self.half_width) or self.half_width < 0:
            raise ValueError("half_width must be finite and nonnegative")
        if not np.isclose(lower, max(0.0, base - self.half_width)) or not np.isclose(
            upper, min(1.0, base + self.half_width)
        ):
            raise ValueError("Review-band thresholds are inconsistent with half_width")
        _unit_interval(self.max_automated_error_rate, "max_automated_error_rate")
        _unit_interval(self.validation_automation_coverage, "validation_automation_coverage")
        _unit_interval(self.validation_automated_error_rate, "validation_automated_error_rate")

    def to_dict(self, *, include_candidates: bool = True) -> dict[str, Any]:
        """Return a JSON-serializable policy and optional validation trace."""
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "selection_method": self.selection_method,
            "decision_inputs": ["model_probability"],
            "base_threshold": self.base_threshold,
            "half_width": self.half_width,
            "lower_threshold": self.lower_threshold,
            "upper_threshold": self.upper_threshold,
            "max_automated_error_rate": self.max_automated_error_rate,
            "min_automated_samples": self.min_automated_samples,
            "validation_n": self.validation_n,
            "validation_automated_n": self.validation_automated_n,
            "validation_review_n": self.validation_review_n,
            "validation_automated_weight": self.validation_automated_weight,
            "validation_review_weight": self.validation_review_weight,
            "validation_automation_coverage": self.validation_automation_coverage,
            "validation_automated_error_rate": self.validation_automated_error_rate,
            "weighted_selection": self.weighted_selection,
        }
        if include_candidates:
            result["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return result


def _candidate_widths(
    probabilities: NDArray[np.float64],
    base_threshold: float,
    candidate_half_widths: Sequence[float] | None,
    max_half_width: float,
) -> list[float]:
    if candidate_half_widths is None:
        distances = np.abs(probabilities - base_threshold)
        values = [0.0, *distances[distances <= max_half_width].tolist(), max_half_width]
        if max_half_width > 0 and (distances == 0).any():
            values.append(float(np.nextafter(0.0, 1.0)))
    else:
        values = [0.0, *candidate_half_widths]

    validated: list[float] = []
    for value in values:
        width = float(value)
        if not np.isfinite(width) or width < 0:
            raise ValueError("candidate_half_widths must contain finite nonnegative values")
        if width > max_half_width:
            raise ValueError("candidate_half_widths cannot exceed max_half_width")
        validated.append(width)
    return sorted(set(validated))


def select_review_band(
    y_true_validation: ArrayLike,
    y_proba_validation: ArrayLike,
    *,
    base_threshold: float = 0.5,
    max_automated_error_rate: float = 0.10,
    min_automated_samples: int = 1,
    max_half_width: float | None = None,
    candidate_half_widths: Sequence[float] | None = None,
    sample_weight: ArrayLike | None = None,
) -> ReviewBandPolicy:
    """Select the widest-coverage feasible global policy on validation data.

    Candidate bands are evaluated efficiently from cumulative confidence-ranked
    errors. Width zero is always considered and means no review. For positive
    widths, rows whose absolute distance from ``base_threshold`` is less than
    or equal to the width are sent to review.
    """
    truth = _binary(y_true_validation, "y_true_validation")
    probabilities = _probabilities(y_proba_validation, "y_proba_validation")
    if len(truth) != len(probabilities):
        raise ValueError("y_true_validation and y_proba_validation must have equal lengths")
    weights, weighted = _weights(sample_weight, len(truth))
    base = _unit_interval(base_threshold, "base_threshold")
    error_limit = _unit_interval(max_automated_error_rate, "max_automated_error_rate")
    if (
        not isinstance(min_automated_samples, int)
        or isinstance(min_automated_samples, bool)
        or min_automated_samples < 1
    ):
        raise ValueError("min_automated_samples must be a positive integer")
    if min_automated_samples > len(truth):
        raise ValueError("min_automated_samples cannot exceed the validation sample size")

    largest_possible_width = max(base, 1 - base)
    if max_half_width is None:
        maximum_width = largest_possible_width
    else:
        maximum_width = float(max_half_width)
        if not np.isfinite(maximum_width) or maximum_width < 0:
            raise ValueError("max_half_width must be finite and nonnegative")
        if maximum_width > largest_possible_width:
            raise ValueError("max_half_width cannot extend beyond the probability range")
    widths = _candidate_widths(probabilities, base, candidate_half_widths, maximum_width)

    predictions = (probabilities >= base).astype(int)
    errors = predictions != truth
    distances = np.abs(probabilities - base)
    order = np.argsort(distances, kind="stable")
    sorted_distances = distances[order]
    sorted_weights = weights[order]
    sorted_error_weight = weights[order] * errors[order]
    cumulative_weight = np.cumsum(sorted_weights)
    cumulative_error_weight = np.cumsum(sorted_error_weight)
    total_weight = float(weights.sum())
    total_error_weight = float(np.sum(weights * errors))

    candidates: list[ReviewBandCandidate] = []
    for width in widths:
        reviewed_n = (
            0 if width == 0 else int(np.searchsorted(sorted_distances, width, side="right"))
        )
        automated_n = len(truth) - reviewed_n
        reviewed_weight = 0.0 if reviewed_n == 0 else float(cumulative_weight[reviewed_n - 1])
        reviewed_error_weight = (
            0.0 if reviewed_n == 0 else float(cumulative_error_weight[reviewed_n - 1])
        )
        automated_weight = max(0.0, total_weight - reviewed_weight)
        automated_error_weight = max(0.0, total_error_weight - reviewed_error_weight)
        coverage = automated_weight / total_weight
        automated_error_rate = (
            automated_error_weight / automated_weight if automated_weight > 0 else None
        )

        reason: str | None = None
        if automated_n < min_automated_samples:
            reason = "too_few_automated_rows"
        elif automated_weight <= 0:
            reason = "zero_automated_weight"
        elif automated_error_rate is None or automated_error_rate > error_limit + 1e-12:
            reason = "error_constraint_exceeded"
        feasible = reason is None
        candidates.append(
            ReviewBandCandidate(
                half_width=width,
                lower_threshold=max(0.0, base - width),
                upper_threshold=min(1.0, base + width),
                automated_n=automated_n,
                review_n=reviewed_n,
                automated_weight=float(automated_weight),
                review_weight=float(reviewed_weight),
                automation_coverage=float(coverage),
                automated_error_rate=(
                    float(automated_error_rate) if automated_error_rate is not None else None
                ),
                feasible=feasible,
                rejection_reason=reason,
            )
        )

    feasible_candidates = [candidate for candidate in candidates if candidate.feasible]
    if not feasible_candidates:
        estimable_errors = [
            candidate.automated_error_rate
            for candidate in candidates
            if candidate.automated_error_rate is not None
        ]
        best_error = min(estimable_errors) if estimable_errors else None
        suffix = f"; best candidate error was {best_error:.6f}" if best_error is not None else ""
        raise ValueError(
            "No review-band candidate satisfies the automated error constraint and minimum "
            f"sample requirement{suffix}"
        )
    chosen = min(
        feasible_candidates,
        key=lambda candidate: (
            -candidate.automation_coverage,
            candidate.automated_error_rate
            if candidate.automated_error_rate is not None
            else float("inf"),
            candidate.half_width,
        ),
    )
    if chosen.automated_error_rate is None:
        raise RuntimeError("Selected candidate unexpectedly has no automated error estimate")

    return ReviewBandPolicy(
        base_threshold=base,
        half_width=chosen.half_width,
        lower_threshold=chosen.lower_threshold,
        upper_threshold=chosen.upper_threshold,
        max_automated_error_rate=error_limit,
        min_automated_samples=min_automated_samples,
        validation_n=len(truth),
        validation_automated_n=chosen.automated_n,
        validation_review_n=chosen.review_n,
        validation_automated_weight=chosen.automated_weight,
        validation_review_weight=chosen.review_weight,
        validation_automation_coverage=chosen.automation_coverage,
        validation_automated_error_rate=chosen.automated_error_rate,
        weighted_selection=weighted,
        candidates=tuple(candidates),
    )


def apply_review_band(
    y_proba: ArrayLike,
    policy: ReviewBandPolicy,
) -> NDArray[np.str_]:
    """Apply a frozen global policy without labels or protected attributes."""
    probabilities = _probabilities(y_proba)
    decisions = np.where(probabilities >= policy.base_threshold, AUTO_POSITIVE, AUTO_NEGATIVE)
    if policy.half_width > 0:
        review_mask = np.abs(probabilities - policy.base_threshold) <= policy.half_width
        decisions = decisions.astype("<U13")
        decisions[review_mask] = REVIEW
    return np.asarray(decisions, dtype=str)


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


def _evaluation_groups(
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
            values = _one_dimensional(groups[name], f"groups[{name!r}]")
            if len(values) != length:
                raise ValueError("Every group dimension must have the same length as y_true")
            columns.append([_json_scalar(value, f"groups[{name!r}]") for value in values])
        return dimensions, list(zip(*columns, strict=True))

    values = _one_dimensional(groups, "groups")
    if len(values) != length:
        raise ValueError("groups must have the same length as y_true")
    return ["group"], [(_json_scalar(value, "groups"),) for value in values]


def _group_id(dimensions: Sequence[str], values: Sequence[str | int | float | bool]) -> str:
    return " | ".join(
        f"{dimension}={json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"
        for dimension, value in zip(dimensions, values, strict=True)
    )


def _range_gap(
    rows: Sequence[dict[str, Any]],
    metric_name: str,
) -> dict[str, Any]:
    values: list[tuple[str, float]] = []
    for row in rows:
        metric = row[metric_name]
        if metric["evidence_status"] != "sufficient" or metric["estimate"] is None:
            continue
        values.append((str(row["group_id"]), float(metric["estimate"])))
    if len(values) < 2:
        return {
            "minimum": None,
            "maximum": None,
            "absolute_gap": None,
            "minimum_group": None,
            "maximum_group": None,
            "eligible_group_count": len(values),
            "evidence_status": "insufficient_groups",
        }
    minimum_group, minimum = min(values, key=lambda item: (item[1], item[0]))
    maximum_group, maximum = max(values, key=lambda item: (item[1], item[0]))
    return {
        "minimum": minimum,
        "maximum": maximum,
        "absolute_gap": maximum - minimum,
        "minimum_group": minimum_group,
        "maximum_group": maximum_group,
        "eligible_group_count": len(values),
        "evidence_status": "sufficient",
    }


def _rate(
    numerator: NDArray[np.bool_],
    denominator: NDArray[np.bool_],
    weights: NDArray[np.float64],
    *,
    evidence_reasons: list[str],
) -> dict[str, Any]:
    denominator_weight = float(weights[denominator].sum())
    if int(denominator.sum()) == 0 or denominator_weight <= 0:
        return {
            "estimate": None,
            "numerator_n": int((numerator & denominator).sum()),
            "denominator_n": int(denominator.sum()),
            "evidence_status": "not_estimable",
            "evidence_reasons": sorted({*evidence_reasons, "zero_metric_denominator"}),
        }
    estimate = float(weights[numerator & denominator].sum()) / denominator_weight
    return {
        "estimate": estimate,
        "numerator_n": int((numerator & denominator).sum()),
        "denominator_n": int(denominator.sum()),
        "evidence_status": "limited" if evidence_reasons else "sufficient",
        "evidence_reasons": sorted(set(evidence_reasons)),
    }


def evaluate_review_band(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    policy: ReviewBandPolicy,
    *,
    groups: Mapping[str, ArrayLike] | ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    min_group_support: int = 10,
    min_automated_group_samples: int = 5,
) -> dict[str, Any]:
    """Evaluate a frozen review band on held-out labels and optional groups.

    Group values are used after decisions are frozen and only for retrospective
    coverage and error diagnostics. The function never retunes ``policy``.
    """
    if (
        not isinstance(min_group_support, int)
        or isinstance(min_group_support, bool)
        or min_group_support < 1
    ):
        raise ValueError("min_group_support must be a positive integer")
    if (
        not isinstance(min_automated_group_samples, int)
        or isinstance(min_automated_group_samples, bool)
        or min_automated_group_samples < 1
    ):
        raise ValueError("min_automated_group_samples must be a positive integer")

    truth = _binary(y_true, "y_true")
    probabilities = _probabilities(y_proba)
    if len(truth) != len(probabilities):
        raise ValueError("y_true and y_proba must have equal lengths")
    weights, weighted = _weights(sample_weight, len(truth))
    decisions = apply_review_band(probabilities, policy)
    automated = decisions != REVIEW
    reviewed = ~automated
    predictions = (probabilities >= policy.base_threshold).astype(int)
    errors = predictions != truth

    total_weight = float(weights.sum())
    automated_weight = float(weights[automated].sum())
    review_weight = float(weights[reviewed].sum())
    automated_error_rate = (
        float(weights[automated & errors].sum()) / automated_weight
        if automated_weight > 0
        else None
    )
    overall = {
        "n": len(truth),
        "weight": total_weight,
        "automated_n": int(automated.sum()),
        "review_n": int(reviewed.sum()),
        "automated_weight": automated_weight,
        "review_weight": review_weight,
        "automation_coverage": automated_weight / total_weight,
        "review_rate": review_weight / total_weight,
        "automated_error_rate": automated_error_rate,
        "automated_accuracy": (
            1 - automated_error_rate if automated_error_rate is not None else None
        ),
        "auto_positive_n": int((decisions == AUTO_POSITIVE).sum()),
        "auto_negative_n": int((decisions == AUTO_NEGATIVE).sum()),
        "constraint_met_on_held_out": (
            automated_error_rate is not None
            and automated_error_rate <= policy.max_automated_error_rate + 1e-12
        ),
    }

    dimensions: list[str] = []
    rows: list[dict[str, Any]] = []
    if groups is not None:
        dimensions, group_values = _evaluation_groups(groups, len(truth))
        cell_indices: dict[str, list[int]] = {}
        cell_values: dict[str, tuple[str | int | float | bool, ...]] = {}
        for index, cell in enumerate(group_values):
            group_id = _group_id(dimensions, cell)
            cell_indices.setdefault(group_id, []).append(index)
            cell_values[group_id] = cell
        for group_id in sorted(cell_indices):
            cell = cell_values[group_id]
            indices = np.asarray(cell_indices[group_id], dtype=int)
            cell_automated = automated[indices]
            cell_reviewed = reviewed[indices]
            cell_errors = errors[indices]
            cell_weights = weights[indices]
            reasons = ["support_below_minimum"] if len(indices) < min_group_support else []
            automated_reasons = list(reasons)
            if int(cell_automated.sum()) < min_automated_group_samples:
                automated_reasons.append("automated_rows_below_minimum")
            all_rows = np.ones(len(indices), dtype=bool)
            rows.append(
                {
                    "group_id": group_id,
                    "attributes": dict(zip(dimensions, cell, strict=True)),
                    "n": len(indices),
                    "weight": float(cell_weights.sum()),
                    "automated_n": int(cell_automated.sum()),
                    "review_n": int(cell_reviewed.sum()),
                    "automation_coverage": _rate(
                        cell_automated,
                        all_rows,
                        cell_weights,
                        evidence_reasons=reasons,
                    ),
                    "review_rate": _rate(
                        cell_reviewed,
                        all_rows,
                        cell_weights,
                        evidence_reasons=reasons,
                    ),
                    "automated_error_rate": _rate(
                        cell_errors,
                        cell_automated,
                        cell_weights,
                        evidence_reasons=automated_reasons,
                    ),
                }
            )

    return {
        "schema_version": "1.0",
        "policy": policy.to_dict(include_candidates=False),
        "evaluation_scope": "held_out_fixed_policy",
        "decision_inputs": ["model_probability"],
        "group_use": "retrospective_evaluation_only",
        "weighted": weighted,
        "overall": overall,
        "group_dimensions": dimensions,
        "groups": rows,
        "group_gaps": {
            "automation_coverage": _range_gap(rows, "automation_coverage")
            if rows
            else {"evidence_status": "not_requested"},
            "review_rate": _range_gap(rows, "review_rate")
            if rows
            else {"evidence_status": "not_requested"},
            "automated_error_rate": _range_gap(rows, "automated_error_rate")
            if rows
            else {"evidence_status": "not_requested"},
        },
    }
