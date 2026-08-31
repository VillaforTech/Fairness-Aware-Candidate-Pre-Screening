"""Validation-only Pareto search for offline group-threshold policies.

This module is deliberately separated from the serving package. The optimizer needs
ground-truth labels and protected-group values, so it may only be used to study an
offline policy on a validation split. Its output must not be applied by the prediction
API, which intentionally has no protected attributes in its request contract.

The Pareto frontier trades predictive utility (accuracy, maximized) against the
absolute true-positive-rate gap (minimized). Threshold pairs with identical objective
values are collapsed to one deterministic representative so that repeated prediction
operating points do not inflate the returned frontier.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

_BOUNDARY_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class OfflinePolicyConstraints:
    """Validation constraints used to select one offline threshold policy."""

    max_abs_tpr_gap: float
    max_accuracy_loss: float


@dataclass(frozen=True, slots=True)
class OfflinePolicyCandidate:
    """Metrics for one privileged/unprivileged threshold pair.

    Gap signs are always ``privileged - unprivileged``. Disparate impact is the
    unprivileged selection rate divided by the privileged selection rate and is
    ``None`` when the privileged selection rate is zero.
    """

    threshold_privileged: float
    threshold_unprivileged: float
    accuracy: float
    accuracy_loss_vs_global: float
    selection_rate_privileged: float
    selection_rate_unprivileged: float
    selection_rate_gap: float
    disparate_impact: float | None
    tpr_privileged: float
    tpr_unprivileged: float
    tpr_gap: float
    fpr_privileged: float
    fpr_unprivileged: float
    fpr_gap: float

    @property
    def abs_tpr_gap(self) -> float:
        """Return the absolute true-positive-rate gap used by the frontier."""

        return abs(self.tpr_gap)

    @property
    def abs_fpr_gap(self) -> float:
        """Return the absolute false-positive-rate gap."""

        return abs(self.fpr_gap)

    @property
    def abs_selection_rate_gap(self) -> float:
        """Return the absolute selection-rate gap."""

        return abs(self.selection_rate_gap)


@dataclass(frozen=True, slots=True)
class OfflinePolicyOptimizationResult:
    """Result of a validation-only, offline threshold-policy search.

    ``status`` is ``"infeasible"`` and ``selected`` is ``None`` when no enumerated
    threshold pair satisfies both constraints. ``frontier`` is ordered from the
    smallest absolute TPR gap toward increasing accuracy.
    """

    status: Literal["feasible", "infeasible"]
    selected: OfflinePolicyCandidate | None
    frontier: tuple[OfflinePolicyCandidate, ...]
    baseline: OfflinePolicyCandidate
    constraints: OfflinePolicyConstraints
    threshold_grid: tuple[float, ...]
    evaluated_candidates: int
    feasible_candidates: int
    scope: Literal["validation_only_offline_policy"] = "validation_only_offline_policy"


@dataclass(frozen=True, slots=True)
class _GroupThresholdStats:
    threshold: float
    correct: int
    selection_rate: float
    tpr: float
    fpr: float


def _as_one_dimensional(values: ArrayLike, name: str) -> NDArray[Any]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _validate_labels(values: ArrayLike) -> NDArray[np.int8]:
    labels = _as_one_dimensional(values, "y_true")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("y_true must contain only binary labels 0 and 1")
    return labels.astype(np.int8, copy=False)


def _validate_probabilities(values: ArrayLike) -> NDArray[np.float64]:
    raw = _as_one_dimensional(values, "probabilities")
    try:
        probabilities = raw.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("probabilities must contain numeric values") from exc
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must contain only finite values")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("probabilities must contain values between 0 and 1")
    return probabilities


def _validate_unit_interval(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite value between 0 and 1") from exc
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a finite value between 0 and 1")
    return number


def _validate_group_value(value: Any, name: str) -> None:
    if np.asarray(value).ndim != 0:
        raise ValueError(f"{name} must be a scalar group value")
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a hashable scalar group value") from exc


def _values_equal(left: Any, right: Any, name: str) -> bool:
    try:
        result = np.asarray(left == right)
        if result.ndim != 0:
            raise ValueError(f"{name} must contain scalar group values")
        return bool(result.item())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain comparable scalar group values") from exc


def _group_mask(groups: NDArray[Any], target: Any) -> NDArray[np.bool_]:
    return np.fromiter(
        (_values_equal(value, target, "groups") for value in groups),
        dtype=bool,
        count=len(groups),
    )


def _validate_groups(
    values: ArrayLike,
    privileged_value: Any,
    unprivileged_value: Any,
) -> tuple[NDArray[Any], NDArray[np.bool_], NDArray[np.bool_]]:
    groups = _as_one_dimensional(values, "groups")
    _validate_group_value(privileged_value, "privileged_value")
    _validate_group_value(unprivileged_value, "unprivileged_value")
    if _values_equal(privileged_value, unprivileged_value, "group values"):
        raise ValueError("privileged_value and unprivileged_value must be different")

    privileged_mask = _group_mask(groups, privileged_value)
    unprivileged_mask = _group_mask(groups, unprivileged_value)
    if not privileged_mask.any():
        raise ValueError(f"privileged group {privileged_value!r} is absent")
    if not unprivileged_mask.any():
        raise ValueError(f"unprivileged group {unprivileged_value!r} is absent")
    unknown_mask = ~(privileged_mask | unprivileged_mask)
    if unknown_mask.any():
        unexpected = groups[np.flatnonzero(unknown_mask)[0]]
        raise ValueError(
            "groups must contain exactly the declared privileged and unprivileged values; "
            f"found unexpected value {unexpected!r}"
        )
    return groups, privileged_mask, unprivileged_mask


def _validate_estimable_group(
    labels: NDArray[np.int8],
    mask: NDArray[np.bool_],
    group_name: str,
) -> None:
    group_labels = labels[mask]
    if not (group_labels == 1).any():
        raise ValueError(f"TPR is undefined for {group_name}: no positive labels")
    if not (group_labels == 0).any():
        raise ValueError(f"FPR is undefined for {group_name}: no negative labels")


def _thresholds(
    threshold_grid: Sequence[float] | NDArray[np.floating[Any]] | None,
    grid_size: int,
    threshold_range: tuple[float, float],
) -> NDArray[np.float64]:
    if threshold_grid is None:
        if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 2:
            raise ValueError("grid_size must be an integer of at least 2")
        try:
            lower_value, upper_value = threshold_range
        except (TypeError, ValueError) as exc:
            raise ValueError("threshold_range must contain exactly two values") from exc
        lower = _validate_unit_interval(lower_value, "threshold_range minimum")
        upper = _validate_unit_interval(upper_value, "threshold_range maximum")
        if lower >= upper:
            raise ValueError("threshold_range minimum must be smaller than its maximum")
        return np.linspace(lower, upper, grid_size, dtype=float)

    try:
        values = np.asarray(threshold_grid, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold_grid must contain numeric values") from exc
    if values.ndim != 1:
        raise ValueError("threshold_grid must be one-dimensional")
    if values.size == 0:
        raise ValueError("threshold_grid must not be empty")
    if not np.isfinite(values).all():
        raise ValueError("threshold_grid must contain only finite values")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("threshold_grid values must be between 0 and 1")
    return cast(NDArray[np.float64], np.unique(values))


def _group_threshold_stats(
    labels: NDArray[np.int8],
    probabilities: NDArray[np.float64],
    thresholds: NDArray[np.float64],
) -> tuple[_GroupThresholdStats, ...]:
    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())
    stats: list[_GroupThresholdStats] = []
    for threshold in thresholds:
        predictions = probabilities >= threshold
        true_positives = int((predictions & (labels == 1)).sum())
        false_positives = int((predictions & (labels == 0)).sum())
        stats.append(
            _GroupThresholdStats(
                threshold=float(threshold),
                correct=true_positives + (negative_count - false_positives),
                selection_rate=float(predictions.mean()),
                tpr=true_positives / positive_count,
                fpr=false_positives / negative_count,
            )
        )
    return tuple(stats)


def _candidate(
    privileged: _GroupThresholdStats,
    unprivileged: _GroupThresholdStats,
    sample_count: int,
    baseline_accuracy: float,
) -> OfflinePolicyCandidate:
    accuracy = (privileged.correct + unprivileged.correct) / sample_count
    if privileged.selection_rate == 0.0:
        disparate_impact = None
    else:
        disparate_impact = unprivileged.selection_rate / privileged.selection_rate
    return OfflinePolicyCandidate(
        threshold_privileged=privileged.threshold,
        threshold_unprivileged=unprivileged.threshold,
        accuracy=accuracy,
        accuracy_loss_vs_global=baseline_accuracy - accuracy,
        selection_rate_privileged=privileged.selection_rate,
        selection_rate_unprivileged=unprivileged.selection_rate,
        selection_rate_gap=privileged.selection_rate - unprivileged.selection_rate,
        disparate_impact=disparate_impact,
        tpr_privileged=privileged.tpr,
        tpr_unprivileged=unprivileged.tpr,
        tpr_gap=privileged.tpr - unprivileged.tpr,
        fpr_privileged=privileged.fpr,
        fpr_unprivileged=unprivileged.fpr,
        fpr_gap=privileged.fpr - unprivileged.fpr,
    )


def _policy_change_key(
    candidate: OfflinePolicyCandidate, global_threshold: float
) -> tuple[float, ...]:
    privileged_shift = abs(candidate.threshold_privileged - global_threshold)
    unprivileged_shift = abs(candidate.threshold_unprivileged - global_threshold)
    return (
        candidate.abs_fpr_gap,
        candidate.abs_selection_rate_gap,
        privileged_shift + unprivileged_shift,
        max(privileged_shift, unprivileged_shift),
        candidate.threshold_privileged,
        candidate.threshold_unprivileged,
    )


def _selection_key(candidate: OfflinePolicyCandidate, global_threshold: float) -> tuple[float, ...]:
    return (
        -candidate.accuracy,
        candidate.abs_tpr_gap,
        *_policy_change_key(candidate, global_threshold),
    )


def _is_feasible(candidate: OfflinePolicyCandidate, constraints: OfflinePolicyConstraints) -> bool:
    return (
        candidate.abs_tpr_gap <= constraints.max_abs_tpr_gap + _BOUNDARY_TOLERANCE
        and candidate.accuracy_loss_vs_global <= constraints.max_accuracy_loss + _BOUNDARY_TOLERANCE
    )


def _pareto_frontier(
    best_candidate_by_abs_tpr_gap: dict[float, OfflinePolicyCandidate],
) -> tuple[OfflinePolicyCandidate, ...]:
    frontier: list[OfflinePolicyCandidate] = []
    best_accuracy = float("-inf")
    for abs_tpr_gap in sorted(best_candidate_by_abs_tpr_gap):
        candidate = best_candidate_by_abs_tpr_gap[abs_tpr_gap]
        if candidate.accuracy > best_accuracy:
            frontier.append(candidate)
            best_accuracy = candidate.accuracy
    return tuple(frontier)


def optimize_validation_policy_frontier(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    groups: ArrayLike,
    *,
    privileged_value: Any,
    unprivileged_value: Any,
    threshold_grid: Sequence[float] | NDArray[np.floating[Any]] | None = None,
    grid_size: int = 101,
    threshold_range: tuple[float, float] = (0.0, 1.0),
    global_threshold: float = 0.5,
    max_abs_tpr_gap: float = 0.05,
    max_accuracy_loss: float = 0.02,
) -> OfflinePolicyOptimizationResult:
    """Enumerate and select a validation-only, offline group-threshold policy.

    The function evaluates every Cartesian pair in ``threshold_grid`` (or a generated
    evenly spaced grid), computes group fairness and utility metrics, and returns the
    nondominated accuracy-versus-absolute-TPR-gap frontier. It selects the most accurate
    pair satisfying both configured constraints. Ties prefer a smaller absolute TPR
    gap, then smaller FPR and selection-rate gaps, less movement from the global
    threshold, and finally lexicographically smaller thresholds.

    This API requires validation labels and protected attributes. Do not call it on a
    held-out test set to tune a policy, and do not integrate it into online inference or
    expose protected attributes in the serving contract. A selected policy remains an
    offline benchmark artifact; it is not evidence that the policy is appropriate for
    employment decisions.

    Parameters
    ----------
    y_true:
        Binary validation labels.
    probabilities:
        Finite positive-class probabilities in the inclusive range [0, 1].
    groups:
        Binary protected-group values aligned with ``y_true``.
    privileged_value / unprivileged_value:
        Explicit scalar values identifying the two groups.
    threshold_grid:
        Optional explicit grid. Values are sorted and duplicates are collapsed. When
        omitted, ``grid_size`` values over ``threshold_range`` are generated.
    global_threshold:
        Shared threshold used to calculate baseline accuracy and candidate accuracy loss.
    max_abs_tpr_gap / max_accuracy_loss:
        Inclusive constraints used for deterministic policy selection.

    Returns
    -------
    OfflinePolicyOptimizationResult
        The canonical Pareto frontier, shared-threshold baseline, and either a selected
        feasible candidate or an explicit infeasible status.
    """

    labels = _validate_labels(y_true)
    scores = _validate_probabilities(probabilities)
    declared_groups, privileged_mask, unprivileged_mask = _validate_groups(
        groups,
        privileged_value,
        unprivileged_value,
    )
    lengths = {
        "y_true": len(labels),
        "probabilities": len(scores),
        "groups": len(declared_groups),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"inputs must have equal lengths, got {lengths}")

    _validate_estimable_group(labels, privileged_mask, "privileged group")
    _validate_estimable_group(labels, unprivileged_mask, "unprivileged group")
    shared_threshold = _validate_unit_interval(global_threshold, "global_threshold")
    constraints = OfflinePolicyConstraints(
        max_abs_tpr_gap=_validate_unit_interval(max_abs_tpr_gap, "max_abs_tpr_gap"),
        max_accuracy_loss=_validate_unit_interval(max_accuracy_loss, "max_accuracy_loss"),
    )
    thresholds = _thresholds(threshold_grid, grid_size, threshold_range)

    privileged_grid_stats = _group_threshold_stats(
        labels[privileged_mask], scores[privileged_mask], thresholds
    )
    unprivileged_grid_stats = _group_threshold_stats(
        labels[unprivileged_mask], scores[unprivileged_mask], thresholds
    )
    privileged_baseline = _group_threshold_stats(
        labels[privileged_mask],
        scores[privileged_mask],
        np.asarray([shared_threshold]),
    )[0]
    unprivileged_baseline = _group_threshold_stats(
        labels[unprivileged_mask],
        scores[unprivileged_mask],
        np.asarray([shared_threshold]),
    )[0]
    baseline_accuracy = (privileged_baseline.correct + unprivileged_baseline.correct) / len(labels)
    baseline = _candidate(
        privileged_baseline,
        unprivileged_baseline,
        len(labels),
        baseline_accuracy,
    )

    selected: OfflinePolicyCandidate | None = None
    feasible_candidates = 0
    best_candidate_by_abs_tpr_gap: dict[float, OfflinePolicyCandidate] = {}
    for privileged_stats in privileged_grid_stats:
        for unprivileged_stats in unprivileged_grid_stats:
            candidate = _candidate(
                privileged_stats,
                unprivileged_stats,
                len(labels),
                baseline_accuracy,
            )
            if _is_feasible(candidate, constraints):
                feasible_candidates += 1
                if selected is None or _selection_key(candidate, shared_threshold) < _selection_key(
                    selected, shared_threshold
                ):
                    selected = candidate

            same_gap = best_candidate_by_abs_tpr_gap.get(candidate.abs_tpr_gap)
            is_more_accurate = same_gap is None or candidate.accuracy > same_gap.accuracy
            is_better_representative = (
                same_gap is not None
                and candidate.accuracy == same_gap.accuracy
                and _policy_change_key(candidate, shared_threshold)
                < _policy_change_key(same_gap, shared_threshold)
            )
            if is_more_accurate or is_better_representative:
                best_candidate_by_abs_tpr_gap[candidate.abs_tpr_gap] = candidate

    return OfflinePolicyOptimizationResult(
        status="feasible" if selected is not None else "infeasible",
        selected=selected,
        frontier=_pareto_frontier(best_candidate_by_abs_tpr_gap),
        baseline=baseline,
        constraints=constraints,
        threshold_grid=tuple(float(value) for value in thresholds),
        evaluated_candidates=len(thresholds) ** 2,
        feasible_candidates=feasible_candidates,
    )


__all__ = [
    "OfflinePolicyCandidate",
    "OfflinePolicyConstraints",
    "OfflinePolicyOptimizationResult",
    "optimize_validation_policy_frontier",
]
