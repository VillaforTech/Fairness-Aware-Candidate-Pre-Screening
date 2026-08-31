"""Tests for the validation-only offline policy frontier."""

from __future__ import annotations

import numpy as np
import pytest

from fairness_project.fairness.frontier import optimize_validation_policy_frontier


def test_optimizer_selects_accurate_feasible_pair_and_computes_metrics() -> None:
    result = optimize_validation_policy_frontier(
        y_true=np.array([1, 1, 0, 0, 1, 1, 0, 0]),
        probabilities=np.array([0.9, 0.8, 0.7, 0.1, 0.9, 0.4, 0.3, 0.2]),
        groups=np.array(["P"] * 4 + ["U"] * 4),
        privileged_value="P",
        unprivileged_value="U",
        threshold_grid=[0.4, 0.5, 0.8],
        max_abs_tpr_gap=0.0,
        max_accuracy_loss=0.0,
    )

    assert result.status == "feasible"
    assert result.scope == "validation_only_offline_policy"
    assert result.evaluated_candidates == 9
    assert result.selected is not None
    assert result.selected.threshold_privileged == pytest.approx(0.8)
    assert result.selected.threshold_unprivileged == pytest.approx(0.4)
    assert result.selected.accuracy == pytest.approx(1.0)
    assert result.selected.accuracy_loss_vs_global == pytest.approx(-0.25)
    assert result.selected.selection_rate_gap == pytest.approx(0.0)
    assert result.selected.disparate_impact == pytest.approx(1.0)
    assert result.selected.tpr_gap == pytest.approx(0.0)
    assert result.selected.fpr_gap == pytest.approx(0.0)

    assert result.baseline.threshold_privileged == pytest.approx(0.5)
    assert result.baseline.threshold_unprivileged == pytest.approx(0.5)
    assert result.baseline.accuracy == pytest.approx(0.75)
    assert result.baseline.accuracy_loss_vs_global == pytest.approx(0.0)


def test_frontier_is_nondominated_and_ordered_by_absolute_tpr_gap() -> None:
    result = optimize_validation_policy_frontier(
        y_true=np.array([1, 1, 0, 0, 1, 1, 0, 0]),
        probabilities=np.array([0.95, 0.65, 0.55, 0.15, 0.85, 0.45, 0.35, 0.05]),
        groups=np.array(["P"] * 4 + ["U"] * 4),
        privileged_value="P",
        unprivileged_value="U",
        threshold_grid=[0.2, 0.4, 0.5, 0.6, 0.9],
        max_abs_tpr_gap=1.0,
        max_accuracy_loss=1.0,
    )

    gaps = [candidate.abs_tpr_gap for candidate in result.frontier]
    accuracies = [candidate.accuracy for candidate in result.frontier]
    assert gaps == sorted(gaps)
    assert all(left < right for left, right in zip(accuracies, accuracies[1:], strict=False))

    for index, candidate in enumerate(result.frontier):
        for other_index, other in enumerate(result.frontier):
            if index == other_index:
                continue
            dominates = (
                other.accuracy >= candidate.accuracy
                and other.abs_tpr_gap <= candidate.abs_tpr_gap
                and (
                    other.accuracy > candidate.accuracy or other.abs_tpr_gap < candidate.abs_tpr_gap
                )
            )
            assert not dominates


def test_ties_prefer_the_global_threshold_pair() -> None:
    result = optimize_validation_policy_frontier(
        y_true=np.array([1, 0, 1, 0]),
        probabilities=np.array([0.9, 0.1, 0.9, 0.1]),
        groups=np.array(["P", "P", "U", "U"]),
        privileged_value="P",
        unprivileged_value="U",
        threshold_grid=[0.6, 0.4, 0.5, 0.5],
        max_abs_tpr_gap=0.0,
        max_accuracy_loss=0.0,
    )

    assert result.threshold_grid == (0.4, 0.5, 0.6)
    assert result.selected is not None
    assert result.selected.threshold_privileged == pytest.approx(0.5)
    assert result.selected.threshold_unprivileged == pytest.approx(0.5)
    assert len(result.frontier) == 1
    assert result.frontier[0] == result.selected


def test_infeasible_constraints_return_explicit_status() -> None:
    result = optimize_validation_policy_frontier(
        y_true=np.array([1, 0, 1, 0]),
        probabilities=np.array([0.9, 0.8, 0.4, 0.1]),
        groups=np.array(["P", "P", "U", "U"]),
        privileged_value="P",
        unprivileged_value="U",
        threshold_grid=[0.5],
        max_abs_tpr_gap=0.0,
        max_accuracy_loss=1.0,
    )

    assert result.status == "infeasible"
    assert result.selected is None
    assert result.feasible_candidates == 0
    assert result.evaluated_candidates == 1


def test_disparate_impact_is_explicitly_undefined_for_zero_privileged_rate() -> None:
    result = optimize_validation_policy_frontier(
        y_true=np.array([1, 0, 1, 0]),
        probabilities=np.array([0.9, 0.1, 0.8, 0.2]),
        groups=np.array(["P", "P", "U", "U"]),
        privileged_value="P",
        unprivileged_value="U",
        threshold_grid=[1.0],
        max_abs_tpr_gap=0.0,
        max_accuracy_loss=1.0,
    )

    assert result.selected is not None
    assert result.selected.selection_rate_privileged == 0.0
    assert result.selected.disparate_impact is None


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"y_true": [[1, 0], [1, 0]]}, "y_true must be a one-dimensional"),
        ({"y_true": []}, "y_true must not be empty"),
        ({"y_true": [1, 2, 0, 0]}, "binary labels"),
        ({"probabilities": [0.9, np.nan, 0.8, 0.2]}, "finite"),
        ({"probabilities": [0.9, -0.1, 0.8, 0.2]}, "between 0 and 1"),
        ({"probabilities": [0.9, 0.1]}, "equal lengths"),
        ({"groups": ["P", "P", "U", "X"]}, "unexpected value"),
        ({"groups": ["P", "P", "P", "P"]}, "unprivileged group.*absent"),
        ({"privileged_value": "P", "unprivileged_value": "P"}, "must be different"),
        ({"max_abs_tpr_gap": 1.1}, "max_abs_tpr_gap"),
        ({"max_accuracy_loss": -0.1}, "max_accuracy_loss"),
        ({"global_threshold": np.inf}, "global_threshold"),
        ({"threshold_grid": []}, "threshold_grid must not be empty"),
        ({"threshold_grid": [0.4, np.nan]}, "threshold_grid.*finite"),
        ({"threshold_grid": [0.4, 1.1]}, "threshold_grid values"),
    ],
)
def test_optimizer_rejects_invalid_inputs(override, message) -> None:
    arguments = {
        "y_true": [1, 0, 1, 0],
        "probabilities": [0.9, 0.1, 0.8, 0.2],
        "groups": ["P", "P", "U", "U"],
        "privileged_value": "P",
        "unprivileged_value": "U",
        "threshold_grid": [0.5],
    }
    arguments.update(override)

    with pytest.raises(ValueError, match=message):
        optimize_validation_policy_frontier(**arguments)


@pytest.mark.parametrize(
    ("y_true", "message"),
    [
        ([1, 0, 0, 0], "unprivileged group.*no positive"),
        ([1, 0, 1, 1], "unprivileged group.*no negative"),
    ],
)
def test_optimizer_requires_estimable_tpr_and_fpr_per_group(y_true, message) -> None:
    with pytest.raises(ValueError, match=message):
        optimize_validation_policy_frontier(
            y_true=y_true,
            probabilities=[0.9, 0.1, 0.8, 0.2],
            groups=["P", "P", "U", "U"],
            privileged_value="P",
            unprivileged_value="U",
            threshold_grid=[0.5],
        )


def test_generated_grid_validation() -> None:
    common = {
        "y_true": [1, 0, 1, 0],
        "probabilities": [0.9, 0.1, 0.8, 0.2],
        "groups": ["P", "P", "U", "U"],
        "privileged_value": "P",
        "unprivileged_value": "U",
    }
    with pytest.raises(ValueError, match="grid_size"):
        optimize_validation_policy_frontier(**common, grid_size=1)
    with pytest.raises(ValueError, match="exactly two"):
        optimize_validation_policy_frontier(**common, threshold_range=(0.5,))
    with pytest.raises(ValueError, match="smaller"):
        optimize_validation_policy_frontier(**common, threshold_range=(0.5, 0.5))
