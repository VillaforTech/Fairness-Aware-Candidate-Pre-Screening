"""Tests for validation-only global review-band selection and evaluation."""

import numpy as np
import pytest

from fairness_project.fairness.selective import (
    AUTO_NEGATIVE,
    AUTO_POSITIVE,
    REVIEW,
    apply_review_band,
    evaluate_review_band,
    select_review_band,
)


def _policy_with_review_band():
    return select_review_band(
        y_true_validation=np.array([1, 1, 1, 0, 1, 0, 0, 0]),
        y_proba_validation=np.array([0.95, 0.85, 0.70, 0.55, 0.45, 0.30, 0.15, 0.05]),
        max_automated_error_rate=0.10,
    )


def test_selects_maximum_coverage_candidate_under_validation_error_constraint() -> None:
    policy = _policy_with_review_band()
    assert policy.half_width == pytest.approx(0.05)
    assert policy.validation_automated_n == 6
    assert policy.validation_review_n == 2
    assert policy.validation_automation_coverage == pytest.approx(0.75)
    assert policy.validation_automated_error_rate == pytest.approx(0.0)
    assert all(
        candidate.automation_coverage <= policy.validation_automation_coverage
        for candidate in policy.candidates
        if candidate.feasible
    )


def test_perfect_validation_predictions_choose_no_review() -> None:
    policy = select_review_band(
        y_true_validation=np.array([1, 1, 0, 0]),
        y_proba_validation=np.array([0.9, 0.8, 0.2, 0.1]),
        max_automated_error_rate=0.0,
    )
    assert policy.half_width == 0
    assert policy.validation_automation_coverage == 1
    assert apply_review_band(np.array([0.5]), policy).tolist() == [AUTO_POSITIVE]


def test_exact_threshold_rows_can_be_reviewed_without_reviewing_confident_rows() -> None:
    policy = select_review_band(
        y_true_validation=np.array([0, 1, 0]),
        y_proba_validation=np.array([0.5, 0.9, 0.1]),
        max_automated_error_rate=0.0,
    )
    assert policy.half_width > 0
    assert policy.validation_review_n == 1
    assert policy.validation_automation_coverage == pytest.approx(2 / 3)
    assert apply_review_band(np.array([0.5, 0.9, 0.1]), policy).tolist() == [
        REVIEW,
        AUTO_POSITIVE,
        AUTO_NEGATIVE,
    ]


def test_apply_review_band_uses_only_probability_and_global_policy() -> None:
    policy = _policy_with_review_band()
    decisions = apply_review_band(np.array([0.9, 0.55, 0.50, 0.45, 0.1]), policy)
    assert decisions.tolist() == [AUTO_POSITIVE, REVIEW, REVIEW, REVIEW, AUTO_NEGATIVE]
    assert policy.to_dict(include_candidates=False)["decision_inputs"] == ["model_probability"]


def test_held_out_evaluation_reports_review_burden_and_group_gaps() -> None:
    policy = _policy_with_review_band()
    result = evaluate_review_band(
        y_true=np.array([0, 1, 0, 1, 0, 0]),
        y_proba=np.array([0.9, 0.52, 0.51, 0.8, 0.2, 0.1]),
        policy=policy,
        groups={
            "segment": np.array(["A", "A", "A", "B", "B", "B"]),
            "region": np.array(["X"] * 6),
        },
        min_group_support=1,
        min_automated_group_samples=1,
    )
    assert result["overall"]["automated_n"] == 4
    assert result["overall"]["review_n"] == 2
    assert result["overall"]["automation_coverage"] == pytest.approx(4 / 6)
    assert result["group_gaps"]["automation_coverage"]["absolute_gap"] == pytest.approx(2 / 3)
    assert result["group_gaps"]["automated_error_rate"]["absolute_gap"] == pytest.approx(1.0)
    assert result["group_use"] == "retrospective_evaluation_only"
    assert result["decision_inputs"] == ["model_probability"]


def test_held_out_constraint_is_reported_not_reoptimized() -> None:
    policy = select_review_band(
        y_true_validation=np.array([1, 0]),
        y_proba_validation=np.array([0.9, 0.1]),
        max_automated_error_rate=0.0,
    )
    result = evaluate_review_band(
        y_true=np.array([0, 1]),
        y_proba=np.array([0.9, 0.1]),
        policy=policy,
    )
    assert result["policy"]["half_width"] == 0
    assert result["overall"]["automated_error_rate"] == 1
    assert result["overall"]["constraint_met_on_held_out"] is False
    assert result["evaluation_scope"] == "held_out_fixed_policy"


def test_weighted_selection_records_weighted_coverage() -> None:
    policy = select_review_band(
        y_true_validation=np.array([1, 0, 1]),
        y_proba_validation=np.array([0.9, 0.55, 0.9]),
        sample_weight=np.array([1.0, 9.0, 1.0]),
        max_automated_error_rate=0.10,
    )
    assert policy.weighted_selection is True
    assert policy.validation_review_n == 1
    assert policy.validation_automation_coverage == pytest.approx(2 / 11)


def test_no_feasible_policy_raises_instead_of_silently_reviewing_everything() -> None:
    with pytest.raises(ValueError, match="No review-band candidate"):
        select_review_band(
            y_true_validation=np.array([0]),
            y_proba_validation=np.array([0.9]),
            max_automated_error_rate=0.0,
        )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {"y_true_validation": np.array([0, 2]), "y_proba_validation": np.array([0.1, 0.9])},
            "binary values",
        ),
        (
            {"y_true_validation": np.array([0, 1]), "y_proba_validation": np.array([0.1])},
            "equal lengths",
        ),
        (
            {"y_true_validation": np.array([0, 1]), "y_proba_validation": np.array([0.1, np.nan])},
            "finite",
        ),
        (
            {
                "y_true_validation": np.array([0, 1]),
                "y_proba_validation": np.array([0.1, 0.9]),
                "sample_weight": np.array([1.0, -1.0]),
            },
            "nonnegative",
        ),
        (
            {
                "y_true_validation": np.array([0, 1]),
                "y_proba_validation": np.array([0.1, 0.9]),
                "base_threshold": 1.1,
            },
            "base_threshold",
        ),
        (
            {
                "y_true_validation": np.array([0, 1]),
                "y_proba_validation": np.array([0.1, 0.9]),
                "min_automated_samples": 0,
            },
            "positive integer",
        ),
    ],
)
def test_select_review_band_rejects_invalid_inputs(arguments, message) -> None:
    with pytest.raises(ValueError, match=message):
        select_review_band(**arguments)


def test_custom_widths_must_respect_configured_maximum() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        select_review_band(
            y_true_validation=np.array([1, 0]),
            y_proba_validation=np.array([0.9, 0.1]),
            max_half_width=0.1,
            candidate_half_widths=[0.2],
        )


def test_evaluation_rejects_invalid_group_evidence_threshold() -> None:
    policy = select_review_band(
        y_true_validation=np.array([1, 0]),
        y_proba_validation=np.array([0.9, 0.1]),
    )
    with pytest.raises(ValueError, match="min_group_support"):
        evaluate_review_band(
            np.array([1, 0]),
            np.array([0.9, 0.1]),
            policy,
            min_group_support=0,
        )
