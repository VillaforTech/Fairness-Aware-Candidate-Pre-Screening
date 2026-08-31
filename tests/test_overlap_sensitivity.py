"""Tests for exact predictive-feature overlap sensitivity."""

import json

import numpy as np
import pandas as pd
import pytest

from fairness_project.data.schema import FEATURE_COLUMNS
from fairness_project.evaluation.overlap import (
    OverlapSensitivityValidationError,
    exact_feature_overlap_mask,
    exact_feature_overlap_sensitivity,
)


def _feature_row(**updates):
    row = {
        "age": 30,
        "workclass": "Private",
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Tech-support",
        "relationship": "Not-in-family",
        "native_country": "United-States",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
    }
    row.update(updates)
    return row


def _inputs():
    reference = pd.DataFrame(
        [
            _feature_row(age=30),
            _feature_row(age=31, occupation="Sales"),
            _feature_row(age=32, education="Masters", education_num=14),
        ]
    )
    heldout = pd.DataFrame(
        [
            _feature_row(age=31, occupation="Sales"),
            _feature_row(age=40, workclass="Self-emp-not-inc"),
            _feature_row(age=41, occupation="Exec-managerial"),
            _feature_row(age=30),
        ]
    )
    return {
        "reference_rows": reference,
        "heldout_rows": heldout,
        "y_true": np.array([1, 0, 1, 0]),
        "baseline_predictions": np.array([1, 0, 0, 1]),
        "adjusted_predictions": np.array([1, 0, 1, 0]),
        "probabilities": np.array([0.8, 0.2, 0.4, 0.7]),
        "sensitive": np.array(["Male", "Female", "Male", "Female"]),
        "privileged_group": "Male",
    }


def test_overlap_uses_exact_canonical_identity_and_reports_both_slices() -> None:
    result = exact_feature_overlap_sensitivity(**_inputs())

    assert result["audit_type"] == "exact_feature_overlap_sensitivity"
    assert result["identity"]["columns"] == FEATURE_COLUMNS
    assert result["identity"]["hash_used_for_final_equality"] is False
    assert result["policy"]["retuned"] is False
    assert result["counts"] == {
        "reference_rows": 3,
        "reference_unique_feature_identities": 3,
        "reference_duplicate_rows_beyond_first": 0,
        "held_out_rows": 4,
        "overlap_rows": 2,
        "novel_rows": 2,
        "overlap_rate": 0.5,
    }
    assert result["overlap_positions"] == [0, 3]

    all_rows = result["slices"]["all_held_out"]
    novel = result["slices"]["overlap_excluded"]
    assert all_rows["baseline"]["metrics"]["accuracy"] == pytest.approx(0.5)
    assert all_rows["adjusted"]["metrics"]["accuracy"] == pytest.approx(1.0)
    assert novel["row_count"] == 2
    assert novel["baseline"]["metrics"]["accuracy"] == pytest.approx(0.5)
    assert novel["adjusted"]["metrics"]["accuracy"] == pytest.approx(1.0)
    assert novel["evidence_status"] == "limited"
    assert "zero_positive_denominator_unprivileged_group" in novel["evidence_reasons"]
    json.dumps(result, sort_keys=True, allow_nan=False)


def test_public_overlap_mask_is_exact_and_boolean() -> None:
    arguments = _inputs()

    mask = exact_feature_overlap_mask(
        reference_rows=arguments["reference_rows"],
        compared_rows=arguments["heldout_rows"],
    )

    assert mask.dtype == np.bool_
    assert mask.tolist() == [True, False, False, True]


def test_duplicate_reference_rows_count_once_for_membership() -> None:
    arguments = _inputs()
    duplicate = arguments["reference_rows"].iloc[[1]].copy()
    arguments["reference_rows"] = pd.concat(
        [arguments["reference_rows"], duplicate, duplicate],
        ignore_index=True,
    )

    result = exact_feature_overlap_sensitivity(**arguments)

    assert result["counts"]["reference_rows"] == 5
    assert result["counts"]["reference_unique_feature_identities"] == 3
    assert result["counts"]["reference_duplicate_rows_beyond_first"] == 2
    assert result["overlap_positions"] == [0, 3]


def test_non_feature_labels_and_conflicts_do_not_affect_identity() -> None:
    arguments = _inputs()
    arguments["reference_rows"] = arguments["reference_rows"].assign(
        income=["<=50K", "<=50K", ">50K"],
        split="train",
    )
    arguments["heldout_rows"] = arguments["heldout_rows"].assign(
        income=[">50K", "<=50K", ">50K", ">50K"],
        split="test",
    )

    result = exact_feature_overlap_sensitivity(**arguments)

    assert result["identity"]["non_feature_columns_ignored"] is True
    assert result["overlap_positions"] == [0, 3]


def test_no_overlap_keeps_the_same_all_and_novel_metrics() -> None:
    arguments = _inputs()
    arguments["reference_rows"] = arguments["reference_rows"].assign(
        age=lambda frame: frame["age"] + 100
    )

    result = exact_feature_overlap_sensitivity(**arguments)

    assert result["counts"]["overlap_rows"] == 0
    assert result["counts"]["overlap_rate"] == 0.0
    assert result["overlap_positions"] == []
    assert (
        result["slices"]["all_held_out"]["baseline"]["metrics"]
        == result["slices"]["overlap_excluded"]["baseline"]["metrics"]
    )
    assert (
        result["slices"]["all_held_out"]["adjusted"]["metrics"]
        == result["slices"]["overlap_excluded"]["adjusted"]["metrics"]
    )


def test_all_overlap_makes_novel_slice_explicitly_not_estimable() -> None:
    arguments = _inputs()
    arguments["reference_rows"] = arguments["heldout_rows"].copy(deep=True)

    result = exact_feature_overlap_sensitivity(**arguments)

    novel = result["slices"]["overlap_excluded"]
    assert result["counts"]["overlap_rows"] == 4
    assert result["counts"]["novel_rows"] == 0
    assert result["counts"]["overlap_rate"] == 1.0
    assert result["overlap_positions"] == [0, 1, 2, 3]
    assert novel["row_count"] == 0
    assert novel["evidence_status"] == "not_estimable"
    assert novel["evidence_reasons"] == ["no_overlap_excluded_rows"]
    assert novel["baseline"]["metrics"] is None
    assert novel["adjusted"]["metrics"] is None
    assert (
        novel["baseline"]["fairness_evidence"]["metrics"]["TPR_gap"]["evidence_status"]
        == "not_estimable"
    )
    json.dumps(result, allow_nan=False)


def test_missing_group_and_denominators_have_explicit_evidence() -> None:
    arguments = _inputs()
    arguments["sensitive"] = np.array(["Male"] * 4)

    group_result = exact_feature_overlap_sensitivity(**arguments)
    all_group = group_result["slices"]["all_held_out"]
    assert all_group["evidence_status"] == "limited"
    assert all_group["baseline"]["fairness_evidence"]["evidence_status"] == "not_estimable"
    assert "unprivileged_group_absent" in all_group["evidence_reasons"]
    assert all_group["baseline"]["metrics"]["SPD"] is None
    assert all_group["baseline"]["metrics"]["accuracy"] == pytest.approx(0.5)

    arguments = _inputs()
    arguments["y_true"] = np.array([1, 1, 1, 1])
    denominator_result = exact_feature_overlap_sensitivity(**arguments)
    fairness = denominator_result["slices"]["all_held_out"]["baseline"]["fairness_evidence"]
    assert fairness["evidence_status"] == "partial"
    assert fairness["metrics"]["FPR_gap"]["evidence_status"] == "not_estimable"
    assert (
        "zero_negative_denominator_privileged_group"
        in fairness["metrics"]["FPR_gap"]["evidence_reasons"]
    )
    assert denominator_result["slices"]["all_held_out"]["baseline"]["metrics"]["FPR_gap"] is None
    json.dumps(denominator_result, allow_nan=False)


def test_inputs_are_not_mutated_and_output_is_deterministic() -> None:
    arguments = _inputs()
    reference_before = arguments["reference_rows"].copy(deep=True)
    heldout_before = arguments["heldout_rows"].copy(deep=True)
    array_before = {
        key: value.copy() for key, value in arguments.items() if isinstance(value, np.ndarray)
    }

    first = exact_feature_overlap_sensitivity(**arguments)
    second = exact_feature_overlap_sensitivity(**arguments)

    assert first == second
    pd.testing.assert_frame_equal(arguments["reference_rows"], reference_before)
    pd.testing.assert_frame_equal(arguments["heldout_rows"], heldout_before)
    for key, before in array_before.items():
        np.testing.assert_array_equal(arguments[key], before)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"reference_rows": "rows"}, "pandas DataFrame"),
        ({"reference_rows": pd.DataFrame()}, "at least one row"),
        (
            {"heldout_rows": pd.DataFrame([_feature_row()]).drop(columns=["age"])},
            "missing canonical feature columns",
        ),
        (
            {
                "heldout_rows": pd.DataFrame([_feature_row()]).assign(age=30.5),
                "y_true": np.array([0]),
                "baseline_predictions": np.array([0]),
                "adjusted_predictions": np.array([0]),
                "probabilities": np.array([0.2]),
                "sensitive": np.array(["Female"]),
            },
            "integer dtype",
        ),
        (
            {
                "heldout_rows": pd.DataFrame([_feature_row()]).assign(workclass=" "),
                "y_true": np.array([0]),
                "baseline_predictions": np.array([0]),
                "adjusted_predictions": np.array([0]),
                "probabilities": np.array([0.2]),
                "sensitive": np.array(["Female"]),
            },
            "non-empty strings",
        ),
        ({"y_true": np.array([0, 1])}, "same length"),
        ({"baseline_predictions": np.array([0, 1, 2, 0])}, "binary values"),
        ({"adjusted_predictions": np.array([0, 1, 0])}, "same length"),
        ({"probabilities": np.array([0.2, np.nan, 0.8, 0.1])}, "finite"),
        ({"probabilities": np.array([0.2, 1.1, 0.8, 0.1])}, "between 0 and 1"),
        ({"sensitive": np.array(["Male", None, "Male", "Female"])}, "missing"),
        ({"privileged_group": np.nan}, "missing"),
        ({"privileged_group": 1}, "consistent scalar type family"),
    ],
)
def test_invalid_inputs_are_rejected(update, message) -> None:
    arguments = _inputs()
    arguments.update(update)
    with pytest.raises(OverlapSensitivityValidationError, match=message):
        exact_feature_overlap_sensitivity(**arguments)


def test_duplicate_column_names_are_rejected() -> None:
    arguments = _inputs()
    heldout = arguments["heldout_rows"].copy()
    heldout.columns = [*heldout.columns[:-1], heldout.columns[-2]]
    arguments["heldout_rows"] = heldout

    with pytest.raises(OverlapSensitivityValidationError, match="duplicate column names"):
        exact_feature_overlap_sensitivity(**arguments)
