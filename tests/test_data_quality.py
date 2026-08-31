"""Tests for deterministic Adult data-semantics audits."""

import hashlib
import json

import pandas as pd
import pytest

from fairness_project.data.quality import (
    DataQualityValidationError,
    audit_processed_quality,
    audit_raw_attrition,
)
from fairness_project.data.schema import FEATURE_COLUMNS


def _adult_row(**updates):
    row = {
        "age": 20,
        "workclass": "Private",
        "fnlwgt": 100,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Never-married",
        "occupation": "Sales",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
        "income": "<=50K",
    }
    row.update(updates)
    return row


def _raw_frames():
    duplicate = _adult_row()
    train = pd.DataFrame(
        [
            duplicate,
            duplicate.copy(),
            _adult_row(age=21, workclass=" ? ", fnlwgt=110),
            _adult_row(age=30, race="White", sex="Male", fnlwgt=120, income=">50K"),
            _adult_row(age=30, race="White", sex="Male", fnlwgt=121, income="<=50K"),
            _adult_row(age=40, race="Asian-Pac-Islander", occupation="   ", fnlwgt=130),
        ]
    )
    test = pd.DataFrame(
        [
            _adult_row(age=30, race="White", sex="Male", fnlwgt=122, income=">50K."),
            _adult_row(age=50, fnlwgt=140),
            _adult_row(age=60, race="White", sex=" ? ", fnlwgt=150),
        ]
    )
    return train, test


def _find_group(rows, **attributes):
    return next(row for row in rows if row["attributes"] == attributes)


def test_raw_attrition_reports_counts_missingness_composition_and_null_groups() -> None:
    train, test = _raw_frames()
    train_before = train.copy(deep=True)
    test_before = test.copy(deep=True)

    result = audit_raw_attrition(train, test, small_group_threshold=2)

    assert result["schema_version"] == "1.0"
    assert result["audit_type"] == "raw_complete_case_attrition"
    assert result["attrition"]["overall"] == {
        "input_rows": 9,
        "complete_case_rows": 6,
        "deleted_rows": 3,
        "deletion_rate": pytest.approx(1 / 3),
    }
    female = _find_group(result["attrition"]["by_sex"], sex="Female")
    assert female["input_rows"] == 5
    assert female["complete_case_rows"] == 3
    assert female["deletion_rate"] == pytest.approx(0.4)
    missing_sex = _find_group(result["attrition"]["by_sex"], sex=None)
    assert missing_sex["input_rows"] == 1
    assert missing_sex["complete_case_rows"] == 0

    female_asian = _find_group(
        result["group_composition"]["by_sex_and_race"],
        sex="Female",
        race="Asian-Pac-Islander",
    )
    assert female_asian["before"]["count"] == 1
    assert female_asian["after"]["count"] == 0

    overall_missingness = {row["column"]: row for row in result["missingness"]["overall"]}
    assert overall_missingness["workclass"]["missing_count"] == 1
    assert overall_missingness["occupation"]["missing_count"] == 1
    assert overall_missingness["sex"]["missing_count"] == 1
    assert result["configuration"]["protected_value_policy"] == "observe_or_null_never_infer"
    assert any(
        flag["attributes"] == {"sex": None}
        and flag["stage"] == "complete_case"
        and flag["count"] == 0
        for flag in result["evidence_flags"]
    )

    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(test, test_before)


def test_raw_attrition_reports_duplicate_and_cross_split_evidence() -> None:
    train, test = _raw_frames()

    result = audit_raw_attrition(train, test, small_group_threshold=1)
    duplicates = result["duplicates"]

    assert duplicates["exact_rows"]["duplicate_groups"] == 1
    assert duplicates["exact_rows"]["rows_in_duplicate_groups"] == 2
    assert duplicates["exact_rows"]["duplicate_rows_beyond_first"] == 1
    assert duplicates["predictive_feature_rows"]["duplicate_groups"] == 2
    assert duplicates["conflicting_labels"]["conflicting_feature_groups"] == 1
    assert duplicates["conflicting_labels"]["rows_in_conflicting_groups"] == 3

    split_overlap = duplicates["cross_split_predictive_features"]
    assert split_overlap["pairwise_overlap_feature_groups"] == 1
    pair = split_overlap["pairs"][0]
    assert pair["left_split"] == "test"
    assert pair["right_split"] == "train"
    assert pair["overlap_feature_groups"] == 1
    assert pair["left_rows_in_overlap"] == 1
    assert pair["right_rows_in_overlap"] == 2
    assert pair["conflicting_label_feature_groups"] == 1


def test_fnlwgt_is_explicitly_audit_only_and_output_is_deterministic_json() -> None:
    train, test = _raw_frames()

    first = audit_raw_attrition(train, test, small_group_threshold=2)
    second = audit_raw_attrition(train, test, small_group_threshold=2)

    assert first == second
    encoded = json.dumps(first, sort_keys=True, allow_nan=False)
    assert json.loads(encoded) == first
    weight = first["fnlwgt"]
    assert weight["contract_role"] == "audit_only_sampling_weight"
    assert weight["included_in_predictive_features"] is False
    assert "weighted sensitivity" in weight["semantics"]
    assert "fnlwgt" not in first["configuration"]["feature_identity_columns"]


def _processed_frame():
    rows = [
        {
            **_adult_row(),
            "split": "train",
            "race_binary": "Non-White",
        },
        {
            **_adult_row(),
            "split": "train",
            "race_binary": "Non-White",
        },
        {
            **_adult_row(fnlwgt=101, income=">50K"),
            "split": "test",
            "race_binary": "Non-White",
        },
        {
            **_adult_row(age=31, race="White", sex="Male", fnlwgt=150, income=">50K"),
            "split": "val",
            "race_binary": "White",
        },
        {
            **_adult_row(age=40, race="Asian-Pac-Islander", fnlwgt=0),
            "split": "test",
            "race_binary": "Non-White",
        },
    ]
    return pd.DataFrame(rows)


def test_processed_quality_reports_splits_conflicts_and_small_groups() -> None:
    frame = _processed_frame()
    before = frame.copy(deep=True)

    result = audit_processed_quality(frame, small_group_threshold=2)

    assert result["audit_type"] == "processed_data_quality"
    assert result["attrition"]["overall"]["input_rows"] == 5
    assert result["attrition"]["overall"]["deleted_rows"] == 0
    train = _find_group(result["attrition"]["by_split"], split="train")
    assert train["input_rows"] == 2
    assert result["duplicates"]["exact_rows"]["duplicate_groups"] == 1
    assert result["duplicates"]["conflicting_labels"]["conflicting_feature_groups"] == 1

    pair = next(
        row
        for row in result["duplicates"]["cross_split_predictive_features"]["pairs"]
        if {row["left_split"], row["right_split"]} == {"test", "train"}
    )
    assert pair["overlap_feature_groups"] == 1
    assert pair["conflicting_label_feature_groups"] == 1
    assert result["fnlwgt"]["non_positive_count"] == 1
    assert any(
        flag["attributes"] == {"race": "Asian-Pac-Islander"} and flag["count"] == 1
        for flag in result["evidence_flags"]
    )
    assert any(
        flag["scope"] == "split"
        and flag["split"] == "val"
        and flag["attributes"] == {"sex": "Male"}
        and flag["count"] == 1
        for flag in result["evidence_flags"]
    )
    json.dumps(result, allow_nan=False)
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ("not-a-frame", _raw_frames()[1]), "pandas DataFrame"),
        (
            lambda: (_raw_frames()[0].drop(columns=["race"]), _raw_frames()[1]),
            "missing required columns",
        ),
        (
            lambda: (
                _raw_frames()[0].assign(sex="Unknown"),
                _raw_frames()[1],
            ),
            "invalid values",
        ),
    ],
)
def test_raw_attrition_rejects_invalid_frames(factory, message) -> None:
    train, test = factory()
    with pytest.raises(DataQualityValidationError, match=message):
        audit_raw_attrition(train, test)


def test_audits_reject_invalid_configuration_and_processed_splits() -> None:
    train, test = _raw_frames()
    with pytest.raises(DataQualityValidationError, match="sampling weight"):
        audit_raw_attrition(train, test, feature_columns=[*FEATURE_COLUMNS, "fnlwgt"])
    with pytest.raises(DataQualityValidationError, match="remain unique"):
        audit_raw_attrition(train, test, missing_markers=["?", " ? "])
    with pytest.raises(DataQualityValidationError, match="positive integer"):
        audit_raw_attrition(train, test, small_group_threshold=True)

    invalid_split = _processed_frame().assign(split="holdout")
    with pytest.raises(DataQualityValidationError, match="invalid values"):
        audit_processed_quality(invalid_split)


def test_processed_quality_rejects_missing_contract_column() -> None:
    with pytest.raises(DataQualityValidationError, match="missing required columns"):
        audit_processed_quality(_processed_frame().drop(columns=["age"]))


def test_preprocessing_writes_a_digest_bound_quality_sidecar(tmp_path) -> None:
    from fairness_project.data.preprocess import prepare_model_ready_data

    train, test = _raw_frames()
    train_path = tmp_path / "adult.data"
    test_path = tmp_path / "adult.test"
    output_path = tmp_path / "adult_model_ready.csv"
    train.to_csv(train_path, header=False, index=False)
    test_path.write_text(
        "|1x3 Cross validator\n" + test.to_csv(header=False, index=False),
        encoding="utf-8",
    )

    prepared = prepare_model_ready_data(
        train_path=train_path,
        test_path=test_path,
        output_path=output_path,
        verbose=False,
    )

    sidecar_path = output_path.with_suffix(".quality.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert len(prepared) == 6
    assert sidecar["raw"]["attrition"]["overall"]["deleted_rows"] == 3
    assert sidecar["processed"]["attrition"]["overall"]["deleted_rows"] == 0
    assert sidecar["model_ready"]["sha256"] == hashlib.sha256(output_path.read_bytes()).hexdigest()
