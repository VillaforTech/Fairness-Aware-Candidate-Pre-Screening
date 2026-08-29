"""Tests for the single canonical validation split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fairness_project.data.split import create_train_val_test_split


def test_split_is_deterministic_disjoint_and_preserves_test(sample_adult_data) -> None:
    before_test = sample_adult_data.index[sample_adult_data["split"] == "test"].tolist()
    first = create_train_val_test_split(sample_adult_data, random_state=7)
    second = create_train_val_test_split(sample_adult_data, random_state=7)

    assert first["split"].tolist() == second["split"].tolist()
    assert first.index[first["split"] == "test"].tolist() == before_test
    assert set(first["split"]) == {"train", "val", "test"}
    assert len(first) == len(sample_adult_data)


def test_split_preserves_joint_target_group_proportions(sample_adult_data) -> None:
    result = create_train_val_test_split(sample_adult_data, val_ratio=0.2, random_state=42)
    original = sample_adult_data[sample_adult_data["split"] == "train"]
    validation = result[result["split"] == "val"]
    columns = ["income", "sex", "race_binary"]
    original_rates = original.groupby(columns).size() / len(original)
    validation_rates = validation.groupby(columns).size() / len(validation)
    aligned = original_rates.to_frame("original").join(
        validation_rates.to_frame("validation"),
        how="left",
    )
    assert aligned["validation"].notna().all()
    assert (aligned["original"] - aligned["validation"]).abs().max() < 0.04


def test_split_does_not_mutate_numpy_global_rng(sample_adult_data) -> None:
    np.random.seed(123)
    expected = np.random.random(3)
    np.random.seed(123)
    create_train_val_test_split(sample_adult_data, random_state=9)
    observed = np.random.random(3)
    np.testing.assert_allclose(observed, expected)


@pytest.mark.parametrize("ratio", [0, 1, -0.1, 1.1])
def test_invalid_validation_ratio_rejected(sample_adult_data, ratio) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        create_train_val_test_split(sample_adult_data, val_ratio=ratio)


def test_sparse_stratum_rejected() -> None:
    frame = pd.DataFrame(
        {
            "split": ["train", "train", "test"],
            "income": [">50K", "<=50K", ">50K"],
            "sex": ["Male", "Female", "Male"],
            "race_binary": ["White", "Non-White", "White"],
        }
    )
    with pytest.raises(ValueError, match="at least two"):
        create_train_val_test_split(frame)
