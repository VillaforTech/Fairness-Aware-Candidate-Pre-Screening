"""Tests for subgroup counts and paired uncertainty output."""

import numpy as np
import pandas as pd

from fairness_project.evaluation.diagnostics import (
    group_diagnostics,
    paired_bootstrap_intervals,
)


def test_group_diagnostics_exposes_cell_denominators() -> None:
    result = group_diagnostics(
        np.array([1, 0, 1, 0]),
        np.array([1, 0, 0, 1]),
        pd.Series(["A", "A", "B", "B"]),
    )
    assert result["A"]["n"] == 2
    assert result["A"]["positive_labels"] == 1
    assert result["B"]["negative_labels"] == 1


def test_paired_bootstrap_is_deterministic() -> None:
    y_true = np.array([1, 0, 1, 0] * 20)
    sensitive = np.array(["Male", "Male", "Female", "Female"] * 20)
    baseline = np.array([1, 0, 0, 0] * 20)
    adjusted = np.array([1, 0, 1, 0] * 20)
    first = paired_bootstrap_intervals(
        y_true=y_true,
        baseline_pred=baseline,
        adjusted_pred=adjusted,
        sensitive=sensitive,
        privileged_group="Male",
        samples=20,
        random_state=7,
    )
    second = paired_bootstrap_intervals(
        y_true=y_true,
        baseline_pred=baseline,
        adjusted_pred=adjusted,
        sensitive=sensitive,
        privileged_group="Male",
        samples=20,
        random_state=7,
    )
    assert first == second
    assert first["intervals"]["change"]["accuracy"]["median"] == 0.25
