"""Tests for the portable HTML audit renderer."""

from __future__ import annotations

import math

import pytest

from fairness_project.reporting import render_audit_html, write_audit_html


def _metrics(accuracy: float, spd: float, di: float, tpr_gap: float) -> dict[str, float]:
    return {"accuracy": accuracy, "SPD": spd, "DI": di, "TPR_gap": tpr_gap}


def _report() -> dict:
    baseline_group = {
        "n": 70,
        "positive_labels": 20,
        "negative_labels": 50,
        "predicted_positive_rate": 0.31,
        "tpr": 0.70,
        "fpr": 0.16,
    }
    adjusted_group = {**baseline_group, "predicted_positive_rate": 0.34, "tpr": 0.80}
    return {
        "schema_version": "1.0",
        "metadata": {
            "run_id": "xgb-seed-42",
            "timestamp": "2026-08-30T12:00:00+00:00",
            "seed": 42,
            "model_type": "xgb",
            "git_commit": "a" * 40,
            "dirty_worktree": False,
            "data_sha256": "b" * 64,
            "source_sha256": "c" * 64,
            "config_sha256": "d" * 64,
        },
        "protocol": {
            "dataset": "UCI Adult (1994 Census income classification)",
            "scope": "benchmark evaluation only; not a hiring validity study",
            "feature_contract_id": "adult-income-v2",
            "threshold_tuning_split": "val",
            "final_evaluation_split": "test",
        },
        "results": {
            "baseline_metrics": _metrics(0.87, 0.18, 0.34, 0.05),
            "metrics": _metrics(0.866, 0.14, 0.42, -0.012),
            "validation_tuning": {
                "selection": {
                    "status": "feasible",
                    "selected": {
                        "accuracy": 0.86,
                        "tpr_gap": 0.01,
                        "threshold_privileged": 0.5,
                        "threshold_unprivileged": 0.4,
                    },
                    "frontier": [
                        {
                            "accuracy": 0.84,
                            "tpr_gap": 0.0,
                            "threshold_privileged": 0.55,
                            "threshold_unprivileged": 0.38,
                        },
                        {
                            "accuracy": 0.86,
                            "tpr_gap": 0.01,
                            "threshold_privileged": 0.5,
                            "threshold_unprivileged": 0.4,
                        },
                        {
                            "accuracy": 0.87,
                            "tpr_gap": 0.05,
                            "threshold_privileged": 0.5,
                            "threshold_unprivileged": 0.5,
                        },
                    ],
                }
            },
            "validation_dependence": {
                "schema_version": "1.0",
                "counts": {
                    "train_rows": 80,
                    "validation_rows": 20,
                    "exact_feature_overlap_rows": 4,
                    "exact_feature_overlap_rate": 0.2,
                    "exact_full_record_overlap_rows": 1,
                    "exact_full_record_overlap_rate": 0.05,
                    "overlap_excluded_validation_rows": 16,
                },
                "overlap_excluded_retuning": {
                    "status": "completed",
                    "frontier_selection_status": "feasible",
                    "selected_frontier_policy": {
                        "threshold_privileged": 0.52,
                        "threshold_unprivileged": 0.41,
                    },
                    "review_policy": {
                        "lower_threshold": 0.45,
                        "upper_threshold": 0.55,
                    },
                },
            },
            "subgroup_diagnostics": {
                "baseline": {"sex": {"Female": baseline_group}},
                "adjusted": {"sex": {"Female": adjusted_group}},
            },
            "sampling_weight_sensitivity": {
                "status": "sensitivity_only",
                "adjusted_metrics": _metrics(0.861, 0.15, 0.40, -0.02),
                "interpretation": "CPS final weights are used for sensitivity only.",
            },
            "feature_overlap_sensitivity": {
                "counts": {
                    "reference_rows": 100,
                    "held_out_rows": 25,
                    "overlap_rows": 5,
                    "novel_rows": 20,
                    "overlap_rate": 0.2,
                },
                "slices": {
                    "all_held_out": {
                        "baseline": {"metrics": _metrics(0.87, 0.18, 0.34, 0.05)},
                        "adjusted": {"metrics": _metrics(0.866, 0.14, 0.42, -0.012)},
                    },
                    "overlap_excluded": {
                        "evidence_status": "sufficient",
                        "evidence_reasons": [],
                        "baseline": {"metrics": _metrics(0.86, 0.19, 0.33, 0.06)},
                        "adjusted": {"metrics": _metrics(0.855, 0.15, 0.41, -0.02)},
                    },
                },
            },
            "uncertainty": {"method": "paired stratified bootstrap"},
        },
        "governance": {
            "passed": False,
            "report_valid": True,
            "violations": ["DI=0.4200 < min_disparate_impact=0.8"],
            "metrics_checked": {"DI": 0.42},
        },
    }


def test_renders_complete_accessible_self_contained_report() -> None:
    rendered = render_audit_html(
        _report(),
        {"distributions": {"accuracy": [0.85, 0.86, 0.87], "TPR_gap": [-0.02, 0.01]}},
    )

    assert rendered.startswith("<!doctype html>")
    assert "Auditable Fair-ML Policy Lab" in rendered
    assert "What changed" in rendered
    assert "Utility versus |TPR gap|" in rendered
    assert "Cell-level evidence" in rendered
    assert "What remains without repeats" in rendered
    assert "Base novel" in rendered
    assert "Before policy selection" in rendered
    assert "Feature-overlap rows" in rendered
    assert "Sampling-weight sensitivity" in rendered
    assert "Configured gate rejected" in rendered
    assert "DI=0.4200 &lt; min_disparate_impact=0.8" in rendered
    assert "Stability distributions" in rendered
    assert "n=3" in rendered
    assert "Machine-readable provenance snapshot" in rendered
    assert "How to read this artifact" in rendered
    assert 'role="img"' in rendered
    assert "<title id=" in rendered
    assert "@media print" in rendered
    assert "prefers-reduced-motion" in rendered
    assert "https://" not in rendered
    assert "<script" not in rendered.lower()


def test_all_report_text_and_json_are_html_escaped() -> None:
    report = _report()
    attack = '</script><script>alert("owned")</script><img src=x onerror=alert(1)>'
    report["metadata"]["run_id"] = attack
    report["protocol"]["scope"] = attack
    report["results"]["sampling_weight_sensitivity"]["interpretation"] = attack
    report["governance"]["violations"] = [attack]

    rendered = render_audit_html(report, {"distributions": {attack: [1.0, 2.0]}})

    assert attack not in rendered
    assert "&lt;/script&gt;&lt;script&gt;alert(&quot;" in rendered
    assert "&lt;img src=x onerror=alert(1)&gt;" in rendered
    assert "<script" not in rendered.lower()
    assert "onerror=" not in rendered.replace("onerror=alert(1)&gt;", "")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: "not-an-object", "report must be an object"),
        (lambda value: {**value, "schema_version": ""}, "schema_version"),
        (lambda value: {**value, "metadata": None}, "metadata"),
        (lambda value: {**value, "protocol": None}, "protocol"),
        (lambda value: {**value, "results": None}, "results"),
        (
            lambda value: {
                **value,
                "results": {**value["results"], "baseline_metrics": {}},
            },
            "accuracy",
        ),
        (
            lambda value: {
                **value,
                "results": {
                    **value["results"],
                    "metrics": {**value["results"]["metrics"], "TPR_gap": math.nan},
                },
            },
            "TPR_gap",
        ),
    ],
)
def test_rejects_malformed_core_report_shapes(mutate, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        render_audit_html(mutate(_report()))


def test_optional_sections_degrade_explicitly_and_output_is_deterministic() -> None:
    report = _report()
    report["results"].pop("validation_tuning")
    report["results"].pop("subgroup_diagnostics")
    report["results"].pop("sampling_weight_sensitivity")
    report["results"].pop("feature_overlap_sensitivity")
    report["results"].pop("validation_dependence")
    report.pop("governance")

    first = render_audit_html(report)
    second = render_audit_html(report)

    assert first == second
    assert "No validation frontier is recorded" in first
    assert "No subgroup diagnostic cells are recorded" in first
    assert "No sampling-weight sensitivity is recorded" in first
    assert "No exact-feature overlap sensitivity is recorded" in first
    assert "No validation-overlap dependence audit is recorded" in first
    assert "No governance-gate verdict is recorded" in first
    assert "No repeat-run stability artifact" in first


def test_write_audit_html_creates_parent_and_returns_path(tmp_path) -> None:
    destination = tmp_path / "nested" / "audit.html"

    returned = write_audit_html(_report(), destination, {"runs": [{"accuracy": 0.86}]})

    assert returned == destination
    assert destination.read_text(encoding="utf-8") == render_audit_html(
        _report(), {"runs": [{"accuracy": 0.86}]}
    )


def test_canonical_stability_summary_renders_metric_and_threshold_distributions() -> None:
    stability = {
        "conditions": {
            "baseline": {"metrics": {"accuracy": {"values": [0.84, 0.86]}}},
            "adjusted": {
                "metrics": {
                    "accuracy": {"values": [0.83, 0.85]},
                    "TPR_gap": {"values": [-0.01, 0.02]},
                }
            },
        },
        "thresholds": {
            "privileged": {"values": [0.50, 0.51]},
            "unprivileged": {"values": [0.39, 0.41]},
        },
    }

    rendered = render_audit_html(_report(), stability)

    assert "baseline accuracy" in rendered
    assert "adjusted TPR_gap" in rendered
    assert "privileged threshold" in rendered
    assert "The supplied stability artifact contains no numeric distributions" not in rendered


def test_stability_must_be_an_object() -> None:
    with pytest.raises(ValueError, match="stability must be an object"):
        render_audit_html(_report(), stability=[1, 2, 3])  # type: ignore[arg-type]
