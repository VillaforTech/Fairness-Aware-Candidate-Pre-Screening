"""Tests for repeated-seed study orchestration."""

from __future__ import annotations

from pathlib import Path

from fairness_project.config import Config
from fairness_project.experiment import ExperimentResult
from fairness_project.governance.gate import GateResult
from fairness_project.study import run_stability_study


def test_study_propagates_effective_validation_ratio_without_mutating_config(
    monkeypatch, tmp_path
) -> None:
    config = Config()
    config.data.val_size = 0.23
    captured: list[dict] = []

    def fake_run_experiment(**kwargs):
        captured.append(kwargs)
        seed = kwargs["seed"]
        return ExperimentResult(
            run_id=f"lr-seed-{seed}",
            run_dir=tmp_path / "runs" / f"lr-seed-{seed}",
            report={"metadata": {"seed": seed}},
            gate=GateResult(passed=False),
        )

    monkeypatch.setattr("fairness_project.study.run_experiment", fake_run_experiment)
    monkeypatch.setattr(
        "fairness_project.study.summarize_stability",
        lambda reports: {"worst_seed": 3, "reports": reports},
    )

    def fake_write_html(report, destination, stability):
        path = Path(destination)
        path.write_text(str((report, stability)), encoding="utf-8")
        return path

    monkeypatch.setattr("fairness_project.study.write_audit_html", fake_write_html)
    data_path = tmp_path / "adult.csv"
    data_path.write_text("placeholder", encoding="utf-8")

    result = run_stability_study(
        data_path=data_path,
        output_dir=tmp_path / "studies",
        study_id="propagation",
        model_type="lr",
        seeds=[7, 3],
        config=config,
        bootstrap_samples=0,
    )

    assert [call["seed"] for call in captured] == [3, 7]
    assert [call["val_ratio"] for call in captured] == [0.23, 0.23]
    assert all(call["config"] is not config for call in captured)
    assert config.seed == 42
    assert config.model.random_state == 42
    assert result.html_path.is_file()
