"""Repeated-seed robustness study built from complete canonical run bundles."""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from fairness_project.config import Config, config_from_dict, resolved_config
from fairness_project.evaluation.stability import summarize_stability
from fairness_project.experiment import ExperimentResult, run_experiment
from fairness_project.models.artifact import write_json
from fairness_project.models.train import ModelType
from fairness_project.reporting import write_audit_html


@dataclass(frozen=True)
class StabilityRunResult:
    """Paths and constituent runs produced by one robustness study."""

    study_dir: Path
    summary_path: Path
    html_path: Path
    runs: tuple[ExperimentResult, ...]
    summary: dict


def _validate_seeds(seeds: list[int]) -> list[int]:
    if len(seeds) < 2:
        raise ValueError("A stability study requires at least two seeds")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("Seeds must be non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique")
    return sorted(seeds)


def run_stability_study(
    *,
    data_path: str | Path,
    output_dir: str | Path,
    study_id: str,
    model_type: ModelType = "xgb",
    seeds: list[int],
    config: Config | None = None,
    bootstrap_samples: int = 100,
) -> StabilityRunResult:
    """Retrain, retune, evaluate, and aggregate one comparable run per seed."""
    ordered_seeds = _validate_seeds(seeds)
    if study_id in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", study_id):
        raise ValueError(
            "study_id may contain only letters, numbers, dots, underscores, and dashes"
        )
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    destination = Path(output_dir) / study_id
    if destination.exists():
        raise FileExistsError(f"Study directory already exists: {destination}")
    destination.mkdir(parents=True)
    try:
        runs: list[ExperimentResult] = []
        for seed in ordered_seeds:
            run_config = config_from_dict(resolved_config(config or Config()))
            runs.append(
                run_experiment(
                    data_path=data_path,
                    output_dir=destination / "runs",
                    model_type=model_type,
                    seed=seed,
                    val_ratio=run_config.data.val_size,
                    run_id=f"{model_type}-seed-{seed}",
                    config=run_config,
                    bootstrap_samples=bootstrap_samples,
                )
            )
        summary = summarize_stability([run.report for run in runs])
        summary_path = write_json(destination / "stability.json", summary)
        worst_seed = int(summary["worst_seed"])
        representative = next(run for run in runs if run.report["metadata"]["seed"] == worst_seed)
        html_path = write_audit_html(
            representative.report,
            destination / "audit.html",
            stability=summary,
        )
        return StabilityRunResult(
            study_dir=destination,
            summary_path=summary_path,
            html_path=html_path,
            runs=tuple(runs),
            summary=summary,
        )
    except Exception:
        failure_marker = destination / "INCOMPLETE"
        temporary = failure_marker.parent / (f".{failure_marker.name}.tmp-{uuid.uuid4().hex}")
        try:
            temporary.write_text("Study generation did not complete.\n", encoding="utf-8")
            os.replace(temporary, failure_marker)
        finally:
            temporary.unlink(missing_ok=True)
        raise


__all__ = ["StabilityRunResult", "run_stability_study"]
