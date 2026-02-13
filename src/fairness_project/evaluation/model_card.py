"""
Generate a model card with live metrics from a specific run.

Usage:
    python -m fairness_project.evaluation.model_card --run-id <run_id>
    python -m fairness_project.evaluation.model_card --run-id <run_id> --output docs/model_card.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_run_data(run_id: str) -> tuple[dict, dict]:
    """Load run.json and metrics.json for a given run ID.

    Parameters
    ----------
    run_id : str
        The run identifier (directory name under runs/).

    Returns
    -------
    tuple[dict, dict]
        (run_metadata, metrics) dictionaries.
    """
    run_dir = Path("runs") / run_id

    run_path = run_dir / "run.json"
    if not run_path.exists():
        raise FileNotFoundError(f"run.json not found at {run_path}")

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found at {metrics_path}")

    with open(run_path) as f:
        run_meta = json.load(f)
    with open(metrics_path) as f:
        metrics = json.load(f)

    return run_meta, metrics


def generate_model_card(run_meta: dict, metrics: dict) -> str:
    """Generate a Markdown model card from run metadata and metrics.

    Parameters
    ----------
    run_meta : dict
        Run metadata from run.json.
    metrics : dict
        Evaluation metrics from metrics.json.

    Returns
    -------
    str
        Model card as Markdown string.
    """
    from fairness_project.governance.gate import GateThresholds, check_gate

    run_id = run_meta.get("run_id", "unknown")
    timestamp = run_meta.get("timestamp") or run_meta.get("saved_at", "unknown")
    model_type = run_meta.get("model_type", "unknown")

    baseline = metrics.get("baseline", {})
    eo_adjusted = metrics.get("eo_adjusted", {})

    # Build a report structure for the governance gate
    report = {"results": {"metrics": eo_adjusted}}
    thresholds = GateThresholds()
    gate_result = check_gate(report, thresholds)

    lines = []

    # Header
    lines.append("# Model Card: Fairness-Aware Candidate Pre-Screening")
    lines.append("")
    lines.append(f"> **Auto-generated** from run `{run_id}` at {timestamp}")
    lines.append(">")
    lines.append("> To regenerate with live metrics:")
    lines.append("> ```bash")
    lines.append(f"> python -m fairness_project.evaluation.model_card --run-id {run_id}")
    lines.append("> ```")
    lines.append("")

    # Model Details
    lines.append("## Model Details")
    lines.append("")
    lines.append(f"- **Run ID**: `{run_id}`")
    lines.append(f"- **Model Type**: {model_type}")
    lines.append(f"- **Trained At**: {timestamp}")
    lines.append("- **Framework**: scikit-learn, XGBoost")
    lines.append("")

    # Performance Metrics
    lines.append("## Performance Metrics")
    lines.append("")
    lines.append("| Metric | Baseline | EO-Adjusted | Change |")
    lines.append("|--------|----------|-------------|--------|")

    for metric in ("accuracy", "precision", "recall", "f1"):
        base_val = baseline.get(metric, 0)
        eo_val = eo_adjusted.get(metric, 0)
        change = eo_val - base_val
        lines.append(f"| {metric.capitalize()} | {base_val:.4f} | {eo_val:.4f} | {change:+.4f} |")

    lines.append("")

    # Fairness Metrics
    lines.append("## Fairness Metrics")
    lines.append("")
    lines.append("| Metric | Baseline | EO-Adjusted | Target |")
    lines.append("|--------|----------|-------------|--------|")

    fairness_targets = {
        "SPD": f"|SPD| <= {thresholds.max_spd}",
        "DI": f"DI >= {thresholds.min_disparate_impact}",
        "TPR_gap": f"|gap| <= {thresholds.max_tpr_gap}",
    }

    for key, target in fairness_targets.items():
        base_val = baseline.get(key, 0)
        eo_val = eo_adjusted.get(key, 0)
        lines.append(f"| {key} | {base_val:.4f} | {eo_val:.4f} | {target} |")

    lines.append("")

    # Governance Gate Status
    lines.append("## Governance Gate Status")
    lines.append("")
    status = "PASSED" if gate_result.passed else "FAILED"
    lines.append(f"**Overall**: {status}")
    lines.append("")
    lines.append("| Check | Value | Threshold | Status |")
    lines.append("|-------|-------|-----------|--------|")

    checks = [
        (
            "Accuracy",
            eo_adjusted.get("accuracy"),
            f">= {thresholds.min_accuracy}",
            thresholds.min_accuracy,
        ),
        (
            "|TPR Gap|",
            abs(eo_adjusted.get("TPR_gap", 0)),
            f"<= {thresholds.max_tpr_gap}",
            thresholds.max_tpr_gap,
        ),
        (
            "Disparate Impact",
            eo_adjusted.get("DI"),
            f">= {thresholds.min_disparate_impact}",
            thresholds.min_disparate_impact,
        ),
        ("|SPD|", abs(eo_adjusted.get("SPD", 0)), f"<= {thresholds.max_spd}", thresholds.max_spd),
    ]

    for label, value, threshold_str, threshold_val in checks:
        if value is None:
            lines.append(f"| {label} | N/A | {threshold_str} | -- |")
            continue
        if label.startswith("|"):
            passed = value <= threshold_val
        elif "Accuracy" in label:
            passed = value >= threshold_val
        elif "Disparate" in label:
            passed = value >= threshold_val
        else:
            passed = value <= threshold_val
        mark = "PASS" if passed else "FAIL"
        lines.append(f"| {label} | {value:.4f} | {threshold_str} | {mark} |")

    lines.append("")

    # Standard sections
    lines.append("## Intended Use")
    lines.append("")
    lines.append("- **Primary Use**: Educational demonstration of fairness-aware ML pipelines")
    lines.append("- **Intended Users**: ML researchers, students, practitioners")
    lines.append(
        "- **Out-of-Scope Uses**: Production hiring decisions without additional validation"
    )
    lines.append("")

    lines.append("## Ethical Considerations")
    lines.append("")
    lines.append("- **Protected Attributes**: sex, race")
    lines.append("- **Mitigation Applied**: Equal Opportunity post-processing")
    lines.append("- Historical bias in training data (1994 Census)")
    lines.append("- Binary groupings may oversimplify demographic categories")
    lines.append("")

    lines.append("## Evaluation Protocol")
    lines.append("")
    lines.append("- **Split**: Train / Validation (15% of train) / Test")
    lines.append("- **Leakage Prevention**: EO thresholds tuned on validation only")
    lines.append("- **Reporting**: Final metrics computed on held-out test set")
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for model card generation."""
    parser = argparse.ArgumentParser(
        description="Generate a model card with live metrics from a specific run",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run identifier (directory name under runs/)",
    )
    parser.add_argument(
        "--output",
        default="docs/model_card.md",
        help="Output path for the model card (default: docs/model_card.md)",
    )

    args = parser.parse_args(argv)

    try:
        run_meta, metrics = load_run_data(args.run_id)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    content = generate_model_card(run_meta, metrics)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)

    print(f"Model card generated at: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
