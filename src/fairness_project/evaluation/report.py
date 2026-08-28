"""
Report generation for model evaluation results.

Generates JSON and Markdown reports for experiment documentation.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def generate_run_metadata(
    run_id: str | None = None,
    seed: int = 42,
    model_type: str | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    """
    Generate metadata for a run.

    Parameters
    ----------
    run_id : str, optional
        Run identifier. Auto-generated if not provided.
    seed : int
        Random seed used.
    model_type : str, optional
        Type of model trained.
    config_path : str, optional
        Path to config file used.

    Returns
    -------
    dict[str, Any]
        Run metadata.
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Try to get git commit hash
    git_commit = None
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_commit = result.stdout.strip()[:8]
    except Exception:
        pass

    return {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "model_type": model_type,
        "config_path": config_path,
        "git_commit": git_commit,
    }


def generate_json_report(
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    thresholds: dict[str, float] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Generate a JSON report with all experiment results.

    Parameters
    ----------
    metadata : dict[str, Any]
        Run metadata.
    metrics : dict[str, Any]
        Evaluation metrics (baseline and EO-adjusted).
    thresholds : dict[str, float], optional
        EO thresholds used.
    output_path : str | Path, optional
        Path to save JSON report.

    Returns
    -------
    dict[str, Any]
        Complete report as dictionary.
    """
    report = {
        "metadata": metadata,
        "results": {
            "metrics": metrics,
        },
    }

    if thresholds:
        report["results"]["thresholds"] = thresholds

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"JSON report saved to: {output_path}")

    return report


def generate_markdown_report(
    metadata: dict[str, Any],
    baseline_metrics: dict[str, Any],
    eo_metrics: dict[str, Any],
    thresholds: dict[str, float] | None = None,
    output_path: str | Path | None = None,
) -> str:
    """
    Generate a Markdown report summarizing results.

    Parameters
    ----------
    metadata : dict[str, Any]
        Run metadata.
    baseline_metrics : dict[str, Any]
        Baseline evaluation metrics.
    eo_metrics : dict[str, Any]
        EO-adjusted evaluation metrics.
    thresholds : dict[str, float], optional
        EO thresholds used.
    output_path : str | Path, optional
        Path to save Markdown report.

    Returns
    -------
    str
        Markdown report content.
    """
    lines = []

    # Header
    lines.append("# Fairness Evaluation Report")
    lines.append("")
    lines.append(f"**Run ID**: {metadata.get('run_id', 'N/A')}")
    lines.append(f"**Timestamp**: {metadata.get('timestamp', 'N/A')}")
    lines.append(f"**Model**: {metadata.get('model_type', 'N/A')}")
    lines.append(f"**Seed**: {metadata.get('seed', 'N/A')}")
    if metadata.get("git_commit"):
        lines.append(f"**Git Commit**: {metadata['git_commit']}")
    lines.append("")

    # Performance Metrics
    lines.append("## Performance Metrics")
    lines.append("")
    lines.append("| Metric | Baseline | EO-Adjusted | Change |")
    lines.append("|--------|----------|-------------|--------|")

    perf_metrics = ["accuracy", "precision", "recall", "f1"]
    for metric in perf_metrics:
        base = baseline_metrics.get(metric, 0)
        eo = eo_metrics.get(metric, 0)
        change = eo - base if base is not None and eo is not None else None
        change_str = f"{change:+.4f}" if change is not None else "N/A"
        lines.append(f"| {metric.capitalize()} | {base:.4f} | {eo:.4f} | {change_str} |")

    lines.append("")

    # Fairness Metrics
    lines.append("## Fairness Metrics")
    lines.append("")
    lines.append("| Metric | Baseline | EO-Adjusted | Change |")
    lines.append("|--------|----------|-------------|--------|")

    fair_metrics = [("SPD", "SPD"), ("DI", "DI"), ("TPR_gap", "TPR Gap")]
    for key, label in fair_metrics:
        base = baseline_metrics.get(key, 0)
        eo = eo_metrics.get(key, 0)
        change = eo - base if base is not None and eo is not None else None
        change_str = f"{change:+.4f}" if change is not None else "N/A"
        lines.append(f"| {label} | {base:.4f} | {eo:.4f} | {change_str} |")

    lines.append("")

    # TPR by Group
    lines.append("## TPR by Group")
    lines.append("")
    lines.append("| Group | Baseline | EO-Adjusted |")
    lines.append("|-------|----------|-------------|")

    tpr_priv_base = baseline_metrics.get("TPR_priv", 0)
    tpr_priv_eo = eo_metrics.get("TPR_priv", 0)
    tpr_unpriv_base = baseline_metrics.get("TPR_unpriv", 0)
    tpr_unpriv_eo = eo_metrics.get("TPR_unpriv", 0)

    lines.append(f"| Privileged | {tpr_priv_base:.4f} | {tpr_priv_eo:.4f} |")
    lines.append(f"| Unprivileged | {tpr_unpriv_base:.4f} | {tpr_unpriv_eo:.4f} |")
    lines.append("")

    # Thresholds
    if thresholds:
        lines.append("## EO Thresholds")
        lines.append("")
        lines.append(f"- Privileged group threshold: {thresholds.get('privileged', 0.5):.3f}")
        lines.append(f"- Unprivileged group threshold: {thresholds.get('unprivileged', 0.5):.3f}")
        lines.append("")

    # Key Findings
    lines.append("## Key Findings")
    lines.append("")

    tpr_gap_base = baseline_metrics.get("TPR_gap", 0)
    tpr_gap_eo = eo_metrics.get("TPR_gap", 0)
    improvement = abs(tpr_gap_base) - abs(tpr_gap_eo)

    lines.append(f"- **TPR Gap Reduction**: {improvement:.4f}")
    lines.append(f"- **Original TPR Gap**: {tpr_gap_base:.4f}")
    lines.append(f"- **Final TPR Gap**: {tpr_gap_eo:.4f}")

    acc_base = baseline_metrics.get("accuracy", 0)
    acc_eo = eo_metrics.get("accuracy", 0)
    acc_change = acc_eo - acc_base

    lines.append(f"- **Accuracy Change**: {acc_change:+.4f}")
    lines.append("")

    # Notes
    lines.append("## Methodology Notes")
    lines.append("")
    lines.append("This evaluation uses a **leakage-free protocol**:")
    lines.append("- EO thresholds are tuned on validation data only")
    lines.append("- Test set labels are never used during threshold selection")
    lines.append("- This avoids threshold-selection leakage on the test split")
    lines.append("- It does not establish external validity or generalization")
    lines.append("")

    content = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(content)
        print(f"Markdown report saved to: {output_path}")

    return content


def generate_comparison_table(
    results: list[dict[str, Any]],
    output_path: str | Path | None = None,
) -> str:
    """
    Generate a comparison table for multiple models.

    Parameters
    ----------
    results : list[dict[str, Any]]
        List of results dictionaries, each with 'model', 'baseline', 'eo_adjusted'.
    output_path : str | Path, optional
        Path to save Markdown table.

    Returns
    -------
    str
        Markdown table content.
    """
    lines = []
    lines.append("# Model Comparison")
    lines.append("")
    lines.append("## Baseline Performance")
    lines.append("")
    lines.append("| Model | Accuracy | F1 | SPD | DI | TPR Gap |")
    lines.append("|-------|----------|-----|-----|-----|---------|")

    for r in results:
        model = r.get("model", "Unknown")
        base = r.get("baseline", {})
        lines.append(
            f"| {model} | {base.get('accuracy', 0):.4f} | "
            f"{base.get('f1', 0):.4f} | {base.get('SPD', 0):.4f} | "
            f"{base.get('DI', 0):.4f} | {base.get('TPR_gap', 0):.4f} |"
        )

    lines.append("")
    lines.append("## After EO Post-Processing")
    lines.append("")
    lines.append("| Model | Accuracy | F1 | SPD | DI | TPR Gap |")
    lines.append("|-------|----------|-----|-----|-----|---------|")

    for r in results:
        model = r.get("model", "Unknown")
        eo = r.get("eo_adjusted", {})
        lines.append(
            f"| {model} | {eo.get('accuracy', 0):.4f} | "
            f"{eo.get('f1', 0):.4f} | {eo.get('SPD', 0):.4f} | "
            f"{eo.get('DI', 0):.4f} | {eo.get('TPR_gap', 0):.4f} |"
        )

    content = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(content)
        print(f"Comparison table saved to: {output_path}")

    return content
