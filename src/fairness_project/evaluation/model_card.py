"""Render a model card from the same report consumed by the policy gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from fairness_project.models.artifact import load_bundle


def load_run_data(
    run_id: str,
    runs_dir: str | Path = "runs",
) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = load_bundle(Path(runs_dir) / run_id)
    return bundle.manifest, bundle.report


def generate_model_card(manifest: dict[str, Any], report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    protocol = report["protocol"]
    results = report["results"]
    baseline = results["baseline_metrics"]
    adjusted = results["metrics"]
    thresholds = results["thresholds"]
    split_counts = protocol["split_counts"]
    uncertainty = results.get("uncertainty", {})
    adjusted_intervals = uncertainty.get("intervals", {}).get("adjusted", {})
    governance = report["governance"]
    dirty_state = manifest["dirty_worktree"]
    dirty_label = "unavailable (installed distribution)" if dirty_state is None else dirty_state
    lines = [
        "# Model Card: UCI Adult Fairness Audit",
        "",
        f"> Generated from validated run `{manifest['run_id']}`.",
        "> This is a benchmark evaluation artifact, not a hiring model.",
        "",
        "## Provenance",
        "",
        f"- Model: `{manifest['model_type']}`",
        f"- Created: {manifest['created_at']}",
        f"- Git commit: `{manifest['git_commit']}`",
        f"- Data SHA-256: `{manifest['data_sha256']}`",
        f"- Source SHA-256: `{manifest['source_sha256']}`",
        f"- Seed: {metadata['seed']}",
        f"- Dirty worktree recorded: {dirty_label}",
        "",
        "## Evaluation protocol",
        "",
        f"- Dataset: {protocol['dataset']}",
        f"- Validation: {protocol['validation_strategy']}",
        (
            "- Rows: "
            f"{split_counts['train']:,} fit / {split_counts['val']:,} validation / "
            f"{split_counts['test']:,} test"
        ),
        "- EO thresholds were tuned on validation labels only.",
        "- Final metrics were computed once on the preserved official test partition.",
        "- Protected attributes were excluded from model features.",
        (
            "- Frozen offline thresholds: "
            f"Male `{float(thresholds['privileged']):.3f}`, "
            f"Female `{float(thresholds['unprivileged']):.3f}`."
        ),
        "",
        "## Results",
        "",
        "| Metric | Baseline | Offline adjusted | Change |",
        "|---|---:|---:|---:|",
    ]
    for metric in ("accuracy", "precision", "recall", "f1", "SPD", "DI", "TPR_gap"):
        before = float(baseline[metric])
        after = float(adjusted[metric])
        lines.append(f"| {metric} | {before:.4f} | {after:.4f} | {after - before:+.4f} |")

    if adjusted_intervals:
        confidence = float(uncertainty["confidence"]) * 100
        lines.extend(
            [
                "",
                f"### Adjusted {confidence:.0f}% paired-bootstrap intervals",
                "",
                "| Metric | Lower | Upper |",
                "|---|---:|---:|",
            ]
        )
        for metric in ("accuracy", "SPD", "DI", "TPR_gap"):
            interval = adjusted_intervals[metric]
            lines.append(
                f"| {metric} | {float(interval['lower']):.4f} | {float(interval['upper']):.4f} |"
            )

    verdict = "passed" if governance["passed"] else "rejected"
    lines.extend(
        [
            "",
            "## Experimental policy gate",
            "",
            f"### Verdict: {verdict}",
            "",
        ]
    )
    if governance["violations"]:
        lines.extend(f"- {violation}" for violation in governance["violations"])
    else:
        lines.append("- No configured threshold violations.")

    lines.extend(
        [
            "",
            "## Serving boundary",
            "",
            "The local API serves the baseline global threshold only. It does not apply the",
            "offline sex-specific thresholds. API responses name the policy and artifact ID",
            "so the offline fairness experiment cannot be mistaken for deployed behavior.",
            "",
            "## Limitations",
            "",
            "- Adult is a 1994 census-income dataset, not applicant or job-performance data.",
            "- Binary sex and race groupings erase identity and intersectional detail.",
            "- Bootstrap intervals describe test-sample uncertainty, not external validity.",
            "- Passing a configurable gate would not establish safety, legality, or validity.",
            "",
            "## Authors",
            "",
            "Roberto Villafuerte and Charles Santhakumar, University of Helsinki collaboration.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a model card from a run bundle")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output", default="docs/model_card.md")
    args = parser.parse_args(argv)
    try:
        manifest, report = load_run_data(args.run_id, args.runs_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generate_model_card(manifest, report), encoding="utf-8")
    print(f"Model card generated at: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
