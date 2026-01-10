"""
Data and fairness drift monitoring.

Provides utilities to detect drift between reference and production data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-8,
) -> float:
    """
    Compute Population Stability Index (PSI).

    PSI measures distribution shift between reference and current data.
    PSI < 0.1: No significant change
    PSI 0.1-0.25: Moderate change
    PSI > 0.25: Significant change

    Parameters
    ----------
    reference : np.ndarray
        Reference distribution values.
    current : np.ndarray
        Current distribution values.
    n_bins : int
        Number of bins for discretization.
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    float
        PSI value.
    """
    # Create bins from reference data
    bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf

    # Compute bin proportions
    ref_counts = np.histogram(reference, bins=bins)[0] / len(reference)
    cur_counts = np.histogram(current, bins=bins)[0] / len(current)

    # Add epsilon to avoid log(0)
    ref_counts = np.clip(ref_counts, eps, 1)
    cur_counts = np.clip(cur_counts, eps, 1)

    # PSI formula
    psi = np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts))

    return float(psi)


def compute_ks_statistic(
    reference: np.ndarray,
    current: np.ndarray,
) -> tuple[float, float]:
    """
    Compute Kolmogorov-Smirnov statistic.

    Parameters
    ----------
    reference : np.ndarray
        Reference distribution.
    current : np.ndarray
        Current distribution.

    Returns
    -------
    tuple[float, float]
        KS statistic and p-value.
    """
    from scipy import stats

    ks_stat, p_value = stats.ks_2samp(reference, current)
    return float(ks_stat), float(p_value)


def compute_feature_drift(
    df_reference: pd.DataFrame,
    df_current: pd.DataFrame,
    numeric_features: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Compute drift metrics for all numeric features.

    Parameters
    ----------
    df_reference : pd.DataFrame
        Reference data.
    df_current : pd.DataFrame
        Current data.
    numeric_features : list[str], optional
        List of numeric features to check.

    Returns
    -------
    dict[str, dict[str, float]]
        Dictionary mapping feature names to drift metrics.
    """
    if numeric_features is None:
        numeric_features = df_reference.select_dtypes(include=[np.number]).columns.tolist()

    results = {}
    for col in numeric_features:
        if col not in df_current.columns:
            continue

        ref = df_reference[col].dropna().values
        cur = df_current[col].dropna().values

        if len(ref) == 0 or len(cur) == 0:
            continue

        psi = compute_psi(ref, cur)
        ks_stat, ks_pvalue = compute_ks_statistic(ref, cur)

        results[col] = {
            "psi": psi,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pvalue,
            "ref_mean": float(ref.mean()),
            "cur_mean": float(cur.mean()),
            "mean_shift": float(cur.mean() - ref.mean()),
        }

    return results


def compute_fairness_drift(
    df_reference: pd.DataFrame,
    df_current: pd.DataFrame,
    y_pred_ref: np.ndarray,
    y_pred_cur: np.ndarray,
    sensitive_col: str = "sex",
    privileged_value: str = "Male",
) -> dict[str, float]:
    """
    Compute drift in fairness metrics.

    Parameters
    ----------
    df_reference : pd.DataFrame
        Reference data.
    df_current : pd.DataFrame
        Current data.
    y_pred_ref : np.ndarray
        Reference predictions.
    y_pred_cur : np.ndarray
        Current predictions.
    sensitive_col : str
        Sensitive attribute column.
    privileged_value : str
        Privileged group value.

    Returns
    -------
    dict[str, float]
        Dictionary with fairness metric drifts.
    """
    from fairness_project.metrics.fairness import (
        disparate_impact,
        statistical_parity_difference,
    )

    sensitive_ref = df_reference[sensitive_col].values
    sensitive_cur = df_current[sensitive_col].values

    # Compute metrics for reference
    spd_ref = statistical_parity_difference(y_pred_ref, sensitive_ref, privileged_value)
    di_ref = disparate_impact(y_pred_ref, sensitive_ref, privileged_value)

    # Compute metrics for current
    spd_cur = statistical_parity_difference(y_pred_cur, sensitive_cur, privileged_value)
    di_cur = disparate_impact(y_pred_cur, sensitive_cur, privileged_value)

    return {
        "spd_reference": spd_ref,
        "spd_current": spd_cur,
        "spd_drift": spd_cur - spd_ref,
        "di_reference": di_ref,
        "di_current": di_cur,
        "di_drift": (di_cur - di_ref if di_ref is not None and di_cur is not None else None),
    }


def generate_drift_report(
    df_reference: pd.DataFrame,
    df_current: pd.DataFrame,
    y_pred_ref: np.ndarray | None = None,
    y_pred_cur: np.ndarray | None = None,
    sensitive_col: str = "sex",
    privileged_value: str = "Male",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Generate a comprehensive drift report.

    Parameters
    ----------
    df_reference : pd.DataFrame
        Reference data.
    df_current : pd.DataFrame
        Current data.
    y_pred_ref : np.ndarray, optional
        Reference predictions (for fairness drift).
    y_pred_cur : np.ndarray, optional
        Current predictions (for fairness drift).
    sensitive_col : str
        Sensitive attribute column.
    privileged_value : str
        Privileged group value.
    output_path : str | Path, optional
        Path to save report.

    Returns
    -------
    dict[str, Any]
        Complete drift report.
    """
    from datetime import datetime

    report = {
        "generated_at": datetime.now().isoformat(),
        "reference_samples": len(df_reference),
        "current_samples": len(df_current),
    }

    # Feature drift
    feature_drift = compute_feature_drift(df_reference, df_current)
    report["feature_drift"] = feature_drift

    # Identify drifted features
    drifted_features = [
        name for name, metrics in feature_drift.items() if metrics.get("psi", 0) > 0.1
    ]
    report["drifted_features"] = drifted_features

    # Fairness drift (if predictions provided)
    if y_pred_ref is not None and y_pred_cur is not None:
        fairness_drift = compute_fairness_drift(
            df_reference,
            df_current,
            y_pred_ref,
            y_pred_cur,
            sensitive_col,
            privileged_value,
        )
        report["fairness_drift"] = fairness_drift

    # Summary
    n_drifted = len(drifted_features)
    total_features = len(feature_drift)
    report["summary"] = {
        "total_features_checked": total_features,
        "features_with_drift": n_drifted,
        "drift_percentage": (n_drifted / total_features * 100 if total_features > 0 else 0),
    }

    # Save if path provided
    if output_path:
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Drift report saved to: {output_path}")

    return report


def generate_markdown_drift_report(
    report: dict[str, Any],
    output_path: str | Path | None = None,
) -> str:
    """
    Generate Markdown drift report.

    Parameters
    ----------
    report : dict[str, Any]
        Drift report dictionary.
    output_path : str | Path, optional
        Path to save Markdown file.

    Returns
    -------
    str
        Markdown content.
    """
    lines = []

    lines.append("# Data Drift Report")
    lines.append("")
    lines.append(f"**Generated**: {report.get('generated_at', 'N/A')}")
    lines.append(f"**Reference samples**: {report.get('reference_samples', 'N/A')}")
    lines.append(f"**Current samples**: {report.get('current_samples', 'N/A')}")
    lines.append("")

    # Summary
    summary = report.get("summary", {})
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Features checked: {summary.get('total_features_checked', 0)}")
    lines.append(f"- Features with drift: {summary.get('features_with_drift', 0)}")
    lines.append(f"- Drift percentage: {summary.get('drift_percentage', 0):.1f}%")
    lines.append("")

    # Drifted features
    drifted = report.get("drifted_features", [])
    if drifted:
        lines.append("## Drifted Features")
        lines.append("")
        lines.append("| Feature | PSI | KS Statistic | Mean Shift |")
        lines.append("|---------|-----|--------------|------------|")
        for name in drifted:
            metrics = report.get("feature_drift", {}).get(name, {})
            lines.append(
                f"| {name} | {metrics.get('psi', 0):.4f} | "
                f"{metrics.get('ks_statistic', 0):.4f} | "
                f"{metrics.get('mean_shift', 0):.4f} |"
            )
        lines.append("")

    # Fairness drift
    fairness = report.get("fairness_drift", {})
    if fairness:
        lines.append("## Fairness Drift")
        lines.append("")
        lines.append("| Metric | Reference | Current | Drift |")
        lines.append("|--------|-----------|---------|-------|")
        lines.append(
            f"| SPD | {fairness.get('spd_reference', 0):.4f} | "
            f"{fairness.get('spd_current', 0):.4f} | "
            f"{fairness.get('spd_drift', 0):.4f} |"
        )
        di_ref = fairness.get("di_reference", "N/A")
        di_cur = fairness.get("di_current", "N/A")
        di_drift = fairness.get("di_drift", "N/A")
        lines.append(
            f"| DI | {di_ref if di_ref else 'N/A':.4f} | "
            f"{di_cur if di_cur else 'N/A':.4f} | "
            f"{di_drift if di_drift else 'N/A'} |"
        )
        lines.append("")

    content = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(content)
        print(f"Markdown report saved to: {output_path}")

    return content
