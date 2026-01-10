"""Plotting functions for model evaluation and fairness analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import seaborn as sns

if TYPE_CHECKING:
    import pandas as pd

sns.set(style="whitegrid")


def plot_metric_comparison(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    save_path: str | Path,
    title: str | None = None,
    x_col: str = "Model",
    hue_col: str = "Condition",
    figsize: tuple[int, int] = (8, 5),
) -> None:
    """
    Create a bar plot comparing a metric across models and conditions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    metric : str
        Column name of the metric to plot.
    ylabel : str
        Y-axis label.
    save_path : str | Path
        Path to save the figure.
    title : str, optional
        Plot title. Defaults to "{metric}: Before vs After Equal Opportunity".
    x_col : str
        Column name for x-axis grouping.
    hue_col : str
        Column name for color grouping.
    figsize : tuple[int, int]
        Figure size.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=df,
        x=x_col,
        y=metric,
        hue=hue_col,
        palette="Set2",
    )

    if title is None:
        title = f"{metric}: Before vs After Equal Opportunity"
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(x_col)

    # Annotate bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", label_type="edge", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved plot → {save_path}")


def plot_step2_metrics(
    metrics_path: str | Path,
    output_dir: str | Path,
) -> None:
    """
    Generate all Step 2 (Equal Opportunity) comparison plots.

    Parameters
    ----------
    metrics_path : str | Path
        Path to step2_metrics.csv file.
    output_dir : str | Path
        Directory to save plots.
    """
    import pandas as pd

    metrics_path = Path(metrics_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_path)

    # 1. Accuracy
    plot_metric_comparison(
        df=df,
        metric="Accuracy",
        ylabel="Accuracy",
        save_path=output_dir / "accuracy_before_after_eo.png",
    )

    # 2. Statistical Parity Difference (SPD)
    plot_metric_comparison(
        df=df,
        metric="SPD",
        ylabel="SPD (Male - Female)",
        save_path=output_dir / "spd_before_after_eo.png",
    )

    # 3. Disparate Impact (DI)
    plot_metric_comparison(
        df=df,
        metric="DI",
        ylabel="DI (Female / Male)",
        save_path=output_dir / "di_before_after_eo.png",
    )

    # 4. TPR Gap
    if "TPR_gap" in df.columns:
        plot_metric_comparison(
            df=df,
            metric="TPR_gap",
            ylabel="TPR Gap (Male - Female)",
            save_path=output_dir / "tprgap_before_after_eo.png",
        )

    print("\nAll plots generated successfully.")


def main() -> None:
    """Generate plots from default paths (backward compatibility)."""
    from fairness_project.utils.paths import get_metrics_dir, get_plots_dir

    metrics_path = get_metrics_dir() / "step2_metrics.csv"
    output_dir = get_plots_dir() / "step2"

    plot_step2_metrics(metrics_path, output_dir)


if __name__ == "__main__":
    main()
