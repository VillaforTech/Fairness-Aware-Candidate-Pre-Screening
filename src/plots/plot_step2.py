from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")


def plot_metric(df, metric, ylabel, save_path):
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=df, x="Model", y=metric, hue="Condition", palette="Set2")

    plt.title(f"{metric}: Before vs After Equal Opportunity")
    plt.ylabel(ylabel)
    plt.xlabel("Model")

    # Annotate bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", label_type="edge", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved plot → {save_path}")


def main():
    metrics_path = Path("data/metrics/step2_metrics.csv")
    df = pd.read_csv(metrics_path)

    output_dir = Path("data/plots/step2")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Accuracy
    plot_metric(
        df=df,
        metric="Accuracy",
        ylabel="Accuracy",
        save_path=output_dir / "accuracy_before_after_eo.png",
    )

    # 2. Statistical Parity Difference (SPD)
    plot_metric(
        df=df,
        metric="SPD",
        ylabel="SPD (Male - Female)",
        save_path=output_dir / "spd_before_after_eo.png",
    )

    # 3. Disparate Impact (DI)
    plot_metric(
        df=df,
        metric="DI",
        ylabel="DI (Female / Male)",
        save_path=output_dir / "di_before_after_eo.png",
    )

    # 4. TPR Gap
    plot_metric(
        df=df,
        metric="TPR_gap",
        ylabel="TPR Gap (Male - Female)",
        save_path=output_dir / "tprgap_before_after_eo.png",
    )

    print("\nAll plots generated successfully.")


if __name__ == "__main__":
    main()
