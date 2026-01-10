"""
Unified Fairness Pipeline: Baseline Training + Equal Opportunity Post-Processing

This script combines Step 1 (baseline model training) and Step 2 (EO mitigation)
into a single pipeline. It trains three models (LR, RF, XGB), evaluates them on
performance and fairness metrics, and applies Equal Opportunity post-processing.

Usage:
    python src/main.py [--step {1,2,all}] [--data-path PATH]

Outputs:
    - data/predictions/*.csv        - Model predictions
    - data/metrics/step2_metrics.csv - Comparison metrics
    - data/plots/step2/*.png        - Visualization plots (optional)
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.metrics.fairness import compute_fairness_metrics
from src.models.train_lr import train_lr
from src.models.train_rf import train_rf
from src.preprocessing.preprocess import build_preprocessing_pipeline
from src.techniques.equal_opportunity import equal_opportunity_postprocessing
from src.utils.utils import save_predictions

# XGBoost is optional
try:
    from src.models.train_xgb import train_xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    train_xgb = None
    print("Warning: XGBoost not available. Skipping XGB model.")


# =====================================================================
# GLOBAL METRIC STORE
# =====================================================================

all_metrics = []


# =====================================================================
# HELPERS
# =====================================================================


def record_metrics(model_name, condition, y_true, y_pred, df_test, tpr_gap=None):
    """Record metrics for later comparison."""
    fair = compute_fairness_metrics(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=df_test["sex"],
        privileged_group="Male",
    )
    all_metrics.append(
        {
            "Model": model_name,
            "Condition": condition,
            "Accuracy": accuracy_score(y_true, y_pred),
            "SPD": fair["SPD"],
            "DI": fair["DI"],
            "TPR_gap": tpr_gap,
        }
    )


def print_performance(model_name, y_true, y_pred):
    """Print performance metrics."""
    print(f"\n {model_name} PERFORMANCE ")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1-score :", f1_score(y_true, y_pred))


def print_fairness(model_name, y_true, y_pred, df_test):
    """Print fairness metrics for sex and race."""
    print(f"\n {model_name} FAIRNESS ")

    fairness_sex = compute_fairness_metrics(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=df_test["sex"],
        privileged_group="Male",
    )
    print("Sex fairness :", fairness_sex)

    fairness_race = compute_fairness_metrics(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=df_test["race_binary"],
        privileged_group="White",
    )
    print("Race fairness:", fairness_race)


def compute_tpr_by_sex(y_true, y_pred, df_test):
    """Compute TPR for each sex group."""
    y_true_arr = y_true.to_numpy() if hasattr(y_true, "to_numpy") else y_true

    def tpr(actual, pred):
        mask = actual == 1
        if mask.sum() == 0:
            return 0.0
        return (pred[mask] == 1).mean()

    male_mask = (df_test["sex"] == "Male").to_numpy()
    female_mask = (df_test["sex"] == "Female").to_numpy()

    tpr_male = tpr(y_true_arr[male_mask], y_pred[male_mask])
    tpr_female = tpr(y_true_arr[female_mask], y_pred[female_mask])
    gap = tpr_male - tpr_female

    return tpr_male, tpr_female, gap


def print_tpr(label, tpr_male, tpr_female, gap):
    """Print TPR breakdown by sex."""
    print(f"\n {label} TPR BY SEX ")
    print(f"TPR (Male)   : {tpr_male:.4f}")
    print(f"TPR (Female) : {tpr_female:.4f}")
    print(f"TPR gap (M-F): {gap:.4f}")


# =====================================================================
# LOAD DATA
# =====================================================================


def load_data(data_path="data/processed/adult/adult_model_ready.csv"):
    """Load and prepare the dataset."""
    df = pd.read_csv(data_path)
    df["race_binary"] = df["race"].apply(lambda r: "White" if r == "White" else "Non-White")

    df_train = df[df["split"] == "train"].copy()
    df_test = df[df["split"] == "test"].copy()

    y_train = (df_train["income"] == ">50K").astype(int)
    y_test = (df_test["income"] == ">50K").astype(int)

    drop_cols = ["income", "sex", "race", "race_binary", "split"]
    X_train = df_train.drop(columns=drop_cols)
    X_test = df_test.drop(columns=drop_cols)

    return df_train, df_test, X_train, X_test, y_train, y_test


# =====================================================================
# STEP 1: BASELINE TRAINING
# =====================================================================


def run_step1(X_train, X_test, y_train, y_test, df_test, preprocess, save_prefix=""):
    """
    Step 1: Train baseline models and evaluate performance + fairness.

    Returns trained models and their predictions/probabilities.
    """
    print("\n" + "=" * 70)
    print(" STEP 1: BASELINE MODEL TRAINING & EVALUATION ")
    print("=" * 70)

    # Raw fairness (before any model)
    print("\n FAIRNESS BEFORE MODEL TRAINING (RAW LABELS) ")
    raw_fair_sex = compute_fairness_metrics(
        y_true=y_test, y_pred=y_test, sensitive=df_test["sex"], privileged_group="Male"
    )
    print("Raw Fairness (SEX):", raw_fair_sex)

    raw_fair_race = compute_fairness_metrics(
        y_true=y_test,
        y_pred=y_test,
        sensitive=df_test["race_binary"],
        privileged_group="White",
    )
    print("Raw Fairness (RACE):", raw_fair_race)

    models = {}
    predictions = {}
    probabilities = {}

    # Train Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_clf = train_lr(preprocess, X_train, y_train)
    y_pred_lr = lr_clf.predict(X_test)
    y_proba_lr = lr_clf.predict_proba(X_test)[:, 1]

    save_predictions(f"{save_prefix}lr_base", y_test, y_pred_lr, X_test.index)
    print_performance("Logistic Regression (BASE)", y_test, y_pred_lr)
    print_fairness("Logistic Regression (BASE)", y_test, y_pred_lr, df_test)

    tpr_m, tpr_f, gap = compute_tpr_by_sex(y_test, y_pred_lr, df_test)
    print_tpr("Logistic Regression (BASE)", tpr_m, tpr_f, gap)
    record_metrics("LR", "Before EO", y_test, y_pred_lr, df_test, gap)

    models["lr"] = lr_clf
    predictions["lr"] = y_pred_lr
    probabilities["lr"] = y_proba_lr

    # Train Random Forest
    print("\nTraining Random Forest...")
    rf_clf = train_rf(preprocess, X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    y_proba_rf = rf_clf.predict_proba(X_test)[:, 1]

    save_predictions(f"{save_prefix}rf_base", y_test, y_pred_rf, X_test.index)
    print_performance("Random Forest (BASE)", y_test, y_pred_rf)
    print_fairness("Random Forest (BASE)", y_test, y_pred_rf, df_test)

    tpr_m, tpr_f, gap = compute_tpr_by_sex(y_test, y_pred_rf, df_test)
    print_tpr("Random Forest (BASE)", tpr_m, tpr_f, gap)
    record_metrics("RF", "Before EO", y_test, y_pred_rf, df_test, gap)

    models["rf"] = rf_clf
    predictions["rf"] = y_pred_rf
    probabilities["rf"] = y_proba_rf

    # Train XGBoost (if available)
    if HAS_XGBOOST:
        print("\nTraining XGBoost...")
        xgb_clf = train_xgb(preprocess, X_train, y_train)
        y_pred_xgb = xgb_clf.predict(X_test)
        y_proba_xgb = xgb_clf.predict_proba(X_test)[:, 1]

        save_predictions(f"{save_prefix}xgb_base", y_test, y_pred_xgb, X_test.index)
        print_performance("XGBoost (BASE)", y_test, y_pred_xgb)
        print_fairness("XGBoost (BASE)", y_test, y_pred_xgb, df_test)

        tpr_m, tpr_f, gap = compute_tpr_by_sex(y_test, y_pred_xgb, df_test)
        print_tpr("XGBoost (BASE)", tpr_m, tpr_f, gap)
        record_metrics("XGB", "Before EO", y_test, y_pred_xgb, df_test, gap)

        models["xgb"] = xgb_clf
        predictions["xgb"] = y_pred_xgb
        probabilities["xgb"] = y_proba_xgb
    else:
        print("\nSkipping XGBoost (not installed)...")

    print("\n STEP 1 COMPLETED ")
    return models, predictions, probabilities


# =====================================================================
# STEP 2: EQUAL OPPORTUNITY POST-PROCESSING
# =====================================================================


def run_step2(probabilities, y_test, df_test, X_test, save_prefix=""):
    """
    Step 2: Apply Equal Opportunity post-processing to all models.

    Note: This uses test labels for threshold tuning, which is a known
    leakage issue. A future update will implement proper train/val/test
    split for threshold tuning on validation data only.
    """
    print("\n" + "=" * 70)
    print(" STEP 2: EQUAL OPPORTUNITY POST-PROCESSING ")
    print("=" * 70)

    eo_predictions = {}
    model_names = {"lr": "LR", "rf": "RF", "xgb": "XGB"}
    display_names = {
        "lr": "Logistic Regression",
        "rf": "Random Forest",
        "xgb": "XGBoost",
    }

    for key, y_proba in probabilities.items():
        model_name = model_names[key]
        display_name = display_names[key]

        print(f"\n--- {display_name} + EO ---")

        y_eo, info = equal_opportunity_postprocessing(
            y_true=y_test.values,
            y_pred_proba=y_proba,
            sensitive_attr=df_test["sex"].values,
        )

        save_predictions(f"{save_prefix}{key}_eo", y_test, y_eo, X_test.index)

        print("EO info:", info)
        print_performance(f"{display_name} (EO)", y_test, y_eo)
        print_fairness(f"{display_name} (EO)", y_test, y_eo, df_test)

        tpr_m, tpr_f, gap = compute_tpr_by_sex(y_test, y_eo, df_test)
        print_tpr(f"{display_name} (EO)", tpr_m, tpr_f, gap)
        record_metrics(model_name, "After EO", y_test, y_eo, df_test, gap)

        eo_predictions[key] = y_eo

    print("\n STEP 2 COMPLETED ")
    return eo_predictions


# =====================================================================
# SAVE METRICS & GENERATE PLOTS
# =====================================================================


def save_all_metrics(output_path="data/metrics/step2_metrics.csv"):
    """Save all collected metrics to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(output_path, index=False)

    print(f"\n=== STORED ALL METRICS IN {output_path} ===")
    print(metrics_df)
    return metrics_df


def generate_plots(metrics_path="data/metrics/step2_metrics.csv", output_dir="data/plots/step2"):
    """Generate comparison plots."""
    try:
        from src.plots.plot_step2 import main as plot_main

        plot_main()
    except ImportError:
        print("\nPlot module not available. Skipping plot generation.")


# =====================================================================
# MAIN
# =====================================================================


def main(
    step="all",
    data_path="data/processed/adult/adult_model_ready.csv",
    generate_plots_flag=True,
):
    """
    Run the fairness pipeline.

    Args:
        step: Which step to run ("1", "2", or "all")
        data_path: Path to the processed data file
        generate_plots_flag: Whether to generate plots after Step 2
    """
    global all_metrics
    all_metrics = []  # Reset metrics for fresh run

    # Load data
    print(f"\nLoading data from: {data_path}")
    df_train, df_test, X_train, X_test, y_train, y_test = load_data(data_path)
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Build preprocessing pipeline
    preprocess, _, _ = build_preprocessing_pipeline(df_train)

    models = None
    probabilities = None

    # Run Step 1 if requested
    if step in ["1", "all"]:
        models, predictions, probabilities = run_step1(
            X_train, X_test, y_train, y_test, df_test, preprocess
        )

    # Run Step 2 if requested
    if step in ["2", "all"]:
        if probabilities is None:
            # Need to train models first to get probabilities
            print("\nTraining models to get probabilities for Step 2...")
            models, predictions, probabilities = run_step1(
                X_train, X_test, y_train, y_test, df_test, preprocess
            )

        run_step2(probabilities, y_test, df_test, X_test)

        # Save metrics
        save_all_metrics()

        # Generate plots
        if generate_plots_flag:
            generate_plots()

    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETED ")
    print("=" * 70)

    return models


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fairness Pipeline: Train models and apply EO post-processing"
    )
    parser.add_argument(
        "--step",
        choices=["1", "2", "all"],
        default="all",
        help="Which step to run: 1 (baseline), 2 (EO), or all (default: all)",
    )
    parser.add_argument(
        "--data-path",
        default="data/processed/adult/adult_model_ready.csv",
        help="Path to the processed data file",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(step=args.step, data_path=args.data_path, generate_plots_flag=not args.no_plots)
