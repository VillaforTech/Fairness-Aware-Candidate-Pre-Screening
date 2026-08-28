"""
Leakage-Free Fairness Pipeline

This script implements the CORRECT evaluation protocol for Equal Opportunity
post-processing, avoiding the test set leakage in the original implementation.

Key difference from src/main.py:
- Creates train/val/test split (instead of train/test only)
- EO thresholds are tuned on VALIDATION set only
- Final metrics are reported on TEST set

Usage:
    python src/main_leakage_free.py [--val-ratio 0.15] [--seed 42]

This demonstrates proper experimental methodology for fairness mitigation.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.fairness_project.fairness.postprocess import (
    apply_thresholds,
    compute_tpr,
    tune_equal_opportunity,
)
from src.metrics.fairness import compute_fairness_metrics
from src.models.train_lr import train_lr
from src.models.train_rf import train_rf
from src.preprocessing.preprocess import build_preprocessing_pipeline
from src.utils.utils import save_predictions

# XGBoost is optional
try:
    from src.models.train_xgb import train_xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    train_xgb = None


# =====================================================================
# METRIC COLLECTION
# =====================================================================

all_metrics = []


def record_metrics(model_name, condition, y_true, y_pred, sensitive, tpr_gap=None):
    """Record metrics for comparison."""
    fair = compute_fairness_metrics(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive,
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


# =====================================================================
# DATA LOADING
# =====================================================================


def load_and_split_data(data_path, val_ratio=0.15, random_state=42):
    """
    Load data and create train/val/test split.

    The key difference from the original: we create a VALIDATION split
    from the training data for threshold tuning.
    """
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    df["race_binary"] = df["race"].apply(lambda r: "White" if r == "White" else "Non-White")

    # Create validation split from training data
    np.random.seed(random_state)
    train_mask = df["split"] == "train"
    train_indices = df[train_mask].index.tolist()

    n_val = int(len(train_indices) * val_ratio)
    val_indices = np.random.choice(train_indices, size=n_val, replace=False)
    df.loc[val_indices, "split"] = "val"

    # Split dataframes
    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()

    print(f"Train samples: {len(df_train)}")
    print(f"Val samples:   {len(df_val)}")
    print(f"Test samples:  {len(df_test)}")

    # Create X, y
    drop_cols = ["income", "sex", "race", "race_binary", "split"]

    y_train = (df_train["income"] == ">50K").astype(int)
    y_val = (df_val["income"] == ">50K").astype(int)
    y_test = (df_test["income"] == ">50K").astype(int)

    X_train = df_train.drop(columns=drop_cols)
    X_val = df_val.drop(columns=drop_cols)
    X_test = df_test.drop(columns=drop_cols)

    return (df_train, df_val, df_test, X_train, X_val, X_test, y_train, y_val, y_test)


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================


def compute_tpr_gap(y_true, y_pred, sensitive):
    """Compute TPR gap between Male and Female."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    male_mask = sensitive == "Male"
    female_mask = sensitive == "Female"

    tpr_male = compute_tpr(y_true[male_mask], y_pred[male_mask])
    tpr_female = compute_tpr(y_true[female_mask], y_pred[female_mask])

    return tpr_male, tpr_female, tpr_male - tpr_female


def print_metrics(label, y_true, y_pred, sensitive):
    """Print performance and fairness metrics."""
    print(f"\n--- {label} ---")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_true, y_pred):.4f}")

    fair = compute_fairness_metrics(y_true, y_pred, sensitive, "Male")
    print(f"SPD      : {fair['SPD']:.4f}")
    print(f"DI       : {fair['DI']:.4f}")

    tpr_m, tpr_f, gap = compute_tpr_gap(y_true, y_pred, sensitive)
    print(f"TPR Male : {tpr_m:.4f}")
    print(f"TPR Female: {tpr_f:.4f}")
    print(f"TPR Gap  : {gap:.4f}")

    return gap


# =====================================================================
# MAIN PIPELINE
# =====================================================================


def run_leakage_free_pipeline(data_path, val_ratio=0.15, seed=42):
    """
    Run the complete leakage-free fairness pipeline.

    Protocol:
    1. Split data into train/val/test
    2. Train models on train set
    3. Tune EO thresholds on VALIDATION set (not test!)
    4. Apply thresholds to TEST set
    5. Report final metrics on TEST set
    """
    global all_metrics
    all_metrics = []

    print("\n" + "=" * 70)
    print(" LEAKAGE-FREE FAIRNESS PIPELINE ")
    print("=" * 70)
    print("\nThis pipeline ensures EO thresholds are tuned on validation data,")
    print("NOT on test data, to prevent information leakage.\n")

    # Load and split data
    (df_train, df_val, df_test, X_train, X_val, X_test, y_train, y_val, y_test) = (
        load_and_split_data(data_path, val_ratio, seed)
    )

    # Build preprocessing pipeline
    preprocess, _, _ = build_preprocessing_pipeline(df_train)

    # Get sensitive attributes
    sensitive_val = df_val["sex"].values
    sensitive_test = df_test["sex"].values

    # Train and evaluate each model
    models = {}
    model_configs = [("lr", "Logistic Regression", train_lr)]
    model_configs.append(("rf", "Random Forest", train_rf))
    if HAS_XGBOOST:
        model_configs.append(("xgb", "XGBoost", train_xgb))

    for key, name, train_fn in model_configs:
        print(f"\n{'=' * 70}")
        print(f" {name.upper()} ")
        print("=" * 70)

        # Train model
        print(f"\nTraining {name}...")
        model = train_fn(preprocess, X_train, y_train)
        models[key] = model

        # Get predictions and probabilities
        y_proba_val = model.predict_proba(X_val)[:, 1]
        y_proba_test = model.predict_proba(X_test)[:, 1]

        # ============================================================
        # BASELINE (no EO adjustment)
        # ============================================================
        print("\n[BASELINE - threshold 0.5 for all]")
        y_pred_base_test = (y_proba_test >= 0.5).astype(int)

        gap_base = print_metrics(
            f"{name} BASELINE (Test Set)", y_test, y_pred_base_test, sensitive_test
        )
        record_metrics(key.upper(), "Before EO", y_test, y_pred_base_test, sensitive_test, gap_base)

        save_predictions(f"{key}_leakage_free_base", y_test, y_pred_base_test, X_test.index)

        # ============================================================
        # TUNE EO ON VALIDATION (not test!)
        # ============================================================
        print("\n[EO TUNING - using validation set ONLY]")

        # Tune thresholds on validation set
        eo_params = tune_equal_opportunity(
            y_val=y_val.values,
            y_proba_val=y_proba_val,
            sensitive_val=sensitive_val,
        )

        print("Learned thresholds from VALIDATION:")
        print(f"  Threshold (Male)  : {eo_params['threshold_priv']:.3f}")
        print(f"  Threshold (Female): {eo_params['threshold_unpriv']:.3f}")
        print(f"  Val TPR Male      : {eo_params['tpr_priv_val']:.4f}")
        print(f"  Val TPR Female (before): {eo_params['tpr_unpriv_before_val']:.4f}")
        print(f"  Val TPR Female (after) : {eo_params['tpr_unpriv_after_val']:.4f}")

        # ============================================================
        # APPLY TO TEST (without using test labels!)
        # ============================================================
        print("\n[APPLY TO TEST - no test labels used]")

        # Apply learned thresholds to test set
        y_pred_eo_test = apply_thresholds(
            y_pred_proba=y_proba_test,
            sensitive_attr=sensitive_test,
            threshold_priv=eo_params["threshold_priv"],
            threshold_unpriv=eo_params["threshold_unpriv"],
        )

        gap_eo = print_metrics(
            f"{name} EO-ADJUSTED (Test Set)", y_test, y_pred_eo_test, sensitive_test
        )
        record_metrics(key.upper(), "After EO", y_test, y_pred_eo_test, sensitive_test, gap_eo)

        save_predictions(f"{key}_leakage_free_eo", y_test, y_pred_eo_test, X_test.index)

        # Summary for this model
        print(f"\n{name} Summary:")
        print(f"  TPR Gap BEFORE EO: {gap_base:.4f}")
        print(f"  TPR Gap AFTER EO : {gap_eo:.4f}")
        print(f"  Improvement      : {abs(gap_base) - abs(gap_eo):.4f}")

    # Save all metrics
    metrics_df = pd.DataFrame(all_metrics)
    output_path = Path("data/metrics/leakage_free_metrics.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print(" RESULTS SUMMARY (Leakage-Free Protocol) ")
    print("=" * 70)
    print(f"\nMetrics saved to: {output_path}")
    print("\n" + metrics_df.to_string(index=False))

    print("\n" + "=" * 70)
    print(" KEY DIFFERENCE FROM ORIGINAL ")
    print("=" * 70)
    print(
        """
In the ORIGINAL implementation:
- EO thresholds were tuned using TEST set labels
- This is DATA LEAKAGE - thresholds are optimized for the same data
  they're evaluated on, leading to overly optimistic fairness results

In this LEAKAGE-FREE implementation:
- EO thresholds are tuned using VALIDATION set labels only
- TEST set labels are never seen during threshold selection
- Results avoid threshold-selection leakage on this split, but this
  experiment does not establish external validity or generalization
"""
    )

    return models, metrics_df


def main():
    parser = argparse.ArgumentParser(description="Leakage-free fairness evaluation pipeline")
    parser.add_argument(
        "--data-path",
        default="data/processed/adult/adult_model_ready.csv",
        help="Path to processed data",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Ratio of training data to use for validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    run_leakage_free_pipeline(
        data_path=args.data_path,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
