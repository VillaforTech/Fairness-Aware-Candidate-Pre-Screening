"""
Batch inference module.

Provides functionality to run predictions on CSV files using saved models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def validate_input_data(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
) -> list[str]:
    """
    Validate input data for inference.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    required_columns : list[str], optional
        Required column names.

    Returns
    -------
    list[str]
        List of validation errors.
    """
    from fairness_project.data.schema import validate_inference_input

    return validate_inference_input(df, required_columns)


def run_batch_inference(
    model: Any,
    input_path: str | Path,
    output_path: str | Path,
    include_probabilities: bool = True,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Run batch inference on a CSV file.

    Parameters
    ----------
    model : Any
        Trained model with predict and predict_proba methods.
    input_path : str | Path
        Path to input CSV file.
    output_path : str | Path
        Path to save predictions.
    include_probabilities : bool
        Include prediction probabilities in output.
    validate : bool
        Validate input data schema.

    Returns
    -------
    pd.DataFrame
        DataFrame with predictions.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Load input data
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows")

    # Validate if requested
    if validate:
        errors = validate_input_data(df)
        if errors:
            print("Warning: Validation errors found:")
            for e in errors:
                print(f"  - {e}")

    # Prepare features (drop any known non-feature columns if present)
    drop_cols = ["income", "sex", "race", "race_binary", "split", "y_true", "y_pred"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]

    # Run predictions
    print("Running predictions...")
    predictions = model.predict(X)

    # Create output dataframe
    result = df.copy()
    result["prediction"] = predictions

    if include_probabilities and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
        result["probability"] = probabilities

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")

    # Summary
    n_positive = (predictions == 1).sum()
    n_negative = (predictions == 0).sum()
    print("\nPrediction summary:")
    print(f"  Positive (>50K): {n_positive} ({n_positive / len(predictions) * 100:.1f}%)")
    print(f"  Negative (<=50K): {n_negative} ({n_negative / len(predictions) * 100:.1f}%)")

    return result


def run_batch_inference_with_registry(
    run_id: str | None = None,
    model_name: str = "model",
    input_path: str | Path = "data/input.csv",
    output_path: str | Path = "data/predictions_output.csv",
    **kwargs,
) -> pd.DataFrame:
    """
    Run batch inference using a model from the registry.

    Parameters
    ----------
    run_id : str, optional
        Run ID to load model from. Uses latest if not provided.
    model_name : str
        Model name in the registry.
    input_path : str | Path
        Path to input CSV file.
    output_path : str | Path
        Path to save predictions.
    **kwargs
        Additional arguments passed to run_batch_inference.

    Returns
    -------
    pd.DataFrame
        DataFrame with predictions.
    """
    from fairness_project.models.registry import load_model

    model = load_model(run_id, model_name=model_name)
    return run_batch_inference(model, input_path, output_path, **kwargs)


def main() -> None:
    """CLI entry point for batch inference."""
    import argparse

    parser = argparse.ArgumentParser(description="Run batch inference")
    parser.add_argument(
        "--model-path",
        "-m",
        required=True,
        help="Path to saved model (.joblib)",
    )
    parser.add_argument(
        "--input-csv",
        "-i",
        required=True,
        help="Path to input CSV",
    )
    parser.add_argument(
        "--output-csv",
        "-o",
        required=True,
        help="Path to save predictions",
    )
    parser.add_argument(
        "--no-proba",
        action="store_true",
        help="Don't include probabilities",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip input validation",
    )
    args = parser.parse_args()

    import joblib

    model = joblib.load(args.model_path)

    run_batch_inference(
        model=model,
        input_path=args.input_csv,
        output_path=args.output_csv,
        include_probabilities=not args.no_proba,
        validate=not args.no_validate,
    )


if __name__ == "__main__":
    main()
