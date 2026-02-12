"""
Fairness Project CLI

Command-line interface for the fairness-aware candidate pre-screening system.

Usage:
    fairness train --model {lr,rf,xgb} [--seed 42]
    fairness evaluate --run-id <id>
    fairness mitigate eo --sensitive sex
    fairness plot --metrics-path <path>
    fairness predict --model-path <path> --input-csv <file> --output-csv <file>
    fairness data download
    fairness data preprocess
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# Try to use Typer if available, fall back to argparse
try:
    import typer
    from rich.console import Console

    HAS_TYPER = True
    app = typer.Typer(
        name="fairness",
        help="Fairness-aware candidate pre-screening CLI",
        add_completion=False,
    )
    data_app = typer.Typer(help="Data management commands")
    app.add_typer(data_app, name="data")
    console = Console()
except ImportError:
    HAS_TYPER = False
    app = None
    console = None


def _print(msg: str, style: str = "") -> None:
    """Print with optional rich formatting."""
    if console:
        console.print(msg, style=style)
    else:
        print(msg)


def _error(msg: str) -> None:
    """Print error message."""
    _print(f"Error: {msg}", "bold red")


def _success(msg: str) -> None:
    """Print success message."""
    _print(f"{msg}", "bold green")


def _load_config(config_path: str | None) -> None:
    """Load configuration if a path is provided."""
    if config_path:
        from fairness_project.config import init_config

        init_config(config_path)


# ============================================================================
# TRAIN COMMAND
# ============================================================================


def train_command(
    model: str = "xgb",
    seed: int = 42,
    data_path: str = "data/processed/adult/adult_model_ready.csv",
    output_dir: str | None = None,
) -> None:
    """Train a model with fairness evaluation.

    Args:
        model: Model type (lr, rf, or xgb)
        seed: Random seed for reproducibility
        data_path: Path to processed data
        output_dir: Directory for output artifacts
    """
    _print(f"Training {model.upper()} model with seed={seed}")
    _print(f"Data: {data_path}")

    try:
        # Import here to avoid circular imports
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        from src.main import main as run_pipeline

        run_pipeline(step="1", data_path=data_path, generate_plots_flag=False)
        _success("Training completed successfully!")
    except Exception as e:
        _error(f"Training failed: {e}")
        raise


# ============================================================================
# EVALUATE COMMAND
# ============================================================================


def evaluate_command(
    run_id: str | None = None,
    split: str = "test",
    data_path: str = "data/processed/adult/adult_model_ready.csv",
) -> None:
    """Evaluate a trained model.

    Args:
        run_id: Run ID to evaluate (optional, uses latest if not provided)
        split: Data split to evaluate on
        data_path: Path to processed data
    """
    _print(f"Evaluating run_id={run_id or 'latest'} on {split} split")

    try:
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        from src.main import main as run_pipeline

        run_pipeline(step="all", data_path=data_path, generate_plots_flag=False)
        _success("Evaluation completed!")
    except Exception as e:
        _error(f"Evaluation failed: {e}")
        raise


# ============================================================================
# MITIGATE COMMAND
# ============================================================================


def mitigate_command(
    technique: str = "eo",
    sensitive: str = "sex",
    run_id: str | None = None,
    data_path: str = "data/processed/adult/adult_model_ready.csv",
) -> None:
    """Apply fairness mitigation technique.

    Args:
        technique: Mitigation technique (eo for equal opportunity)
        sensitive: Sensitive attribute to mitigate on
        run_id: Run ID to apply mitigation to
        data_path: Path to processed data
    """
    _print(f"Applying {technique.upper()} mitigation on {sensitive}")

    if technique != "eo":
        _error(f"Unknown technique: {technique}. Currently only 'eo' is supported.")
        return

    try:
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        from src.main import main as run_pipeline

        run_pipeline(step="2", data_path=data_path, generate_plots_flag=False)
        _success("Mitigation completed!")
    except Exception as e:
        _error(f"Mitigation failed: {e}")
        raise


# ============================================================================
# PLOT COMMAND
# ============================================================================


def plot_command(
    metrics_path: str = "data/metrics/step2_metrics.csv",
    output_dir: str = "data/plots/step2",
) -> None:
    """Generate visualization plots.

    Args:
        metrics_path: Path to metrics CSV
        output_dir: Directory to save plots
    """
    _print(f"Generating plots from {metrics_path}")

    try:
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        from src.plots.plot_step2 import main as plot_main

        plot_main()
        _success(f"Plots saved to {output_dir}")
    except Exception as e:
        _error(f"Plot generation failed: {e}")
        raise


# ============================================================================
# DATA COMMANDS
# ============================================================================


def data_download_command(
    output_dir: str = "data/raw/adult",
) -> None:
    """Download the Adult dataset.

    Args:
        output_dir: Directory to save raw data
    """
    _print(f"Downloading Adult dataset to {output_dir}")

    try:
        from fairness_project.data.download import download_adult_dataset

        download_adult_dataset(output_dir=output_dir)
        _success("Download completed!")
    except Exception as e:
        _error(f"Download failed: {e}")
        raise


def data_preprocess_command(
    input_path: str = "data/raw/adult",
    output_path: str = "data/processed/adult/adult_model_ready.csv",
) -> None:
    """Preprocess the dataset.

    Args:
        input_path: Path to raw data directory
        output_path: Path to save processed data
    """
    _print(f"Preprocessing data from {input_path}")

    try:
        from fairness_project.data.preprocess import prepare_model_ready_data

        input_dir = Path(input_path)
        train_path = input_dir / "adult.data"
        test_path = input_dir / "adult.test"

        prepare_model_ready_data(
            train_path=str(train_path),
            test_path=str(test_path),
            output_path=output_path,
        )
        _success(f"Preprocessed data saved to {output_path}")
    except Exception as e:
        _error(f"Preprocessing failed: {e}")
        raise


# ============================================================================
# PREDICT COMMAND
# ============================================================================


def predict_command(
    model_path: str,
    input_csv: str,
    output_csv: str,
) -> None:
    """Run batch prediction on a CSV file.

    Args:
        model_path: Path to saved model
        input_csv: Path to input CSV file
        output_csv: Path to save predictions
    """
    _print(f"Loading model from {model_path}")
    _print(f"Processing {input_csv}")

    try:
        import joblib

        from fairness_project.inference.batch import run_batch_inference

        model = joblib.load(model_path)
        run_batch_inference(model=model, input_path=input_csv, output_path=output_csv)
        _success(f"Predictions saved to {output_csv}")
    except Exception as e:
        _error(f"Prediction failed: {e}")
        raise


# ============================================================================
# TYPER CLI (when available)
# ============================================================================

if HAS_TYPER:

    @app.callback()
    def cli_callback(
        config: str = typer.Option(None, "--config", "-c", help="Path to YAML config file"),
    ) -> None:
        """Fairness-aware candidate pre-screening CLI."""
        _load_config(config)

    @app.command()
    def train(
        model: str = typer.Option("xgb", "--model", "-m", help="Model type: lr, rf, xgb"),
        seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
        data_path: str = typer.Option(
            "data/processed/adult/adult_model_ready.csv",
            "--data-path",
            "-d",
            help="Path to processed data",
        ),
    ) -> None:
        """Train a fairness-aware model."""
        train_command(model=model, seed=seed, data_path=data_path)

    @app.command()
    def evaluate(
        run_id: str = typer.Option(None, "--run-id", "-r", help="Run ID to evaluate"),
        split: str = typer.Option("test", "--split", help="Data split"),
        data_path: str = typer.Option(
            "data/processed/adult/adult_model_ready.csv",
            "--data-path",
            "-d",
        ),
    ) -> None:
        """Evaluate a trained model."""
        evaluate_command(run_id=run_id, split=split, data_path=data_path)

    @app.command()
    def mitigate(
        technique: str = typer.Argument("eo", help="Mitigation technique"),
        sensitive: str = typer.Option("sex", "--sensitive", "-s"),
        run_id: str = typer.Option(None, "--run-id", "-r"),
        data_path: str = typer.Option(
            "data/processed/adult/adult_model_ready.csv",
            "--data-path",
            "-d",
        ),
    ) -> None:
        """Apply fairness mitigation."""
        mitigate_command(
            technique=technique,
            sensitive=sensitive,
            run_id=run_id,
            data_path=data_path,
        )

    @app.command()
    def plot(
        metrics_path: str = typer.Option(
            "data/metrics/step2_metrics.csv",
            "--metrics-path",
            "-m",
        ),
        output_dir: str = typer.Option("data/plots/step2", "--output-dir", "-o"),
    ) -> None:
        """Generate visualization plots."""
        plot_command(metrics_path=metrics_path, output_dir=output_dir)

    @app.command()
    def predict(
        model_path: str = typer.Option(..., "--model-path", "-m", help="Path to model"),
        input_csv: str = typer.Option(..., "--input-csv", "-i", help="Input CSV"),
        output_csv: str = typer.Option(..., "--output-csv", "-o", help="Output CSV"),
    ) -> None:
        """Run batch prediction."""
        predict_command(model_path=model_path, input_csv=input_csv, output_csv=output_csv)

    @data_app.command("download")
    def download(
        output_dir: str = typer.Option("data/raw/adult", "--output-dir", "-o"),
    ) -> None:
        """Download the Adult dataset."""
        data_download_command(output_dir=output_dir)

    @data_app.command("preprocess")
    def preprocess(
        input_path: str = typer.Option("data/raw/adult", "--input-path", "-i"),
        output_path: str = typer.Option(
            "data/processed/adult/adult_model_ready.csv",
            "--output-path",
            "-o",
        ),
    ) -> None:
        """Preprocess the dataset."""
        data_preprocess_command(input_path=input_path, output_path=output_path)


# ============================================================================
# ARGPARSE CLI (fallback)
# ============================================================================


def build_argparse_cli() -> argparse.ArgumentParser:
    """Build argparse CLI for when Typer is not available."""
    parser = argparse.ArgumentParser(
        prog="fairness",
        description="Fairness-aware candidate pre-screening CLI",
    )
    parser.add_argument("--config", "-c", default=None, help="Path to YAML config file")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--model", "-m", default="xgb", choices=["lr", "rf", "xgb"])
    train_parser.add_argument("--seed", "-s", type=int, default=42)
    train_parser.add_argument(
        "--data-path", "-d", default="data/processed/adult/adult_model_ready.csv"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model")
    eval_parser.add_argument("--run-id", "-r", default=None)
    eval_parser.add_argument("--split", default="test")
    eval_parser.add_argument(
        "--data-path", "-d", default="data/processed/adult/adult_model_ready.csv"
    )

    # Mitigate command
    mit_parser = subparsers.add_parser("mitigate", help="Apply fairness mitigation")
    mit_parser.add_argument("technique", default="eo", nargs="?")
    mit_parser.add_argument("--sensitive", "-s", default="sex")
    mit_parser.add_argument("--run-id", "-r", default=None)
    mit_parser.add_argument(
        "--data-path", "-d", default="data/processed/adult/adult_model_ready.csv"
    )

    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Generate plots")
    plot_parser.add_argument("--metrics-path", "-m", default="data/metrics/step2_metrics.csv")
    plot_parser.add_argument("--output-dir", "-o", default="data/plots/step2")

    # Predict command
    pred_parser = subparsers.add_parser("predict", help="Batch prediction")
    pred_parser.add_argument("--model-path", "-m", required=True)
    pred_parser.add_argument("--input-csv", "-i", required=True)
    pred_parser.add_argument("--output-csv", "-o", required=True)

    # Data subcommands
    data_parser = subparsers.add_parser("data", help="Data management")
    data_sub = data_parser.add_subparsers(dest="data_command")

    dl_parser = data_sub.add_parser("download", help="Download dataset")
    dl_parser.add_argument("--output-dir", "-o", default="data/raw/adult")

    prep_parser = data_sub.add_parser("preprocess", help="Preprocess dataset")
    prep_parser.add_argument("--input-path", "-i", default="data/raw/adult")
    prep_parser.add_argument(
        "--output-path", "-o", default="data/processed/adult/adult_model_ready.csv"
    )

    return parser


def run_argparse_cli() -> None:
    """Run the argparse-based CLI."""
    parser = build_argparse_cli()
    args = parser.parse_args()

    # Load config if provided
    _load_config(getattr(args, "config", None))

    if args.command == "train":
        train_command(
            model=args.model,
            seed=args.seed,
            data_path=args.data_path,
        )
    elif args.command == "evaluate":
        evaluate_command(
            run_id=args.run_id,
            split=args.split,
            data_path=args.data_path,
        )
    elif args.command == "mitigate":
        mitigate_command(
            technique=args.technique,
            sensitive=args.sensitive,
            run_id=args.run_id,
            data_path=args.data_path,
        )
    elif args.command == "plot":
        plot_command(
            metrics_path=args.metrics_path,
            output_dir=args.output_dir,
        )
    elif args.command == "predict":
        predict_command(
            model_path=args.model_path,
            input_csv=args.input_csv,
            output_csv=args.output_csv,
        )
    elif args.command == "data":
        if args.data_command == "download":
            data_download_command(output_dir=args.output_dir)
        elif args.data_command == "preprocess":
            data_preprocess_command(
                input_path=args.input_path,
                output_path=args.output_path,
            )
        else:
            parser.print_help()
    else:
        parser.print_help()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main() -> None:
    """Main CLI entry point."""
    if HAS_TYPER and app is not None:
        app()
    else:
        run_argparse_cli()


if __name__ == "__main__":
    main()
