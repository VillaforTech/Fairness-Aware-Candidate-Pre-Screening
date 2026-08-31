"""Fail-closed batch inference using the same service as the HTTP API."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pandas as pd

from fairness_project.inference.service import InferenceService


def run_batch_inference(
    *,
    service: InferenceService,
    input_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Validate, predict, and atomically write one CSV result."""
    source = Path(input_path)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"Input CSV not found: {source}")

    frame = pd.read_csv(source)
    prediction = service.predict(frame)
    result = frame.copy()
    result["prediction"] = [
        None if int(value) == -1 else int(value) for value in prediction.predictions
    ]
    result["decision"] = prediction.decisions
    result["probability"] = prediction.probabilities
    result["decision_threshold"] = prediction.threshold
    result["decision_policy"] = prediction.policy_id
    result["artifact_id"] = prediction.artifact_id

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        result.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def main() -> None:
    """Standalone batch entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation-only batch simulation")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    run_batch_inference(
        service=InferenceService.from_run(args.run_dir),
        input_path=args.input_csv,
        output_path=args.output_csv,
    )


if __name__ == "__main__":
    main()
