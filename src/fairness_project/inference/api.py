"""
FastAPI inference API.

Provides a REST API for model predictions.

Usage:
    uvicorn fairness_project.inference.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Check for FastAPI availability
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


if HAS_FASTAPI:
    # ================================================================
    # Request/Response Models
    # ================================================================

    class PredictionInput(BaseModel):
        """Input schema for a single prediction."""

        age: int = Field(..., ge=0, le=120, description="Age of individual")
        workclass: str = Field(..., description="Type of employment")
        fnlwgt: int = Field(..., ge=0, description="Final weight")
        education: str = Field(..., description="Highest education level")
        education_num: int = Field(..., ge=1, le=20, description="Education as number")
        marital_status: str = Field(..., description="Marital status")
        occupation: str = Field(..., description="Type of occupation")
        relationship: str = Field(..., description="Relationship status")
        native_country: str = Field(..., description="Country of origin")
        capital_gain: int = Field(..., ge=0, description="Capital gains")
        capital_loss: int = Field(..., ge=0, description="Capital losses")
        hours_per_week: int = Field(..., ge=0, le=168, description="Hours worked per week")

        model_config = {
            "json_schema_extra": {
                "example": {
                    "age": 35,
                    "workclass": "Private",
                    "fnlwgt": 200000,
                    "education": "Bachelors",
                    "education_num": 13,
                    "marital_status": "Married-civ-spouse",
                    "occupation": "Exec-managerial",
                    "relationship": "Husband",
                    "native_country": "United-States",
                    "capital_gain": 5000,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                }
            }
        }

    class PredictionOutput(BaseModel):
        """Output schema for prediction."""

        prediction: int = Field(..., description="Predicted class (0=<=50K, 1=>50K)")
        probability: float = Field(..., description="Probability of >50K")
        label: str = Field(..., description="Human-readable prediction")

    class BatchPredictionInput(BaseModel):
        """Input schema for batch predictions."""

        instances: list[PredictionInput]

    class BatchPredictionOutput(BaseModel):
        """Output schema for batch predictions."""

        predictions: list[PredictionOutput]

    class HealthResponse(BaseModel):
        """Health check response."""

        status: str
        model_loaded: bool

    # ================================================================
    # Application
    # ================================================================

    app = FastAPI(
        title="Fairness Project API",
        description="API for fairness-aware candidate pre-screening predictions",
        version="0.1.0",
    )

    # Global model storage
    _model: Any = None

    def get_model() -> Any:
        """Get the loaded model or raise error."""
        global _model
        if _model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Call /load-model first or set MODEL_PATH env var.",
            )
        return _model

    @app.on_event("startup")
    async def startup_event():
        """Load model on startup if MODEL_PATH is set."""
        global _model
        model_path = os.environ.get("MODEL_PATH")
        if model_path and Path(model_path).exists():
            import joblib

            _model = joblib.load(model_path)
            print(f"Model loaded from: {model_path}")

    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(status="healthy", model_loaded=_model is not None)

    @app.post("/load-model")
    async def load_model(model_path: str) -> dict[str, str]:
        """Load a model from path."""
        global _model
        path = Path(model_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")

        import joblib

        _model = joblib.load(path)
        return {"status": "loaded", "model_path": model_path}

    @app.post("/predict", response_model=PredictionOutput)
    async def predict(input_data: PredictionInput) -> PredictionOutput:
        """Make a single prediction."""
        import pandas as pd

        model = get_model()

        # Convert to DataFrame
        df = pd.DataFrame([input_data.model_dump()])

        # Predict
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0, 1]

        return PredictionOutput(
            prediction=int(prediction),
            probability=float(probability),
            label=">50K" if prediction == 1 else "<=50K",
        )

    @app.post("/predict-batch", response_model=BatchPredictionOutput)
    async def predict_batch(input_data: BatchPredictionInput) -> BatchPredictionOutput:
        """Make batch predictions."""
        import pandas as pd

        model = get_model()

        # Convert to DataFrame
        df = pd.DataFrame([inst.model_dump() for inst in input_data.instances])

        # Predict
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]

        results = [
            PredictionOutput(
                prediction=int(pred),
                probability=float(prob),
                label=">50K" if pred == 1 else "<=50K",
            )
            for pred, prob in zip(predictions, probabilities, strict=False)
        ]

        return BatchPredictionOutput(predictions=results)

else:
    # Fallback when FastAPI not available
    app = None

    def get_model():
        raise RuntimeError("FastAPI not installed. Install with: pip install fastapi uvicorn")


def main() -> None:
    """Run the API server."""
    if not HAS_FASTAPI:
        print("Error: FastAPI not installed.")
        print("Install with: pip install 'fairness-project[api]'")
        return

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
