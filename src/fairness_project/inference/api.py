"""
FastAPI inference API.

Provides a versioned REST API for model predictions with structured audit logging.

Usage:
    uvicorn fairness_project.inference.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

# Check for FastAPI availability
try:
    from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
    from pydantic import BaseModel, Field

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


# ================================================================
# Structured JSON Logging
# ================================================================


class JSONFormatter(logging.Formatter):
    """Format log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)
        return json.dumps(log_data, default=str)


def _setup_logging() -> logging.Logger:
    """Configure structured logging for the API."""
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger = logging.getLogger("fairness_project.api")
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    return logger


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

    class MetadataResponse(BaseModel):
        """Metadata response for governance and audit."""

        model_version: str | None = None
        trained_at: str | None = None
        model_type: str | None = None
        fairness_metrics: dict[str, Any] = Field(default_factory=dict)
        api_version: str = "v1"

    # ================================================================
    # Dependency Injection
    # ================================================================

    def get_model(request: Request) -> Any:
        """Get the loaded model from app state or raise 503."""
        model = getattr(request.app.state, "model", None)
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Set MODEL_PATH env var.",
            )
        return model

    # ================================================================
    # Lifespan
    # ================================================================

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage application startup and shutdown."""
        logger = _setup_logging()

        # Initialize config if CONFIG_PATH is set
        config_path = os.environ.get("CONFIG_PATH")
        if config_path and Path(config_path).exists():
            from fairness_project.config import init_config

            init_config(config_path)
            logger.info("Configuration loaded from %s", config_path)

        # Load model
        app.state.model = None
        app.state.metadata = {}

        model_path = os.environ.get("MODEL_PATH")
        if model_path and Path(model_path).exists():
            import joblib

            app.state.model = joblib.load(model_path)
            logger.info("Model loaded from %s", model_path)

            # Try to load metadata from registry run.json alongside model
            run_json = Path(model_path).parent / "run.json"
            if run_json.exists():
                with open(run_json) as f:
                    app.state.metadata = json.load(f)
                logger.info(
                    "Metadata loaded, run_id=%s, model_type=%s",
                    app.state.metadata.get("run_id"),
                    app.state.metadata.get("model_type"),
                )

            # Try to load fairness metrics
            metrics_json = Path(model_path).parent / "metrics.json"
            if metrics_json.exists():
                with open(metrics_json) as f:
                    app.state.metadata["fairness_metrics"] = json.load(f)

        else:
            logger.warning("No MODEL_PATH set or file not found; model not loaded")

        yield

        logger.info("Shutting down API")

    # ================================================================
    # Application & Routers
    # ================================================================

    app = FastAPI(
        title="Fairness Project API",
        description="Educational API for the Adult-income classification example",
        version="1.0.0",
        lifespan=lifespan,
    )

    v1_router = APIRouter(prefix="/v1")

    # ================================================================
    # Root-level health check (unversioned)
    # ================================================================

    @app.get("/health", response_model=HealthResponse)
    async def health_check(request: Request) -> HealthResponse:
        """Health check endpoint."""
        model = getattr(request.app.state, "model", None)
        return HealthResponse(status="healthy", model_loaded=model is not None)

    # ================================================================
    # V1 Routes
    # ================================================================

    @v1_router.get("/metadata", response_model=MetadataResponse)
    async def metadata(request: Request) -> MetadataResponse:
        """Return model and governance metadata."""
        meta = getattr(request.app.state, "metadata", {})
        return MetadataResponse(
            model_version=meta.get("run_id"),
            trained_at=meta.get("saved_at") or meta.get("timestamp"),
            model_type=meta.get("model_type"),
            fairness_metrics=meta.get("fairness_metrics", {}),
            api_version="v1",
        )

    @v1_router.post("/predict", response_model=PredictionOutput)
    async def predict(
        input_data: PredictionInput, model: Any = Depends(get_model)
    ) -> PredictionOutput:
        """Make a single prediction."""
        import pandas as pd

        logger = logging.getLogger("fairness_project.api")

        df = pd.DataFrame([input_data.model_dump()])
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0, 1]

        result = PredictionOutput(
            prediction=int(prediction),
            probability=float(probability),
            label=">50K" if prediction == 1 else "<=50K",
        )

        # Audit log (hash input to avoid PII in logs)
        input_hash = sha256(
            json.dumps(input_data.model_dump(), sort_keys=True).encode()
        ).hexdigest()[:12]
        logger.info(
            "prediction",
            extra={
                "extra_data": {
                    "endpoint": "/v1/predict",
                    "input_hash": input_hash,
                    "prediction": result.prediction,
                    "probability": round(result.probability, 4),
                }
            },
        )

        return result

    @v1_router.post("/predict-batch", response_model=BatchPredictionOutput)
    async def predict_batch(
        input_data: BatchPredictionInput, model: Any = Depends(get_model)
    ) -> BatchPredictionOutput:
        """Make batch predictions."""
        import pandas as pd

        logger = logging.getLogger("fairness_project.api")

        df = pd.DataFrame([inst.model_dump() for inst in input_data.instances])
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

        logger.info(
            "batch_prediction",
            extra={
                "extra_data": {
                    "endpoint": "/v1/predict-batch",
                    "batch_size": len(input_data.instances),
                }
            },
        )

        return BatchPredictionOutput(predictions=results)

    # Mount v1 router
    app.include_router(v1_router)

else:
    # Fallback when FastAPI not available
    app = None

    def get_model(request=None):
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
