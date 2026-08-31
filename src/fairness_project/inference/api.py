"""Local FastAPI demo backed by a validated experiment bundle."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from fairness_project.inference.service import InferenceContractError, InferenceService
from fairness_project.models.artifact import ArtifactValidationError


class JSONFormatter(logging.Formatter):
    """Format API events as JSON lines without recording input values or hashes."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra_data = getattr(record, "extra_data", None)
        if isinstance(extra_data, dict):
            payload.update(extra_data)
        return json.dumps(payload, default=str)


def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("fairness_project.api")
    logger.setLevel(getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
    return logger


class PredictionInput(BaseModel):
    """The exact 11-feature scoring contract used by training."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    age: int = Field(..., strict=True, ge=0, le=120)
    workclass: str = Field(..., min_length=1)
    education: str = Field(..., min_length=1)
    education_num: int = Field(..., strict=True, ge=1, le=20)
    marital_status: str = Field(..., min_length=1)
    occupation: str = Field(..., min_length=1)
    relationship: str = Field(..., min_length=1)
    native_country: str = Field(..., min_length=1)
    capital_gain: int = Field(..., strict=True, ge=0)
    capital_loss: int = Field(..., strict=True, ge=0)
    hours_per_week: int = Field(..., strict=True, ge=0, le=168)


class PredictionOutput(BaseModel):
    prediction: int | None
    decision: str
    probability: float
    label: str
    decision_threshold: float
    review_lower_threshold: float
    review_upper_threshold: float
    decision_policy: str
    artifact_id: str


class BatchPredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[PredictionInput] = Field(..., min_length=1, max_length=1000)


class BatchPredictionOutput(BaseModel):
    predictions: list[PredictionOutput]


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool
    artifact_id: str


class MetadataResponse(BaseModel):
    artifact_id: str
    schema_version: str
    model_type: str
    created_at: str
    decision_policy: dict[str, object]
    governance: dict[str, object]
    evaluation_only: bool
    api_version: str = "v2"


def _output_rows(batch) -> list[PredictionOutput]:
    return [
        PredictionOutput(
            prediction=None if int(prediction) == -1 else int(prediction),
            decision=str(decision),
            probability=float(probability),
            label=(
                "manual_review_required"
                if int(prediction) == -1
                else ">50K"
                if int(prediction) == 1
                else "<=50K"
            ),
            decision_threshold=batch.threshold,
            review_lower_threshold=batch.lower_threshold,
            review_upper_threshold=batch.upper_threshold,
            decision_policy=batch.policy_id,
            artifact_id=batch.artifact_id,
        )
        for prediction, decision, probability in zip(
            batch.predictions,
            batch.decisions,
            batch.probabilities,
            strict=True,
        )
    ]


def create_app(initial_service: InferenceService | None = None) -> FastAPI:
    """Create an app that loads exactly one validated run bundle."""
    logger = _setup_logging()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.service = initial_service
        application.state.readiness_error = None
        if application.state.service is None:
            run_dir = os.environ.get("RUN_DIR")
            if not run_dir:
                application.state.readiness_error = "RUN_DIR is not set"
            else:
                try:
                    allow_rejected = os.environ.get("ALLOW_REJECTED_RESEARCH_BUNDLE") == "1"
                    application.state.service = InferenceService.from_run(
                        Path(run_dir),
                        allow_governance_rejected=allow_rejected,
                    )
                except (ArtifactValidationError, OSError, ValueError) as exc:
                    application.state.readiness_error = str(exc)
        if application.state.service is None:
            logger.warning("API not ready: %s", application.state.readiness_error)
        else:
            logger.info(
                "artifact_loaded",
                extra={
                    "extra_data": {
                        "artifact_id": application.state.service.metadata["artifact_id"],
                        "decision_policy": application.state.service.metadata["decision_policy"][
                            "policy_id"
                        ],
                    }
                },
            )
        yield

    application = FastAPI(
        title="Adult Income Audit API",
        description=(
            "Evaluation-only policy simulation for a validated local run bundle. "
            "It uses a global review band and never serves protected-attribute thresholds."
        ),
        version="2.0.0",
        lifespan=lifespan,
    )
    router = APIRouter(prefix="/v2")

    def get_service(request: Request) -> InferenceService:
        service = getattr(request.app.state, "service", None)
        if not isinstance(service, InferenceService):
            reason = getattr(request.app.state, "readiness_error", "artifact unavailable")
            raise HTTPException(status_code=503, detail=f"API is not ready: {reason}")
        return service

    @application.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @application.get("/ready", response_model=ReadyResponse)
    async def ready(service: InferenceService = Depends(get_service)) -> ReadyResponse:
        return ReadyResponse(ready=True, artifact_id=service.metadata["artifact_id"])

    @router.get("/metadata", response_model=MetadataResponse)
    async def metadata(service: InferenceService = Depends(get_service)) -> MetadataResponse:
        return MetadataResponse(**service.metadata, api_version="v2")

    @router.post("/simulate", response_model=PredictionOutput)
    async def simulate(
        input_data: PredictionInput,
        service: InferenceService = Depends(get_service),
    ) -> PredictionOutput:
        try:
            batch = service.predict(pd.DataFrame([input_data.model_dump()]))
        except InferenceContractError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result = _output_rows(batch)[0]
        logger.info(
            "prediction",
            extra={
                "extra_data": {
                    "endpoint": "/v2/simulate",
                    "artifact_id": result.artifact_id,
                    "decision_policy": result.decision_policy,
                    "decision": result.decision,
                }
            },
        )
        return result

    @router.post("/simulate-batch", response_model=BatchPredictionOutput)
    async def simulate_batch(
        input_data: BatchPredictionInput,
        service: InferenceService = Depends(get_service),
    ) -> BatchPredictionOutput:
        frame = pd.DataFrame([instance.model_dump() for instance in input_data.instances])
        try:
            batch = service.predict(frame)
        except InferenceContractError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return BatchPredictionOutput(predictions=_output_rows(batch))

    application.include_router(router)
    return application


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
