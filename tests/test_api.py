"""Tests for the FastAPI inference API."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

try:
    from fastapi.testclient import TestClient

    from fairness_project.inference.api import app

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")


SAMPLE_INPUT = {
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


@pytest.fixture()
def mock_model():
    """Create a mock model that returns predictions."""
    model = MagicMock()
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    return model


@pytest.fixture()
def client(mock_model):
    """Create a test client with a mock model loaded."""
    app.state.model = mock_model
    app.state.metadata = {
        "run_id": "test_run_001",
        "saved_at": "2024-01-01T00:00:00",
        "model_type": "xgb",
    }
    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    def test_health_with_model(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_without_model(self):
        app.state.model = None
        app.state.metadata = {}
        test_client = TestClient(app, raise_server_exceptions=False)
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is False


class TestMetadataEndpoint:
    def test_metadata_returns_model_info(self, client):
        response = client.get("/v1/metadata")
        assert response.status_code == 200
        data = response.json()
        assert data["model_version"] == "test_run_001"
        assert data["model_type"] == "xgb"
        assert data["api_version"] == "v1"

    def test_metadata_empty_when_no_model(self):
        app.state.model = None
        app.state.metadata = {}
        test_client = TestClient(app, raise_server_exceptions=False)
        response = test_client.get("/v1/metadata")
        assert response.status_code == 200
        data = response.json()
        assert data["model_version"] is None
        assert data["api_version"] == "v1"


class TestV1PredictEndpoint:
    def test_predict_success(self, client):
        response = client.post("/v1/predict", json=SAMPLE_INPUT)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 1
        assert data["probability"] == pytest.approx(0.7)
        assert data["label"] == ">50K"

    def test_predict_no_model_returns_503(self):
        app.state.model = None
        app.state.metadata = {}
        test_client = TestClient(app, raise_server_exceptions=False)
        response = test_client.post("/v1/predict", json=SAMPLE_INPUT)
        assert response.status_code == 503

    def test_predict_invalid_input(self, client):
        response = client.post("/v1/predict", json={"age": -1})
        assert response.status_code == 422


class TestV1BatchPredictEndpoint:
    def test_batch_predict_success(self, client, mock_model):
        # Setup mock for batch
        mock_model.predict.return_value = np.array([1, 0])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])

        response = client.post(
            "/v1/predict-batch",
            json={"instances": [SAMPLE_INPUT, SAMPLE_INPUT]},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 2
        assert data["predictions"][0]["prediction"] == 1
        assert data["predictions"][1]["prediction"] == 0
