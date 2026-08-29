"""End-to-end API tests using a real serialized Logistic Regression bundle."""

from __future__ import annotations

import json
import shutil
from importlib import metadata

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fairness_project.evaluation.model_card import generate_model_card
from fairness_project.experiment import run_experiment
from fairness_project.inference.api import create_app
from fairness_project.inference.batch import run_batch_inference
from fairness_project.inference.service import InferenceService
from fairness_project.models.artifact import ArtifactValidationError, load_bundle

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


def _synthetic_adult_frame(rows: int = 480) -> pd.DataFrame:
    records = []
    for index in range(rows):
        sex = "Male" if index % 2 == 0 else "Female"
        race = "White" if (index // 2) % 2 == 0 else "Black"
        income = ">50K" if (index // 4) % 2 == 0 else "<=50K"
        records.append(
            {
                "age": 22 + index % 45,
                "workclass": "Private" if index % 3 else "State-gov",
                "fnlwgt": 100000 + index,
                "education": "Bachelors" if index % 2 else "HS-grad",
                "education_num": 13 if index % 2 else 9,
                "marital_status": "Married-civ-spouse" if index % 2 else "Never-married",
                "occupation": "Exec-managerial" if index % 3 else "Sales",
                "relationship": "Husband" if sex == "Male" else "Wife",
                "race": race,
                "sex": sex,
                "capital_gain": (index % 5) * 100,
                "capital_loss": 0,
                "hours_per_week": 30 + index % 20,
                "native_country": "United-States",
                "income": income,
                "split": "train" if index < 400 else "test",
                "race_binary": "White" if race == "White" else "Non-White",
            }
        )
    return pd.DataFrame(records)


@pytest.fixture(scope="module")
def service(tmp_path_factory) -> InferenceService:
    workspace = tmp_path_factory.mktemp("api-bundle")
    data_path = workspace / "adult.csv"
    _synthetic_adult_frame().to_csv(data_path, index=False)
    result = run_experiment(
        data_path=data_path,
        output_dir=workspace / "runs",
        model_type="lr",
        seed=42,
        run_id="api-test",
        bootstrap_samples=0,
    )
    return InferenceService.from_run(result.run_dir)


@pytest.fixture()
def client(service):
    with TestClient(create_app(service)) as test_client:
        yield test_client


def test_health_is_liveness_even_without_artifact(monkeypatch) -> None:
    monkeypatch.delenv("RUN_DIR", raising=False)
    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ok"}
        response = client.get("/ready")
        assert response.status_code == 503
        assert "RUN_DIR is not set" in response.json()["detail"]


def test_ready_and_metadata_name_the_served_policy(client) -> None:
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json() == {"ready": True, "artifact_id": "api-test"}

    metadata = client.get("/v1/metadata").json()
    assert metadata["decision_policy"]["policy_id"] == "global-threshold-v1"
    assert metadata["decision_policy"]["fairness_adjustment_applied"] is False
    assert metadata["evaluation_only"] is True


def test_real_artifact_prediction_returns_policy_provenance(client) -> None:
    response = client.post("/v1/predict", json=SAMPLE_INPUT)
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] in (0, 1)
    assert 0 <= payload["probability"] <= 1
    assert payload["decision_threshold"] == pytest.approx(0.5)
    assert payload["decision_policy"] == "global-threshold-v1"
    assert payload["artifact_id"] == "api-test"


def test_protected_and_unknown_fields_are_rejected(client) -> None:
    response = client.post("/v1/predict", json={**SAMPLE_INPUT, "sex": "Male"})
    assert response.status_code == 422


def test_invalid_input_is_rejected(client) -> None:
    response = client.post("/v1/predict", json={**SAMPLE_INPUT, "age": -1})
    assert response.status_code == 422

    blank = client.post("/v1/predict", json={**SAMPLE_INPUT, "occupation": "  "})
    assert blank.status_code == 422

    boolean = client.post("/v1/predict", json={**SAMPLE_INPUT, "age": True})
    assert boolean.status_code == 422


def test_batch_is_bounded_and_uses_same_policy(client) -> None:
    empty = client.post("/v1/predict-batch", json={"instances": []})
    assert empty.status_code == 422

    response = client.post(
        "/v1/predict-batch",
        json={"instances": [SAMPLE_INPUT, SAMPLE_INPUT]},
    )
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    assert {item["decision_policy"] for item in predictions} == {"global-threshold-v1"}


def test_batch_validation_fails_before_writing(service, tmp_path) -> None:
    input_path = tmp_path / "invalid.csv"
    output_path = tmp_path / "output.csv"
    pd.DataFrame([{"age": 35}]).to_csv(input_path, index=False)
    with pytest.raises(ValueError, match="missing="):
        run_batch_inference(
            service=service,
            input_path=input_path,
            output_path=output_path,
        )
    assert not output_path.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("age", 121, "age must be"),
        ("education_num", 12.5, "education_num must contain integers"),
        ("occupation", "   ", "occupation cannot be blank"),
    ],
)
def test_batch_and_service_enforce_the_same_value_contract(
    service,
    tmp_path,
    field,
    value,
    message,
) -> None:
    input_path = tmp_path / f"invalid-{field}.csv"
    output_path = tmp_path / f"invalid-{field}-output.csv"
    pd.DataFrame([{**SAMPLE_INPUT, field: value}]).to_csv(input_path, index=False)
    with pytest.raises(ValueError, match=message):
        run_batch_inference(
            service=service,
            input_path=input_path,
            output_path=output_path,
        )
    assert not output_path.exists()


def test_bundle_loader_rejects_tampered_report(service, tmp_path) -> None:
    copied = tmp_path / "tampered"
    shutil.copytree(service.bundle.run_dir, copied)
    report_path = copied / "report.json"
    report = json.loads(report_path.read_text())
    report["metadata"]["seed"] = 99
    report_path.write_text(json.dumps(report))
    with pytest.raises(ArtifactValidationError, match="Report digest"):
        load_bundle(copied)


def test_bundle_loader_rejects_tampered_policy(service, tmp_path) -> None:
    copied = tmp_path / "tampered-policy"
    shutil.copytree(service.bundle.run_dir, copied)
    policy_path = copied / "policy.json"
    policy = json.loads(policy_path.read_text())
    policy["offline_evaluation"]["thresholds"]["unprivileged"] = 0.01
    policy_path.write_text(json.dumps(policy))
    with pytest.raises(ArtifactValidationError, match="Policy digest"):
        load_bundle(copied)


def test_bundle_loader_rejects_runtime_dependency_drift(service, monkeypatch) -> None:
    real_version = metadata.version

    def mismatched_version(package: str) -> str:
        if package == "xgboost":
            return "0.0.0"
        return real_version(package)

    monkeypatch.setattr(
        "fairness_project.models.artifact.importlib.metadata.version",
        mismatched_version,
    )
    with pytest.raises(ArtifactValidationError, match="Runtime dependency mismatch for xgboost"):
        load_bundle(service.bundle.run_dir)


def test_model_card_is_generated_from_the_validated_bundle(service) -> None:
    card = generate_model_card(service.bundle.manifest, service.bundle.report)
    assert "Generated from validated run `api-test`" in card
    assert "Frozen offline thresholds" in card
    assert "Experimental policy gate" in card
    assert service.bundle.manifest["git_commit"] in card
