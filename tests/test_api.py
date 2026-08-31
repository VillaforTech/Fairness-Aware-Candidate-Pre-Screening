"""End-to-end API tests using a real serialized Logistic Regression bundle."""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fairness_project.config import Config
from fairness_project.evaluation.model_card import generate_model_card
from fairness_project.experiment import run_experiment
from fairness_project.inference.api import create_app
from fairness_project.inference.batch import run_batch_inference
from fairness_project.inference.service import InferenceService
from fairness_project.models.artifact import (
    ArtifactValidationError,
    load_bundle,
    sha256_file,
    write_json,
)

SAMPLE_INPUT = {
    "age": 35,
    "workclass": "Private",
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
    config = Config()
    config.data.val_size = 0.20
    config.fairness.review_max_automated_error = 1.0
    result = run_experiment(
        data_path=data_path,
        output_dir=workspace / "runs",
        model_type="lr",
        seed=42,
        run_id="api-test",
        bootstrap_samples=0,
        config=config,
    )
    assert config.model.model_type == "xgb"
    assert config.data.val_size == pytest.approx(0.20)
    return InferenceService.from_run(result.run_dir, allow_governance_rejected=True)


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

    metadata = client.get("/v2/metadata").json()
    assert metadata["decision_policy"]["policy_id"] == "global-review-band-v2"
    assert metadata["decision_policy"]["fairness_adjustment_applied"] is False
    assert metadata["evaluation_only"] is True


def test_effective_execution_configuration_is_bound_to_provenance(service) -> None:
    manifest = service.bundle.manifest
    report = service.bundle.report

    assert manifest["model_type"] == "lr"
    assert manifest["resolved_config"]["model"]["model_type"] == "lr"
    assert manifest["resolved_config"]["data"]["val_size"] == pytest.approx(0.20)
    assert report["protocol"]["validation_ratio"] == pytest.approx(0.20)
    dependence = report["results"]["validation_dependence"]
    assert dependence["counts"]["validation_rows"] == 80
    assert dependence["counts"]["overlap_excluded_validation_rows"] <= 80
    assert dependence["overlap_excluded_retuning"]["status"] in {
        "completed",
        "not_estimable",
    }
    oov = manifest["preprocessing"]["categorical_oov_evidence"]
    assert oov["reference_split"] == "train"
    assert set(oov["splits"]) == {"validation", "test"}


def test_real_artifact_prediction_returns_policy_provenance(client) -> None:
    response = client.post("/v2/simulate", json=SAMPLE_INPUT)
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] in (0, 1, None)
    assert payload["decision"] in {
        "auto_negative",
        "auto_positive",
        "manual_review_required",
    }
    assert 0 <= payload["probability"] <= 1
    assert payload["decision_threshold"] == pytest.approx(0.5)
    assert payload["decision_policy"] == "global-review-band-v2"
    assert payload["artifact_id"] == "api-test"


def test_zero_width_review_policy_never_reviews_the_exact_threshold(service, monkeypatch) -> None:
    serving = service.bundle.policy["serving"]
    monkeypatch.setitem(serving, "lower_threshold", 0.5)
    monkeypatch.setitem(serving, "upper_threshold", 0.5)
    monkeypatch.setattr(
        service.bundle.model,
        "predict_proba",
        lambda frame: np.tile(np.array([[0.5, 0.5]]), (len(frame), 1)),
    )

    result = service.predict(pd.DataFrame([SAMPLE_INPUT]))

    assert result.decisions.tolist() == ["auto_positive"]
    assert result.predictions.tolist() == [1]


def test_governance_rejection_requires_an_explicit_research_override(service) -> None:
    with pytest.raises(ArtifactValidationError, match="Governance rejected this artifact"):
        InferenceService.from_run(service.bundle.run_dir)


@pytest.mark.parametrize("field", ["sex", "race", "race_binary", "fnlwgt"])
def test_protected_audit_and_unknown_fields_are_rejected(client, field) -> None:
    response = client.post("/v2/simulate", json={**SAMPLE_INPUT, field: "Male"})
    assert response.status_code == 422


def test_invalid_input_is_rejected(client) -> None:
    response = client.post("/v2/simulate", json={**SAMPLE_INPUT, "age": -1})
    assert response.status_code == 422

    blank = client.post("/v2/simulate", json={**SAMPLE_INPUT, "occupation": "  "})
    assert blank.status_code == 422

    boolean = client.post("/v2/simulate", json={**SAMPLE_INPUT, "age": True})
    assert boolean.status_code == 422

    unseen = client.post(
        "/v2/simulate",
        json={**SAMPLE_INPUT, "occupation": "Category-not-seen-during-training"},
    )
    assert unseen.status_code == 422
    assert "not seen during training" in unseen.json()["detail"]


def test_batch_is_bounded_and_uses_same_policy(client) -> None:
    empty = client.post("/v2/simulate-batch", json={"instances": []})
    assert empty.status_code == 422

    response = client.post(
        "/v2/simulate-batch",
        json={"instances": [SAMPLE_INPUT, SAMPLE_INPUT]},
    )
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    assert {item["decision_policy"] for item in predictions} == {"global-review-band-v2"}


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


def test_bundle_loader_rejects_integrity_rehashed_policy_report_drift(service, tmp_path) -> None:
    copied = tmp_path / "policy-report-drift"
    shutil.copytree(service.bundle.run_dir, copied)
    policy_path = copied / "policy.json"
    manifest_path = copied / "manifest.json"
    policy = json.loads(policy_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    policy["serving"]["lower_threshold"] = 0.0
    policy["serving"]["upper_threshold"] = 1.0
    write_json(policy_path, policy)
    manifest["policy_sha256"] = sha256_file(policy_path)
    write_json(manifest_path, manifest)

    with pytest.raises(ArtifactValidationError, match="selective-review evidence"):
        load_bundle(copied)


def test_bundle_loader_recomputes_governance_after_rehashed_report_tamper(
    service, tmp_path
) -> None:
    copied = tmp_path / "stale-governance"
    shutil.copytree(service.bundle.run_dir, copied)
    report_path = copied / "report.json"
    manifest_path = copied / "manifest.json"
    report = json.loads(report_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    report["results"]["metrics"]["accuracy"] = 0.0
    write_json(report_path, report)
    manifest["report_sha256"] = sha256_file(report_path)
    write_json(manifest_path, manifest)

    with pytest.raises(ArtifactValidationError, match="fresh gate evaluation"):
        load_bundle(copied)


def test_bundle_loader_rejects_non_boolean_governance_even_when_copies_match(
    service, tmp_path
) -> None:
    copied = tmp_path / "malformed-governance"
    shutil.copytree(service.bundle.run_dir, copied)
    report_path = copied / "report.json"
    manifest_path = copied / "manifest.json"
    report = json.loads(report_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    report["governance"]["passed"] = "false"
    manifest["governance"] = report["governance"]
    write_json(report_path, report)
    manifest["report_sha256"] = sha256_file(report_path)
    write_json(manifest_path, manifest)

    with pytest.raises(ArtifactValidationError, match="passed must be Boolean"):
        load_bundle(copied)


def test_bundle_loader_rejects_rotated_categorical_contract(service, tmp_path) -> None:
    copied = tmp_path / "rotated-categorical-contract"
    shutil.copytree(service.bundle.run_dir, copied)
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    categories = manifest["preprocessing"]["categorical_features"]
    manifest["preprocessing"]["categorical_features"] = categories[1:] + categories[:1]
    write_json(manifest_path, manifest)

    with pytest.raises(ArtifactValidationError, match="categorical_features"):
        load_bundle(copied)


def test_bundle_loader_rejects_inverted_positive_class(service, tmp_path) -> None:
    copied = tmp_path / "inverted-positive-class"
    shutil.copytree(service.bundle.run_dir, copied)
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["positive_class"] = 0
    write_json(manifest_path, manifest)

    with pytest.raises(ArtifactValidationError, match="positive_class must be 1"):
        load_bundle(copied)


def test_atomic_json_writer_uses_collision_free_temporary_files(tmp_path) -> None:
    destination = tmp_path / "shared.json"
    payloads = [{"writer": index, "values": [index] * 25} for index in range(20)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        completed = list(pool.map(lambda payload: write_json(destination, payload), payloads))

    assert completed == [destination] * len(payloads)
    assert json.loads(destination.read_text()) in payloads
    assert list(tmp_path.glob(".shared.json.tmp-*")) == []


def test_bundle_loader_rejects_tampered_predictions(service, tmp_path) -> None:
    copied = tmp_path / "tampered-predictions"
    shutil.copytree(service.bundle.run_dir, copied)
    predictions_path = copied / "predictions.csv"
    predictions_path.write_text(predictions_path.read_text() + "tampered\n")
    with pytest.raises(ArtifactValidationError, match="Predictions digest"):
        load_bundle(copied)


def test_bundle_loader_rejects_tampered_audit_html(service, tmp_path) -> None:
    copied = tmp_path / "tampered-audit-html"
    shutil.copytree(service.bundle.run_dir, copied)
    audit_path = copied / "audit.html"
    audit_path.write_text(audit_path.read_text() + "<!-- tampered -->\n")
    with pytest.raises(ArtifactValidationError, match="Audit HTML digest"):
        load_bundle(copied)


def test_bundle_loader_rejects_tampered_monitoring_snapshot(service, tmp_path) -> None:
    copied = tmp_path / "tampered-monitoring"
    shutil.copytree(service.bundle.run_dir, copied)
    monitoring_path = copied / "monitoring.json"
    monitoring = json.loads(monitoring_path.read_text())
    monitoring["row_count"] += 1
    monitoring_path.write_text(json.dumps(monitoring))
    with pytest.raises(ArtifactValidationError, match="Monitoring snapshot digest"):
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
    assert "Exact-feature overlap sensitivity" in card
    assert "Experimental policy gate" in card
    assert service.bundle.manifest["git_commit"] in card
