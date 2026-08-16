"""Integration tests for the FastAPI churn-prediction service."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="Model artifact not found; run `python -m src.models.train_model` first"
)

VALID_PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.4,
    "TotalCharges": 427.0,
}


@pytest.fixture(scope="module")
def client():
    from api.main import app

    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model_name" in body


def test_predict_valid_payload(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in {"Yes", "No"}
    assert body["risk_tier"] in {"Low", "Medium", "High"}


def test_predict_rejects_invalid_categorical_value(client):
    bad_payload = {**VALID_PAYLOAD, "gender": "Unknown"}
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422  # Pydantic validation error


def test_predict_rejects_negative_tenure(client):
    bad_payload = {**VALID_PAYLOAD, "tenure": -5}
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_predict_batch(client):
    resp = client.post("/predict/batch", json={"customers": [VALID_PAYLOAD, VALID_PAYLOAD]})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["predictions"]) == 2


def test_predict_batch_rejects_empty_list(client):
    resp = client.post("/predict/batch", json={"customers": []})
    assert resp.status_code == 400


def test_model_metadata_endpoint(client):
    resp = client.get("/model/metadata")
    assert resp.status_code == 200
    body = resp.json()
    assert "best_model" in body
    assert "test_metrics" in body


def test_high_risk_customer_scores_higher_than_low_risk(client):
    low_risk = {
        **VALID_PAYLOAD,
        "tenure": 60,
        "Contract": "Two year",
        "PaymentMethod": "Credit card (automatic)",
        "OnlineSecurity": "Yes",
        "TechSupport": "Yes",
    }
    high_risk = VALID_PAYLOAD  # month-to-month, electronic check, no security/support

    low_proba = client.post("/predict", json=low_risk).json()["churn_probability"]
    high_proba = client.post("/predict", json=high_risk).json()["churn_probability"]
    assert high_proba > low_proba
