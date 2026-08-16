"""
FastAPI model-serving application for the Customer Churn Prediction model.

Exposes:
    GET  /health           - liveness + model metadata
    POST /predict           - single-customer churn probability
    POST /predict/batch      - batch scoring for multiple customers
    GET  /model/metadata     - training metrics, hyperparameters, feature importances

Run locally:
    uvicorn api.main:app --reload --port 8000

Interactive docs (Swagger UI) are auto-generated at /docs.
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    HealthResponse,
    PredictionResponse,
)
from src.features.build_features import engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
MODEL_VERSION = "1.0.0"

MODEL_STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model artifact from %s", MODEL_PATH)
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. Run `python -m src.models.train_model` first."
        )
    artifact = joblib.load(MODEL_PATH)
    MODEL_STATE["pipeline"] = artifact["pipeline"]
    MODEL_STATE["model_name"] = artifact["model_name"]
    MODEL_STATE["cv_roc_auc"] = artifact["cv_roc_auc"]
    MODEL_STATE["feature_columns"] = artifact["feature_columns"]
    if METRICS_PATH.exists():
        MODEL_STATE["metrics"] = json.loads(METRICS_PATH.read_text())
    logger.info("Model loaded: %s (CV ROC-AUC=%.4f)", artifact["model_name"], artifact["cv_roc_auc"])
    yield
    MODEL_STATE.clear()


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Production-style ML inference service for predicting telecom customer churn. "
        "Built with FastAPI + scikit-learn + XGBoost; served behind a versioned REST "
        "contract with Pydantic request/response validation."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _risk_tier(prob: float) -> str:
    if prob < 0.33:
        return "Low"
    if prob < 0.66:
        return "Medium"
    return "High"


def _score_customers(customers: list[CustomerFeatures]) -> list[PredictionResponse]:
    raw_df = pd.DataFrame([c.model_dump() for c in customers])
    engineered_df = engineer_features(raw_df)

    missing = set(MODEL_STATE["feature_columns"]) - set(engineered_df.columns)
    if missing:
        raise HTTPException(status_code=500, detail=f"Feature engineering mismatch, missing: {missing}")
    engineered_df = engineered_df[MODEL_STATE["feature_columns"]]

    pipeline = MODEL_STATE["pipeline"]
    probabilities = pipeline.predict_proba(engineered_df)[:, 1]

    return [
        PredictionResponse(
            churn_probability=round(float(p), 4),
            churn_prediction="Yes" if p >= 0.5 else "No",
            risk_tier=_risk_tier(p),
            model_name=MODEL_STATE["model_name"],
            model_version=MODEL_VERSION,
        )
        for p in probabilities
    ]


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
def health() -> HealthResponse:
    """Liveness probe + currently-loaded model metadata."""
    return HealthResponse(
        status="ok",
        model_name=MODEL_STATE["model_name"],
        cv_roc_auc=round(float(MODEL_STATE["cv_roc_auc"]), 4),
    )


@app.get("/model/metadata", tags=["Ops"])
def model_metadata() -> dict:
    """Full training metrics: CV score, held-out test metrics, and the
    3-model comparison table produced by `src/models/train_model.py`."""
    if "metrics" not in MODEL_STATE:
        raise HTTPException(status_code=404, detail="metrics.json not found; run training pipeline")
    return MODEL_STATE["metrics"]


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(customer: CustomerFeatures) -> PredictionResponse:
    """Score a single customer and return churn probability + risk tier."""
    return _score_customers([customer])[0]


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Score a batch of customers in one request (used by the Streamlit dashboard's
    CSV-upload flow and any downstream batch-scoring job)."""
    if not request.customers:
        raise HTTPException(status_code=400, detail="customers list must not be empty")
    return BatchPredictionResponse(predictions=_score_customers(request.customers))


@app.get("/", tags=["Ops"])
def root() -> dict:
    return {"message": "Customer Churn Prediction API. See /docs for interactive Swagger UI."}
