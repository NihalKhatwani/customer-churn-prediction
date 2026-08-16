"""Tests for the trained model artifact and evaluation metrics.

These tests exercise the *persisted* model artifact produced by
`python -m src.models.train_model`, so they double as a regression check
that the deployed model still meets a minimum quality bar.
"""
from pathlib import Path

import joblib
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists() or not TEST_DATA_PATH.exists(),
    reason="Model artifact not found; run `python -m src.models.train_model` first",
)


@pytest.fixture(scope="module")
def artifact():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def test_df():
    return pd.read_csv(TEST_DATA_PATH)


def test_artifact_has_required_keys(artifact):
    for key in ("pipeline", "model_name", "feature_columns", "cv_roc_auc"):
        assert key in artifact


def test_pipeline_predicts_valid_probabilities(artifact, test_df):
    pipeline = artifact["pipeline"]
    X = test_df.drop(columns=["Churn"])
    proba = pipeline.predict_proba(X)[:, 1]
    assert (proba >= 0).all() and (proba <= 1).all()
    assert len(proba) == len(test_df)


def test_pipeline_predictions_are_binary(artifact, test_df):
    pipeline = artifact["pipeline"]
    X = test_df.drop(columns=["Churn"])
    preds = pipeline.predict(X)
    assert set(preds).issubset({0, 1})


def test_model_meets_minimum_roc_auc_bar(artifact):
    # regression guard: fail CI if a future retrain regresses below a sane floor
    assert artifact["cv_roc_auc"] >= 0.80


def test_model_beats_majority_class_baseline(artifact, test_df):
    from sklearn.metrics import roc_auc_score

    pipeline = artifact["pipeline"]
    X, y = test_df.drop(columns=["Churn"]), test_df["Churn"]
    proba = pipeline.predict_proba(X)[:, 1]
    assert roc_auc_score(y, proba) > 0.75
