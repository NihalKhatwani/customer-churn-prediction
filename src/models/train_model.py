"""
Model training & hyperparameter tuning module.

Trains and compares three classifiers -- Logistic Regression, Random Forest,
and XGBoost -- inside an imbalanced-learn Pipeline (preprocessing -> SMOTE
oversampling -> classifier) to correctly handle the ~27% minority (churn)
class without leaking synthetic samples across CV folds. Each model is tuned
via GridSearchCV (5-fold stratified CV, ROC-AUC scoring). The best overall
model is refit on the full training set, evaluated on the held-out test set,
and persisted as a single deployable artifact.

Usage:
    python -m src.models.train_model
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.features.build_features import TARGET, build_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
COMPARISON_PATH = PROJECT_ROOT / "reports" / "model_comparison.csv"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"

RANDOM_STATE = 42
CV_FOLDS = 5

MODEL_GRID: dict[str, dict] = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "param_grid": {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__solver": ["lbfgs"],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [6, 10, None],
            "clf__min_samples_leaf": [1, 3],
        },
    },
    "xgboost": {
        "estimator": XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        ),
        "param_grid": {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [3, 5],
            "clf__learning_rate": [0.05, 0.1],
        },
    },
}


def build_pipeline(estimator, preprocessor) -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", estimator),
        ]
    )


def evaluate(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }


def main() -> None:
    logger.info("Loading train/test splits")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
    X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    comparison_rows = []
    best_overall = {"name": None, "pipeline": None, "cv_roc_auc": -1.0, "test_metrics": None, "best_params": None}

    for name, spec in MODEL_GRID.items():
        logger.info("=== Tuning %s (%d-fold CV, ROC-AUC) ===", name, CV_FOLDS)
        preprocessor, _, _ = build_preprocessor(train_df)
        pipeline = build_pipeline(spec["estimator"], preprocessor)

        search = GridSearchCV(
            pipeline,
            param_grid=spec["param_grid"],
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        t0 = time.time()
        search.fit(X_train, y_train)
        elapsed = time.time() - t0

        test_metrics = evaluate(search.best_estimator_, X_test, y_test)
        logger.info(
            "%s -> best CV ROC-AUC=%.4f | test ROC-AUC=%.4f | test F1=%.4f (%.1fs, params=%s)",
            name, search.best_score_, test_metrics["roc_auc"], test_metrics["f1_score"], elapsed, search.best_params_,
        )

        comparison_rows.append(
            {
                "model": name,
                "cv_roc_auc": round(search.best_score_, 4),
                "best_params": json.dumps(search.best_params_),
                "train_seconds": round(elapsed, 1),
                **test_metrics,
            }
        )

        if search.best_score_ > best_overall["cv_roc_auc"]:
            best_overall.update(
                name=name,
                pipeline=search.best_estimator_,
                cv_roc_auc=search.best_score_,
                test_metrics=test_metrics,
                best_params=search.best_params_,
            )

    comparison_df = pd.DataFrame(comparison_rows).sort_values("cv_roc_auc", ascending=False)
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(COMPARISON_PATH, index=False)
    logger.info("Saved model comparison table to %s", COMPARISON_PATH)

    logger.info(
        "\n*** BEST MODEL: %s | CV ROC-AUC=%.4f | Test metrics=%s ***",
        best_overall["name"], best_overall["cv_roc_auc"], best_overall["test_metrics"],
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": best_overall["pipeline"],
            "model_name": best_overall["name"],
            "feature_columns": X_train.columns.tolist(),
            "cv_roc_auc": best_overall["cv_roc_auc"],
            "best_params": best_overall["best_params"],
        },
        MODEL_PATH,
    )
    logger.info("Saved deployable model artifact to %s", MODEL_PATH)

    metrics_out = {
        "best_model": best_overall["name"],
        "cv_roc_auc": round(float(best_overall["cv_roc_auc"]), 4),
        "best_params": best_overall["best_params"],
        "test_metrics": best_overall["test_metrics"],
        "model_comparison": comparison_rows,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)
    logger.info("Saved metrics summary to %s", METRICS_PATH)


if __name__ == "__main__":
    main()
