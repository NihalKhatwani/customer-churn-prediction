"""
Model evaluation & interpretability module.

Loads the persisted best-model pipeline, scores it on the held-out test
set, and produces the standard model-quality artifacts a stakeholder /
hiring manager expects to see: confusion matrix, ROC curve, precision-
recall curve, classification report, and a feature-importance chart
(native importances for tree models, |coefficient| for linear models).

Usage:
    python -m src.models.evaluate_model
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
)

from src.features.build_features import TARGET

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "classification_report.txt"

sns.set_theme(style="whitegrid")


def get_feature_names(pipeline) -> list[str]:
    """Recover human-readable feature names after ColumnTransformer + OneHotEncoder."""
    preprocessor = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def plot_confusion_matrix(y_test, y_pred) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Retained", "Churned"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix (Test Set)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_roc_curve(pipeline, X_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, name="XGBoost (best model)")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    ax.set_title("ROC Curve (Test Set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_roc_curve.png", dpi=150)
    plt.close(fig)


def plot_pr_curve(pipeline, X_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, name="XGBoost (best model)")
    ax.set_title("Precision-Recall Curve (Test Set)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_precision_recall_curve.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(pipeline, top_n: int = 15) -> None:
    clf = pipeline.named_steps["clf"]
    feature_names = get_feature_names(pipeline)

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        title = "Top Feature Importances (XGBoost gain-based)"
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
        title = "Top Feature Importances (|Logistic Regression coefficient|)"
    else:
        logger.warning("Model type has no native importance attribute; skipping importance plot")
        return

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=imp_df, x="importance", y="feature", ax=ax, hue="feature", legend=False, palette="viridis")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_feature_importance.png", dpi=150)
    plt.close(fig)

    imp_df.to_csv(PROJECT_ROOT / "reports" / "feature_importance.csv", index=False)


def main() -> None:
    logger.info("Loading model artifact from %s", MODEL_PATH)
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]

    test_df = pd.read_csv(TEST_PATH)
    X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]
    y_pred = pipeline.predict(X_test)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(pipeline, X_test, y_test)
    plot_pr_curve(pipeline, X_test, y_test)
    plot_feature_importance(pipeline)
    logger.info("Saved confusion matrix, ROC curve, PR curve, and feature-importance figures to %s", FIG_DIR)

    report = classification_report(y_test, y_pred, target_names=["Retained", "Churned"])
    logger.info("\nClassification Report:\n%s", report)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    logger.info("Saved classification report to %s", REPORT_PATH)


if __name__ == "__main__":
    main()
