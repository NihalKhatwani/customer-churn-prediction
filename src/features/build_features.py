"""
Feature engineering module.

Engineers domain-driven features on top of the cleaned Telco churn dataset,
builds a scikit-learn preprocessing Pipeline/ColumnTransformer (imputation +
scaling + one-hot encoding), performs a stratified train/test split, and
persists both the engineered datasets and the fitted preprocessing pipeline
so training and inference stay perfectly consistent.

Usage:
    python -m src.features.build_features
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "churn_clean.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "churn_features.csv"
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
PIPELINE_PATH = PROJECT_ROOT / "models" / "preprocessing_pipeline.joblib"

TARGET = "Churn"
RANDOM_STATE = 42

# The set of "add-on" service columns used to build engineered aggregate features
SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new predictive features from the raw cleaned columns.

    New features:
        - tenure_bucket: categorical bucket of customer tenure (New/Established/Loyal/VIP)
        - num_active_services: count of subscribed add-on services (0-8)
        - has_internet: boolean flag derived from InternetService
        - avg_monthly_spend: TotalCharges / max(tenure, 1) -- smooths billing history
        - charge_per_service: MonthlyCharges / (num_active_services + 1) -- pricing efficiency
        - is_month_to_month: boolean flag, the single strongest churn predictor in the domain
        - contract_risk_score: ordinal encoding of contract commitment (0=Month-to-month .. 2=Two year)
    """
    df = df.copy()

    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 100],
        labels=["New(0-12mo)", "Established(1-2yr)", "Loyal(2-4yr)", "VIP(4yr+)"],
    ).astype(str)

    def _count_active(row) -> int:
        return sum(1 for col in SERVICE_COLUMNS if row[col] == "Yes")

    df["num_active_services"] = df.apply(_count_active, axis=1)

    df["has_internet"] = (df["InternetService"] != "No").astype(int)

    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)

    df["charge_per_service"] = df["MonthlyCharges"] / (df["num_active_services"] + 1)

    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

    contract_order = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    df["contract_risk_score"] = df["Contract"].map(contract_order)

    return df


def build_preprocessor(df: pd.DataFrame, target: str = TARGET) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build a ColumnTransformer that imputes/scales numeric features and
    imputes/one-hot-encodes categorical features. Returns the transformer
    plus the resolved numeric/categorical column lists.
    """
    feature_df = df.drop(columns=[target])
    numeric_cols = feature_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object", "str"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
    logger.info("Numeric features (%d): %s", len(numeric_cols), numeric_cols)
    logger.info("Categorical features (%d): %s", len(categorical_cols), categorical_cols)
    return preprocessor, numeric_cols, categorical_cols


def main() -> None:
    logger.info("Loading cleaned data from %s", PROCESSED_PATH)
    df = pd.read_csv(PROCESSED_PATH)

    df_feat = engineer_features(df)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(FEATURES_PATH, index=False)
    logger.info("Saved engineered feature set (%s) to %s", df_feat.shape, FEATURES_PATH)

    train_df, test_df = train_test_split(
        df_feat,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df_feat[TARGET],
    )
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    logger.info("Train shape: %s | Test shape: %s", train_df.shape, test_df.shape)
    logger.info(
        "Train churn rate: %.2f%% | Test churn rate: %.2f%%",
        100 * train_df[TARGET].mean(),
        100 * test_df[TARGET].mean(),
    )

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(train_df)
    preprocessor.fit(train_df.drop(columns=[TARGET]))

    PIPELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "target": TARGET,
        },
        PIPELINE_PATH,
    )
    logger.info("Saved fitted preprocessing pipeline to %s", PIPELINE_PATH)


if __name__ == "__main__":
    main()
