"""
Data ingestion & cleaning module.

Loads the raw IBM Telco Customer Churn dataset, performs cleaning
(type coercion, missing-value handling, target encoding) and writes
an analysis-ready CSV to data/processed/.

Usage:
    python -m src.data.make_dataset
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "churn_clean.csv"


def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame."""
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    logger.info("Raw shape: %s", df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning steps to the raw Telco churn dataframe.

    Steps:
        1. Coerce TotalCharges to numeric (11 rows ship as blank strings
           for customers with tenure == 0); impute with 0.
        2. Standardize the SeniorCitizen flag to Yes/No for consistency
           with the other categorical columns.
        3. Binary-encode the target column (Churn: Yes/No -> 1/0).
        4. Drop the customerID identifier column (not predictive).
        5. Drop exact duplicate rows, if any.
    """
    df = df.copy()

    # 1. TotalCharges arrives as an object dtype because of blank strings
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = df["TotalCharges"].isna().sum()
    if n_missing:
        logger.info("Imputing %d missing TotalCharges values with 0 (tenure == 0 customers)", n_missing)
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # 2. SeniorCitizen ships as 0/1 int; make it categorical Yes/No like its peers
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # 3. Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    # 4. Drop identifier
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # 5. Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    if len(df) != before:
        logger.info("Dropped %d duplicate rows", before - len(df))

    df = df.reset_index(drop=True)
    logger.info("Clean shape: %s | churn rate: %.2f%%", df.shape, 100 * df["Churn"].mean())
    return df


def main() -> pd.DataFrame:
    df = load_raw_data()
    df_clean = clean_data(df)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(PROCESSED_PATH, index=False)
    logger.info("Saved cleaned dataset to %s", PROCESSED_PATH)
    return df_clean


if __name__ == "__main__":
    main()
