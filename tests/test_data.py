"""Unit tests for the data cleaning module."""
import pandas as pd
import pytest

from src.data.make_dataset import clean_data


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["0001-AAA", "0002-BBB", "0003-CCC"],
            "gender": ["Female", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0],
            "tenure": [1, 0, 24],
            "MonthlyCharges": [29.85, 56.95, 53.85],
            "TotalCharges": ["29.85", " ", "1300.5"],
            "Churn": ["No", "Yes", "No"],
        }
    )


def test_total_charges_coerced_to_numeric(raw_sample):
    cleaned = clean_data(raw_sample)
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])


def test_blank_total_charges_imputed_with_zero(raw_sample):
    cleaned = clean_data(raw_sample)
    # the row with tenure == 0 had a blank TotalCharges string
    assert cleaned.loc[cleaned["tenure"] == 0, "TotalCharges"].iloc[0] == 0


def test_senior_citizen_mapped_to_yes_no(raw_sample):
    cleaned = clean_data(raw_sample)
    assert set(cleaned["SeniorCitizen"].unique()) <= {"Yes", "No"}


def test_churn_target_binary_encoded(raw_sample):
    cleaned = clean_data(raw_sample)
    assert set(cleaned["Churn"].unique()) <= {0, 1}
    assert cleaned["Churn"].dtype.kind in "iu"


def test_customer_id_dropped(raw_sample):
    cleaned = clean_data(raw_sample)
    assert "customerID" not in cleaned.columns


def test_no_missing_values_after_cleaning(raw_sample):
    cleaned = clean_data(raw_sample)
    assert cleaned.isna().sum().sum() == 0
