"""Unit tests for the feature engineering module."""
import pandas as pd
import pytest

from src.features.build_features import build_preprocessor, engineer_features


@pytest.fixture
def clean_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gender": ["Female", "Male"],
            "SeniorCitizen": ["No", "Yes"],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "tenure": [2, 50],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "Yes"],
            "InternetService": ["Fiber optic", "DSL"],
            "OnlineSecurity": ["No", "Yes"],
            "OnlineBackup": ["No", "Yes"],
            "DeviceProtection": ["No", "Yes"],
            "TechSupport": ["No", "Yes"],
            "StreamingTV": ["Yes", "No"],
            "StreamingMovies": ["No", "No"],
            "Contract": ["Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Credit card (automatic)"],
            "MonthlyCharges": [85.4, 45.0],
            "TotalCharges": [170.8, 2250.0],
            "Churn": [1, 0],
        }
    )


def test_engineer_features_adds_expected_columns(clean_sample):
    result = engineer_features(clean_sample)
    expected_new_cols = {
        "tenure_bucket", "num_active_services", "has_internet",
        "avg_monthly_spend", "charge_per_service", "is_month_to_month", "contract_risk_score",
    }
    assert expected_new_cols.issubset(result.columns)


def test_num_active_services_counted_correctly(clean_sample):
    result = engineer_features(clean_sample)
    # SERVICE_COLUMNS = PhoneService, MultipleLines, OnlineSecurity, OnlineBackup,
    # DeviceProtection, TechSupport, StreamingTV, StreamingMovies
    # second row: PhoneService, MultipleLines, OnlineSecurity, OnlineBackup,
    # DeviceProtection, TechSupport = 6 "Yes"
    assert result.loc[1, "num_active_services"] == 6
    # first row: PhoneService, StreamingTV = 2 "Yes"
    assert result.loc[0, "num_active_services"] == 2


def test_is_month_to_month_flag(clean_sample):
    result = engineer_features(clean_sample)
    assert result.loc[0, "is_month_to_month"] == 1
    assert result.loc[1, "is_month_to_month"] == 0


def test_contract_risk_score_ordinal(clean_sample):
    result = engineer_features(clean_sample)
    assert result.loc[0, "contract_risk_score"] == 0  # Month-to-month
    assert result.loc[1, "contract_risk_score"] == 2  # Two year


def test_avg_monthly_spend_does_not_divide_by_zero(clean_sample):
    clean_sample.loc[0, "tenure"] = 0
    clean_sample.loc[0, "TotalCharges"] = 0
    result = engineer_features(clean_sample)
    assert result.loc[0, "avg_monthly_spend"] == 0  # clipped to tenure=1, TotalCharges=0


def test_preprocessor_transforms_without_error(clean_sample):
    engineered = engineer_features(clean_sample)
    preprocessor, numeric_cols, _categorical_cols = build_preprocessor(engineered)
    X = engineered.drop(columns=["Churn"])
    X_transformed = preprocessor.fit_transform(X)
    assert X_transformed.shape[0] == len(clean_sample)
    assert X_transformed.shape[1] > len(numeric_cols)  # one-hot expands categorical columns


def test_preprocessor_handles_unseen_category_at_inference(clean_sample):
    engineered = engineer_features(clean_sample)
    preprocessor, _, _ = build_preprocessor(engineered)
    X = engineered.drop(columns=["Churn"])
    preprocessor.fit(X)

    new_row = X.iloc[[0]].copy()
    new_row["PaymentMethod"] = "Some Brand New Payment Method"
    # should not raise thanks to handle_unknown='ignore'
    preprocessor.transform(new_row)
