"""Pydantic request/response schemas for the churn prediction API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """Raw customer attributes, matching the cleaned Telco churn schema
    (see `src/data/make_dataset.py`). Engineered features are derived
    server-side so API consumers never need to replicate feature logic.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal["Yes", "No"]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100, description="Months as a customer")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class PredictionResponse(BaseModel):
    churn_probability: float = Field(description="Predicted probability of churn, 0-1")
    churn_prediction: Literal["Yes", "No"]
    risk_tier: Literal["Low", "Medium", "High"]
    model_name: str
    model_version: str


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_name: str
    cv_roc_auc: float
