"""
Streamlit dashboard for the Customer Churn Prediction project.

Three tabs:
    1. Overview        - dataset KPIs, class balance, headline business insights
    2. Predict          - interactive single-customer scoring form (calls the
                           trained pipeline directly, no API dependency required)
    3. Model Performance - test-set metrics, ROC/PR curves, feature importance

Run:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.features.build_features import TARGET, engineer_features

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")


@st.cache_resource
def load_model():
    artifact = joblib.load(PROJECT_ROOT / "models" / "churn_model.joblib")
    return artifact


@st.cache_data
def load_data():
    return pd.read_csv(PROJECT_ROOT / "data" / "processed" / "churn_clean.csv")


@st.cache_data
def load_metrics():
    with open(PROJECT_ROOT / "reports" / "metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_feature_importance():
    return pd.read_csv(PROJECT_ROOT / "reports" / "feature_importance.csv")


artifact = load_model()
pipeline = artifact["pipeline"]
df = load_data()
metrics = load_metrics()
feat_imp = load_feature_importance()

st.title("📉 Customer Churn Prediction Dashboard")
st.caption(
    "End-to-end ML system — EDA → feature engineering → SMOTE-balanced XGBoost/Random Forest/"
    "Logistic Regression comparison → hyperparameter tuning → deployment. "
    f"Best model: **{artifact['model_name']}** | Cross-validated ROC-AUC: **{artifact['cv_roc_auc']:.3f}**"
)

tab_overview, tab_predict, tab_performance = st.tabs(["📊 Overview", "🔮 Predict", "🧪 Model Performance"])

# ---------------------------------------------------------------------------
# TAB 1: Overview
# ---------------------------------------------------------------------------
with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Churn Rate", f"{df['Churn'].mean():.1%}")
    col3.metric("Avg. Monthly Revenue at Risk", f"${(df.loc[df['Churn']==1,'MonthlyCharges']).sum():,.0f}")
    col4.metric("Avg. Tenure (Churned)", f"{df.loc[df['Churn']==1,'tenure'].mean():.1f} mo")

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        rate = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False).reset_index()
        rate["Churn"] = rate["Churn"] * 100
        fig = px.bar(rate, x="Contract", y="Churn", title="Churn Rate by Contract Type (%)",
                     color="Churn", color_continuous_scale="Reds")
        fig.update_layout(yaxis_title="Churn Rate (%)", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        seg = df.groupby(["Contract", "InternetService"])["Churn"].agg(["mean", "count"]).reset_index()
        seg = seg[seg["count"] >= 50].sort_values("mean", ascending=False).head(5)
        seg["mean"] = seg["mean"] * 100
        seg["segment"] = seg["Contract"] + " / " + seg["InternetService"]
        fig2 = px.bar(seg, x="segment", y="mean", title="Top 5 Highest-Risk Segments (%)",
                      color="mean", color_continuous_scale="Reds")
        fig2.update_layout(yaxis_title="Churn Rate (%)", xaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "💡 **Business insight:** Month-to-month, fiber-optic customers churn at **~54%** — nearly "
        "20x the rate of two-year contract customers on DSL/no internet. Electronic-check payers "
        "churn at **45%** vs. ~15-19% for automatic payment methods. See `reports/sql_insights.md` "
        "for the full SQL-driven business analysis."
    )

# ---------------------------------------------------------------------------
# TAB 2: Predict
# ---------------------------------------------------------------------------
with tab_predict:
    st.subheader("Score a customer")
    st.caption("Fill in customer attributes and get a live churn-risk prediction from the deployed pipeline.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with c2:
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        with c3:
            support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)

        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly_charges * max(tenure, 1)))
        submitted = st.form_submit_button("Predict Churn Risk", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple_lines,
            "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
            "DeviceProtection": protection, "TechSupport": support, "StreamingTV": tv,
            "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        }])
        engineered = engineer_features(row)[artifact["feature_columns"]]
        proba = float(pipeline.predict_proba(engineered)[0, 1])

        risk = "🔴 High" if proba >= 0.66 else "🟡 Medium" if proba >= 0.33 else "🟢 Low"
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Churn Probability", f"{proba:.1%}")
            st.metric("Risk Tier", risk)
        with c2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                title={"text": "Churn Risk (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkred" if proba >= 0.66 else "orange" if proba >= 0.33 else "green"},
                    "steps": [
                        {"range": [0, 33], "color": "#d4edda"},
                        {"range": [33, 66], "color": "#fff3cd"},
                        {"range": [66, 100], "color": "#f8d7da"},
                    ],
                },
            ))
            fig.update_layout(height=250, margin={"l": 20, "r": 20, "t": 40, "b": 20})
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3: Model Performance
# ---------------------------------------------------------------------------
with tab_performance:
    st.subheader(f"Best model: {metrics['best_model']} (5-fold CV ROC-AUC: {metrics['cv_roc_auc']:.4f})")

    tm = metrics["test_metrics"]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{tm['accuracy']:.1%}")
    m2.metric("Precision", f"{tm['precision']:.1%}")
    m3.metric("Recall", f"{tm['recall']:.1%}")
    m4.metric("F1-Score", f"{tm['f1_score']:.3f}")
    m5.metric("ROC-AUC", f"{tm['roc_auc']:.3f}")

    st.divider()
    st.markdown("#### Model comparison (5-fold cross-validated ROC-AUC)")
    comp_df = pd.DataFrame(metrics["model_comparison"])
    st.dataframe(
        comp_df[["model", "cv_roc_auc", "accuracy", "precision", "recall", "f1_score", "roc_auc"]],
        use_container_width=True,
        hide_index=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(feat_imp.sort_values("importance"), x="importance", y="feature",
                     orientation="h", title="Top 15 Feature Importances (XGBoost)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        test_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")
        X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC={tm['roc_auc']:.3f})"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line={"dash": "dash", "color": "gray"}, name="Random"))
        fig_roc.update_layout(title="ROC Curve (Test Set)", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

st.divider()
st.caption(
    "Built with scikit-learn, XGBoost, imbalanced-learn (SMOTE), FastAPI, and Streamlit. "
    "Source: github.com/NihalKhatwani/customer-churn-prediction"
)
