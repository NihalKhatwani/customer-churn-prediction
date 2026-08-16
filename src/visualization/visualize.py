"""
Exploratory data analysis & statistical hypothesis testing module.

Generates the core EDA figures (class balance, tenure/charges distributions,
churn-by-segment breakdowns, correlation heatmap) and runs formal statistical
significance tests (chi-square test of independence for categorical
predictors, Welch's t-test for numeric predictors) to quantify which
features are statistically associated with churn.

Usage:
    python -m src.visualization.visualize
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for script/CI execution
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "churn_clean.csv"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
STATS_PATH = PROJECT_ROOT / "reports" / "statistical_tests.json"

sns.set_theme(style="whitegrid", palette="deep")

CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def plot_churn_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["Churn"].value_counts().sort_index()
    labels = ["Retained", "Churned"]
    ax.bar(labels, counts.values, color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, f"{v}\n({v / len(df):.1%})", ha="center")
    ax.set_title("Customer Churn Class Distribution")
    ax.set_ylabel("Number of Customers")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_churn_distribution.png", dpi=150)
    plt.close(fig)


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, NUMERIC_COLS):
        sns.histplot(data=df, x=col, hue="Churn", kde=True, ax=ax, palette=["#4C72B0", "#C44E52"], element="step")
        ax.set_title(f"{col} distribution by Churn")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_numeric_distributions.png", dpi=150)
    plt.close(fig)


def plot_churn_by_contract(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    rate = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
    sns.barplot(x=rate.index, y=rate.values, ax=ax, hue=rate.index, legend=False, palette="Reds_r")
    ax.set_ylabel("Churn Rate")
    ax.set_title("Churn Rate by Contract Type")
    for i, v in enumerate(rate.values):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_churn_by_contract.png", dpi=150)
    plt.close(fig)


def plot_churn_by_tenure_and_internet(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    tenure_bins = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 100], labels=["0-12mo", "13-24mo", "25-48mo", "49mo+"])
    rate_tenure = df.groupby(tenure_bins, observed=True)["Churn"].mean()
    sns.barplot(x=rate_tenure.index.astype(str), y=rate_tenure.values, ax=axes[0], hue=rate_tenure.index.astype(str), legend=False, palette="Blues_r")
    axes[0].set_title("Churn Rate by Tenure Bucket")
    axes[0].set_ylabel("Churn Rate")

    rate_internet = df.groupby("InternetService")["Churn"].mean().sort_values(ascending=False)
    sns.barplot(x=rate_internet.index, y=rate_internet.values, ax=axes[1], hue=rate_internet.index, legend=False, palette="Purples_r")
    axes[1].set_title("Churn Rate by Internet Service")
    axes[1].set_ylabel("Churn Rate")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_churn_by_tenure_internet.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df[NUMERIC_COLS + ["Churn"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap (Numeric Features)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_correlation_heatmap.png", dpi=150)
    plt.close(fig)


def run_statistical_tests(df: pd.DataFrame) -> dict:
    """Chi-square test of independence for each categorical feature vs.
    Churn, and Welch's t-test for each numeric feature vs. Churn.
    Returns a JSON-serializable dict of results, sorted by significance.
    """
    results = {"chi_square_tests": [], "t_tests": []}

    for col in CATEGORICAL_COLS:
        contingency = pd.crosstab(df[col], df["Churn"])
        chi2, p, dof, _ = stats.chi2_contingency(contingency)
        results["chi_square_tests"].append(
            {
                "feature": col,
                "chi2_statistic": round(float(chi2), 3),
                "p_value": float(p),
                "degrees_of_freedom": int(dof),
                "significant_at_0.05": bool(p < 0.05),
            }
        )

    for col in NUMERIC_COLS:
        churned = df.loc[df["Churn"] == 1, col]
        retained = df.loc[df["Churn"] == 0, col]
        t_stat, p = stats.ttest_ind(churned, retained, equal_var=False)  # Welch's t-test
        results["t_tests"].append(
            {
                "feature": col,
                "t_statistic": round(float(t_stat), 3),
                "p_value": float(p),
                "mean_churned": round(float(churned.mean()), 2),
                "mean_retained": round(float(retained.mean()), 2),
                "significant_at_0.05": bool(p < 0.05),
            }
        )

    results["chi_square_tests"].sort(key=lambda r: r["p_value"])
    results["t_tests"].sort(key=lambda r: r["p_value"])
    return results


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading data from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    logger.info("Generating EDA figures...")
    plot_churn_distribution(df)
    plot_numeric_distributions(df)
    plot_churn_by_contract(df)
    plot_churn_by_tenure_and_internet(df)
    plot_correlation_heatmap(df)
    logger.info("Saved 5 figures to %s", FIG_DIR)

    logger.info("Running chi-square and t-test statistical significance tests...")
    results = run_statistical_tests(df)
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    n_sig_cat = sum(r["significant_at_0.05"] for r in results["chi_square_tests"])
    n_sig_num = sum(r["significant_at_0.05"] for r in results["t_tests"])
    logger.info(
        "%d/%d categorical and %d/%d numeric features significantly associated with churn (p<0.05)",
        n_sig_cat, len(CATEGORICAL_COLS), n_sig_num, len(NUMERIC_COLS),
    )
    logger.info("Saved statistical test results to %s", STATS_PATH)


if __name__ == "__main__":
    main()
