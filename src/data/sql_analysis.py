"""
SQL analysis module.

Loads the cleaned churn dataset into a local SQLite database and answers
a set of business-stakeholder questions using pure SQL (window functions,
CTEs, aggregations) -- demonstrating the kind of ad-hoc analytical SQL a
Data Scientist writes against a warehouse table. Results are exported to
a Markdown report for easy review.

Usage:
    python -m src.data.sql_analysis
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "churn_clean.csv"
DB_PATH = PROJECT_ROOT / "data" / "processed" / "churn.db"
REPORT_PATH = PROJECT_ROOT / "reports" / "sql_insights.md"

QUERIES: dict[str, str] = {
    "Overall churn rate": """
        SELECT
            COUNT(*) AS total_customers,
            SUM(Churn) AS churned_customers,
            ROUND(100.0 * SUM(Churn) / COUNT(*), 2) AS churn_rate_pct
        FROM customers;
    """,
    "Churn rate & revenue at risk by contract type": """
        SELECT
            Contract,
            COUNT(*) AS customers,
            ROUND(100.0 * SUM(Churn) / COUNT(*), 2) AS churn_rate_pct,
            ROUND(SUM(CASE WHEN Churn = 1 THEN MonthlyCharges ELSE 0 END), 2) AS monthly_revenue_at_risk
        FROM customers
        GROUP BY Contract
        ORDER BY churn_rate_pct DESC;
    """,
    "Top 5 highest-risk customer segments (contract x internet service)": """
        SELECT
            Contract,
            InternetService,
            COUNT(*) AS customers,
            ROUND(100.0 * SUM(Churn) / COUNT(*), 2) AS churn_rate_pct
        FROM customers
        GROUP BY Contract, InternetService
        HAVING COUNT(*) >= 50
        ORDER BY churn_rate_pct DESC
        LIMIT 5;
    """,
    "Tenure cohort churn rate (using CASE-based binning)": """
        SELECT
            CASE
                WHEN tenure <= 12 THEN '0-12mo'
                WHEN tenure <= 24 THEN '13-24mo'
                WHEN tenure <= 48 THEN '25-48mo'
                ELSE '49mo+'
            END AS tenure_cohort,
            COUNT(*) AS customers,
            ROUND(100.0 * SUM(Churn) / COUNT(*), 2) AS churn_rate_pct,
            ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
        FROM customers
        GROUP BY tenure_cohort
        ORDER BY churn_rate_pct DESC;
    """,
    "Payment method ranked by churn rate (window function)": """
        SELECT
            PaymentMethod,
            churn_rate_pct,
            RANK() OVER (ORDER BY churn_rate_pct DESC) AS risk_rank
        FROM (
            SELECT
                PaymentMethod,
                ROUND(100.0 * SUM(Churn) / COUNT(*), 2) AS churn_rate_pct
            FROM customers
            GROUP BY PaymentMethod
        ) sub
        ORDER BY risk_rank;
    """,
    "Customers with no add-on services vs. full-service customers": """
        WITH service_counts AS (
            SELECT
                *,
                (CASE WHEN OnlineSecurity = 'Yes' THEN 1 ELSE 0 END +
                 CASE WHEN OnlineBackup = 'Yes' THEN 1 ELSE 0 END +
                 CASE WHEN DeviceProtection = 'Yes' THEN 1 ELSE 0 END +
                 CASE WHEN TechSupport = 'Yes' THEN 1 ELSE 0 END) AS addon_count
            FROM customers
        )
        SELECT
            CASE WHEN addon_count = 0 THEN 'No add-ons' ELSE 'Has add-ons' END AS segment,
            COUNT(*) AS customers,
            ROUND(100.0 * SUM(Churn) / COUNT(*), 2) AS churn_rate_pct
        FROM service_counts
        GROUP BY segment;
    """,
}


def load_to_sqlite(csv_path: Path = DATA_PATH, db_path: Path = DB_PATH) -> None:
    df = pd.read_csv(csv_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql("customers", conn, if_exists="replace", index=False)
    logger.info("Loaded %d rows into SQLite table 'customers' at %s", len(df), db_path)


def run_queries(db_path: Path = DB_PATH) -> dict[str, pd.DataFrame]:
    results = {}
    with sqlite3.connect(db_path) as conn:
        for name, sql in QUERIES.items():
            results[name] = pd.read_sql_query(sql, conn)
    return results


def write_report(results: dict[str, pd.DataFrame], path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# SQL Business Insights\n", "Generated via `src/data/sql_analysis.py` against a SQLite copy of the cleaned dataset.\n"]
    for name, df in results.items():
        lines.append(f"## {name}\n")
        lines.append(df.to_markdown(index=False))
        lines.append("\n")
    path.write_text("\n".join(lines))
    logger.info("Wrote SQL insights report to %s", path)


def main() -> None:
    load_to_sqlite()
    results = run_queries()
    for name, df in results.items():
        logger.info("\n--- %s ---\n%s", name, df.to_string(index=False))
    write_report(results)


if __name__ == "__main__":
    main()
