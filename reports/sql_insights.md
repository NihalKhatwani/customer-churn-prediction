# SQL Business Insights

Generated via `src/data/sql_analysis.py` against a SQLite copy of the cleaned dataset.

## Overall churn rate

|   total_customers |   churned_customers |   churn_rate_pct |
|------------------:|--------------------:|-----------------:|
|              7021 |                1857 |            26.45 |


## Churn rate & revenue at risk by contract type

| Contract       |   customers |   churn_rate_pct |   monthly_revenue_at_risk |
|:---------------|------------:|-----------------:|--------------------------:|
| Month-to-month |        3853 |            42.64 |                  120256   |
| One year       |        1473 |            11.27 |                   14118.5 |
| Two year       |        1695 |             2.83 |                    4165.3 |


## Top 5 highest-risk customer segments (contract x internet service)

| Contract       | InternetService   |   customers |   churn_rate_pct |
|:---------------|:------------------|------------:|-----------------:|
| Month-to-month | Fiber optic       |        2122 |            54.48 |
| Month-to-month | DSL               |        1221 |            32.1  |
| One year       | Fiber optic       |         539 |            19.29 |
| Month-to-month | No                |         510 |            18.63 |
| One year       | DSL               |         570 |             9.3  |


## Tenure cohort churn rate (using CASE-based binning)

| tenure_cohort   |   customers |   churn_rate_pct |   avg_monthly_charges |
|:----------------|------------:|-----------------:|----------------------:|
| 0-12mo          |        2164 |            47.37 |                 56.3  |
| 13-24mo         |        1024 |            28.71 |                 61.36 |
| 25-48mo         |        1594 |            20.39 |                 65.93 |
| 49mo+           |        2239 |             9.51 |                 73.95 |


## Payment method ranked by churn rate (window function)

| PaymentMethod             |   churn_rate_pct |   risk_rank |
|:--------------------------|-----------------:|------------:|
| Electronic check          |            45.15 |           1 |
| Mailed check              |            18.92 |           2 |
| Bank transfer (automatic) |            16.71 |           3 |
| Credit card (automatic)   |            15.24 |           4 |


## Customers with no add-on services vs. full-service customers

| segment     |   customers |   churn_rate_pct |
|:------------|------------:|-----------------:|
| Has add-ons |        4250 |            24.42 |
| No add-ons  |        2771 |            29.56 |

