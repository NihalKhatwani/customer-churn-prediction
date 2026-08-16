# Resume Bullet Points — Customer Churn Prediction

Copy the section below into your resume's **Projects** section. Pick 3-4 bullets (don't use all
of them — recruiters skim). Swap `<your-repo-url>` for your actual GitHub link. Every bullet is
built from the real, reproducible numbers in this repo (`reports/metrics.json`,
`reports/statistical_tests.json`, `reports/sql_insights.md`) — verify them against your own run
before publishing so you can defend every number in an interview.

---

## Project header (for your resume)

**Customer Churn Prediction — End-to-End ML System** | Python, scikit-learn, XGBoost, FastAPI, Docker
`github.com/<your-username>/customer-churn-prediction` — [live demo link if deployed]

---

## Bullet points (pick 3-4)

- Engineered an end-to-end **machine learning** pipeline in **Python** to predict customer churn
  on 7,000+ telecom records, achieving **0.84 ROC-AUC** by comparing Logistic Regression, Random
  Forest, and **XGBoost** via **5-fold cross-validated GridSearchCV hyperparameter tuning**.

- Performed **exploratory data analysis** and **statistical hypothesis testing** (chi-square,
  Welch's t-test) in **Pandas**/**SciPy**, identifying 17 statistically significant churn drivers
  (p < 0.05) and validating findings with SQL-based cohort analysis using **window functions** and
  **CTEs** against a **SQLite** database.

- Engineered 7 domain-driven predictive features and built a scikit-learn **preprocessing
  pipeline** (`ColumnTransformer`); addressed **class imbalance** (26% minority class) with
  **SMOTE oversampling** inside cross-validation folds to prevent data leakage.

- Deployed the trained model as a versioned **REST API** with **FastAPI** and **Pydantic** schema
  validation (single + batch inference endpoints), containerized with **Docker**/**Docker
  Compose**, and automated testing/build via a **GitHub Actions CI/CD** pipeline.

- Built an interactive **Streamlit** analytics dashboard (**Plotly** visualizations) surfacing
  live churn-risk scoring, model performance metrics, and a business insight showing
  month-to-month, fiber-optic customers churn at **~54%** — nearly 20x the two-year-contract
  segment — quantifying **$120K+/month** in at-risk recurring revenue.

- Wrote **26 automated tests** (**pytest**) covering data validation, feature engineering, model
  quality regression guards, and API contract/behavior, integrated into a CI pipeline that lints,
  retrains, and re-validates the model on every push.

---

## Shorter, single-line alternatives

- Built and deployed an XGBoost churn-prediction model (0.84 ROC-AUC) as a FastAPI REST service
  with Docker, Streamlit dashboard, and full CI/CD pipeline.
- Applied statistical hypothesis testing and SQL analysis to identify churn drivers, then
  productionized a SMOTE-balanced, hyperparameter-tuned ML pipeline serving live predictions via
  REST API.

---

## Skills line (for resume's Skills/Technical Skills section)

**Languages:** Python, SQL
**ML/Data Science:** scikit-learn, XGBoost, Pandas, NumPy, SciPy, imbalanced-learn, feature
engineering, hyperparameter tuning, cross-validation, statistical hypothesis testing, A/B testing
concepts, model evaluation & interpretability
**Visualization:** Matplotlib, Seaborn, Plotly, Streamlit
**MLOps/Engineering:** FastAPI, Pydantic, Docker, Docker Compose, GitHub Actions (CI/CD), pytest,
Git, REST APIs, SQLite

---

## Interview talking points (be ready to explain, don't just list)

- **Why SMOTE inside the CV pipeline and not before the split?** To avoid data leakage — fitting
  SMOTE on the full dataset before splitting would let synthetic points derived from test-set
  neighbors leak into training, inflating validation scores unrealistically.
- **Why ROC-AUC over accuracy as the primary metric?** The target is imbalanced (~26% churn), so
  a model that always predicts "no churn" gets ~74% accuracy while being useless — ROC-AUC
  measures ranking quality independent of the class threshold.
- **Why XGBoost won:** Highest cross-validated ROC-AUC (0.847) among the three tuned candidates,
  with the best precision/accuracy tradeoff on the held-out test set — see
  `reports/model_comparison.csv` for the full 3-model comparison.
- **How the API avoids train/serve skew:** the exact same `engineer_features()` function and
  fitted `ColumnTransformer` used in training are reused by both the FastAPI service and the
  Streamlit dashboard, so there's a single source of truth for feature logic.
