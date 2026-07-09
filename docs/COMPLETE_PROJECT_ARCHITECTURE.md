# Complete Project Architecture

Created During Phase 7 Final Engineering Readiness Audit  
Creation date: 2026-06-02

## Why This File Exists
This document is the public technical map for reviewers and future maintainers. It complements `PROJECT_EXECUTIVE_SUMMARY.md` with artifact ownership and execution flow.

## Runtime Architecture
```text
data/raw/telco_customer_churn.csv
  -> notebooks/01_moaz_eda_preprocessing.ipynb
  -> data/cleaned/cleaned_telco.csv
  -> notebooks/02_mohy_feature_engineering.ipynb
  -> data/cleaned/processed_telco.csv
  -> notebooks/03_mahmoud_modeling_baseline.ipynb
  -> models/logistic_regression_pipeline.pkl
  -> models/random_forest_pipeline.pkl
  -> models/feature_schema.pkl
  -> notebooks/04_ali_xgboost_optimization.ipynb
  -> models/xgboost_pipeline.pkl
  -> models/final_model_pipeline.pkl
  -> app/app.py
  -> deterministic Streamlit inference
```

## Ownership Boundaries
| Area | Owner File | Output |
| --- | --- | --- |
| Cleaning and EDA | `notebooks/01_moaz_eda_preprocessing.ipynb` | `cleaned_telco.csv`, EDA plots |
| Feature engineering and encoding | `notebooks/02_mohy_feature_engineering.ipynb` | `processed_telco.csv`, feature plots |
| Baselines and schema | `notebooks/03_mahmoud_modeling_baseline.ipynb` | LR, RF, schema artifacts |
| Candidate tuning and final selection | `notebooks/04_ali_xgboost_optimization.ipynb` | XGBoost and final LR artifact |
| Serving | `app/app.py` | Streamlit UI and adapter |
| Integrity gate | `tests/final_project_validation.py` | file, manifest, schema, leakage, app, and smoke checks |

## Artifact Safety
Notebook writes call `runtime_audit_utils.py`, which creates timestamped backups before overwriting and updates `runtime_tests/regeneration_manifest.json`. The validator checks manifest hashes so stale or partially regenerated outputs fail validation.

## Training-Serving Contract
The model expects the ordered 27-column list stored in `models/feature_schema.pkl`. The app manually maps human-readable fields, reconstructs engineered values, uses NumPy-compatible rounding for `Avg_Monthly_Spend`, validates basic business rules, and reorders columns before prediction.

## Deployment Contract
`app/app.py` resolves files relative to its own location, not the terminal working directory. It validates:
- model and schema files exist;
- schema is a list with exactly 27 entries;
- the pipeline exposes `predict` and `predict_proba`;
- pipeline steps are exactly `imputer`, `scaler`, and `model`;
- model feature names match the schema exactly.

