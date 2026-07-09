# Project Executive Summary

Created During Phase 7 Final Engineering Readiness Audit  
Creation date: 2026-06-02

## Why This File Exists
This is the concise public overview for professors, recruiters, teammates, and technical reviewers. Use it with the root `README.md`, `MODEL_CARD.md`, and `DEPLOYMENT_GUIDE.md`.

## Project Outcome
The project predicts telecom customer churn using a local-first, reproducible machine learning workflow and a Streamlit application. The deployed champion is Logistic Regression because it achieved the strongest recall among the evaluated release candidates while remaining lightweight and explainable.

## Verified Release Contract
- Dataset: IBM Telco Customer Churn, 7,043 rows.
- Modeling table: `data/cleaned/processed_telco.csv`, 27 numeric features plus `Churn Label`.
- Expected missing values: 11 source-missing `Total Charges` values retained until model inference or training.
- Champion artifact: `models/final_model_pipeline.pkl`.
- Pipeline steps: `SimpleImputer(strategy="median")` -> `StandardScaler` -> `LogisticRegression`.
- Schema artifact: `models/feature_schema.pkl`, exactly 27 ordered feature names.
- Serving entrypoint: `app/app.py`.

## Evaluation Integrity
The modeling workflow uses a deterministic 64% train, 16% validation, and 20% untouched test split. Candidate selection occurs on validation data. The final test holdout is used once after selection and refitting on development data.

## Deployment Safety
The Streamlit app reconstructs the same 27 features used during training, orders them using the frozen schema, and sends them through the serialized sklearn pipeline. Phase 6A verification confirmed exact adapter parity for all 7,043 source rows, repeated deterministic predictions, clean rendering under pinned `streamlit==1.31.0`, and startup rejection for stale pipelines or reordered schemas.

## Scope Boundary
This is a strong academic and portfolio deployment, not a complete telecom production platform. It does not include authentication, monitoring, drift detection, a prediction database, or automated retraining.

