# Telco Churn Predictor App

This directory contains the Phase 5 Streamlit application for the DEPI Customer Churn Prediction project.

## Overview
This application provides a simple, business-friendly interface for predicting customer churn risk. It securely loads the hardened Logistic Regression pipeline (`final_model_pipeline.pkl`) and the locked feature schema (`feature_schema.pkl`) generated in earlier phases.

## How to Run
From the project directory, execute:

```bash
streamlit run app/app.py
```

## Architecture Notes
- **Inference Only:** This app does not retrain, tune, or fit any models.
- **Input Adapter:** The app collects human-readable inputs via UI dropdowns and manually adapts them into the strict 27-column numeric schema. It natively calculates the 5 engineered features required by the pipeline.
- **No Dynamic Encoding:** `pd.get_dummies()` is intentionally absent to prevent schema drifting during single-row inference.
