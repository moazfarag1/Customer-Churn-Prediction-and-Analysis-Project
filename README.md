# Customer Churn Prediction & Analysis

![Project Status](https://img.shields.io/badge/Status-Complete-success)
![DEPI Project](https://img.shields.io/badge/DEPI-Graduation_Project-blue)
![Model](https://img.shields.io/badge/Model-Logistic_Regression-orange)

## 1. Project Overview

This is our final **DEPI Graduation Project**, focused on analyzing and predicting customer churn. The goal is to identify users at risk of leaving the service so proactive retention strategies can be employed. This repository contains a fully working ML pipeline from data ingestion to model deployment via a local Streamlit web application.

## 2. Business Problem

Customer acquisition is expensive, and minimizing customer attrition is crucial for long-term profitability. By leveraging historical customer data (demographics, services used, account details), we built a machine learning model to predict churn probability, focusing primarily on **Recall**. Maximizing Recall ensures we catch as many at-risk customers as possible, prioritizing business intervention over pure accuracy.

## 3. Dataset Summary

- **Source:** IBM Telco Customer Churn
- **Size:** 7,043 instances. Originally 33 columns before cleaning, reduced through preprocessing, then final modeling schema became 28 columns (27 features + 1 target).
- **Preprocessed Schema:** 27 purely numerical columns designed for robust model inference.
- **Data Condition:** Fully encoded features. The 11 source-missing `Total Charges` values are preserved until training-only median imputation inside the fitted model pipeline.

## 4. Final Architecture Overview

This project is structured sequentially and adheres strictly to a **reproducible, local-first ML pipeline**. We avoided over-engineered cloud MLOps tools to ensure the project remains beginner-friendly, deployment-safe, and easy to study.

The final deployed app strictly uses a frozen `feature_schema.pkl` to transform user inputs safely without data leakage or dynamically changing schemas.

## 5. Phase-by-Phase Workflow

1. **Phase 1: EDA & Preprocessing:** Data cleaning, handling missing values, standardizing formats, and initial visualizations.
2. **Phase 2: Feature Engineering:** Creating new predictive features, mapping categorical data to numeric structures, and establishing the final 27-column schema.
3. **Phase 3: Baseline Modeling:** Establishing Logistic Regression and Random Forest performance.
4. **Phase 4: Optimization & Final Selection:** Comparing tuned XGBoost against the baselines to make the final model choice.
5. **Phase 5: Streamlit Integration:** Building an interactive UI that uses our frozen model pipeline for safe, fast local inference.
6. **Phase 6: Final Polish:** Hardening the repository, refining documentation, and preparing for graduation submission.

## 6. Final Model Decision: Why Logistic Regression Won

After rigorous testing against an optimized XGBoost model, we finalized **Logistic Regression** as the champion model for deployment. 

**Reasoning:**
- **Superior Recall:** It proved more reliable at identifying true churners, which aligns best with our core business goal.
- **Simplicity:** It provides better reproducibility and is easier to explain to stakeholders.
- **Deployment Safety:** It's lightweight, predictable, and fully embedded inside a standard scikit-learn Pipeline (with training-fitted median imputation and a `StandardScaler`).

*For a full breakdown of the model performance, see `MODEL_CARD.md`.*

## 7. Streamlit App & Installation

The Streamlit app allows you to enter hypothetical customer data and receive a live prediction regarding their churn risk. 

### Quick Start
1. **Clone the repository:**
   ```bash
   git clone https://github.com/moazfarag1/Customer-Churn-Prediction-and-Analysis-Project.git
   cd Customer-Churn-Prediction-and-Analysis-Project
   ```
2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the App:**
   ```bash
   streamlit run app/app.py
   ```
*See `DEPLOYMENT_GUIDE.md` for more details.*

## 8. Artifact Contracts & Reproducibility

To ensure the Streamlit app works seamlessly out of the box, we strictly rely on two static artifacts:
- `models/final_model_pipeline.pkl`: The trained sklearn Pipeline.
- `models/feature_schema.pkl`: The 27-column reference list.

The app reconstructs the user's data according to this exact schema, guaranteeing safe inference and eliminating training/deployment skew.

## 9. Project Structure

```text
app/                  # Streamlit application scripts
assets/plots/         # Visualization images saved from EDA & Modeling
data/                 # Raw and cleaned dataset versions
models/               # Saved `.pkl` model artifacts and schema
notebooks/            # Sequential Jupyter notebooks (01 to 04)
reports/              # In-depth project documentation
runtime_audit_utils.py # Shared notebook utility for backups and regeneration manifests
requirements.txt      # Python dependencies
README.md             # This file
```

## 10. Deployment

This project is fully ready for deployment on **Streamlit Community Cloud**.
For a comprehensive step-by-step guide on how to deploy this project, please refer to our [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).

## 11. Team Structure

- **Moaz Farag:** EDA, preprocessing, cleaning
- **Mohamed Mohy:** Feature engineering, encoding, statistical analysis
- **Mohamed Mahmoud:** Logistic Regression, Random Forest, evaluation
- **Mohamed Ali:** XGBoost, tuning, final model selection
- **Ali Mahmoud:** Streamlit deployment, GitHub coordination, final integration

## 12. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Status: Complete & Submission-Ready.*
