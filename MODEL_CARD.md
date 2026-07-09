# Model Card: Logistic Regression Churn Predictor

## Model Details
- **Model Type:** Logistic Regression within a Scikit-Learn Pipeline
- **Preprocessing:** Includes training-fitted median imputation followed by a `StandardScaler`.
- **Pipeline Components:** `SimpleImputer(strategy="median")` -> `StandardScaler` -> `LogisticRegression(max_iter=1000)`
- **Framework:** Scikit-Learn

## Intended Use
- **Primary Use Case:** Predict the likelihood of a customer churning (canceling their service) based on their account attributes and demographic data.
- **Target Audience:** Business stakeholders and retention teams looking to intervene proactively.

## Feature Schema
The model strictly expects a 27-column numeric input array (`models/feature_schema.pkl`). All human-readable data must be preprocessed and encoded to match this schema exactly before inference. No dynamic encoding (e.g., `pd.get_dummies()`) is performed during inference to prevent data leakage and schema mismatch.

## Performance Metrics
- **Primary Metric Optimized:** Recall
- **Recall (Class 1 - Churn):** The model excels at identifying true churners, prioritizing false positives over false negatives. Catching a churner is far more valuable than the minor cost of a redundant retention offer.
- *Note: XGBoost showed competitive ROC-AUC/precision in some comparisons, but Logistic Regression achieved better Recall, which was the project's primary business metric.*

## Limitations
- **Static Weights:** The model was trained on a historical snapshot of the IBM Telco dataset. It cannot adapt to changing consumer trends unless retrained.
- **Feature Dependency:** The model relies heavily on tenure and contract type; novel churn drivers not present in the historical dataset will go unnoticed.

## Deployment Safety
The `models/final_model_pipeline.pkl` artifact is completely decoupled from the training environment. It only requires Scikit-Learn and Pandas to generate predictions, making it incredibly stable for local Streamlit deployment.
