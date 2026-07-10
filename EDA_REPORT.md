# Exploratory Data Analysis (EDA) & Preprocessing Report

**Project:** Customer Churn Prediction and Analysis
**Phase:** Milestone 1 - Data Preparation
**Author:** Moaz Farag

---

## 1. Executive Summary
This report outlines the foundational steps taken during the Exploratory Data Analysis (EDA) and data preprocessing phase. The primary objective was to thoroughly understand the Telco customer dataset, identify underlying churn patterns, clean the raw data, and prepare a robust dataset for subsequent feature engineering and modeling. The most critical decision made during this phase was the removal of "Data Leakage" columns to ensure the final models are reliable and viable for production.

---

## 2. Dataset Overview
- **Source:** IBM Telco Customer Churn Dataset
- **Initial Shape:** 7,043 rows and 33 columns.
- **Target Variable:** `Churn Label` (Indicates whether a customer left the company).

---

## 3. Key EDA Insights
Through statistical summaries and data visualization, several critical business insights were uncovered:

### A. Target Variable Imbalance
The dataset exhibits a moderate class imbalance:
- **Retained Customers:** 73.5%
- **Churned Customers:** 26.5%
*Actionable Insight:* Evaluation metrics must account for this imbalance (e.g., relying on F1-Score or Recall rather than just Accuracy).

### B. High-Risk Customer Segments
1. **Contract Type:** Customers on a **Month-to-month** contract have the highest churn rate by a significant margin compared to those on 1-year or 2-year contracts.
2. **Tenure:** The churn rate is highly concentrated among **new customers** (low tenure). Customer retention drops significantly after the first 12 months.
3. **Internet Service:** Users subscribed to **Fiber Optic** internet show a disproportionately high churn rate, potentially indicating pricing dissatisfaction or service quality issues.
4. **Monthly Charges:** Customers with **higher monthly bills** are more likely to churn, confirming price sensitivity.

---

## 4. Preprocessing Decisions

### A. Eliminating Data Leakage
Data leakage occurs when a model is trained on information that will not be available at the time of prediction.
- **Removed Columns:** `Churn Score`, `Churn Reason`, and `CLTV`.
- **Reasoning:** These features are calculated *after* a customer has churned. Keeping them would allow the model to "cheat," resulting in an artificially high accuracy of 100% during training but complete failure in real-world deployment.

### B. Data Cleaning
- **Total Charges Transformation:** The `Total Charges` column was initially parsed as a text (`object`) datatype because 11 new customers had empty strings (" ") instead of numbers.
  - *Fix:* Converted to numeric (`float64`), forcing empty strings to become `NaN`. 
  - *Note:* Imputation (filling missing values) was intentionally deferred to the Modeling Pipeline (Milestone 3) to prevent data leakage between the train and test splits.
- **Duplicate Rows:** The dataset was scanned for complete duplicate rows. None were found, ensuring no biased weighting in the models.
- **Outliers:** Outliers in billing and tenure were retained. High-paying customers or extremely loyal customers are valid data points (VIPs) and tree-based models (Random Forest, XGBoost) are highly resilient to such outliers.

### C. Feature Pruning
- **Removed Columns:** `CustomerID`, `Country`, `State`, `City`, `Lat Long`, `Zip Code`.
- **Reasoning:** 
  - `CustomerID` possesses no predictive power and can cause overfitting.
  - Geographical data was highly granular (High Cardinality) and localized to a single country/state, making it noise rather than signal.

### D. Target Variable Encoding
- The `Churn Label` was encoded from textual `Yes`/`No` to binary `1`/`0`. This provides immediate compatibility with correlation matrices and mathematical models.

---

## 5. Output
The preprocessing phase successfully reduced the dataset from 33 noisy columns down to a clean, highly predictive set of 20 columns. 
- **Exported Artifact:** `cleaned_telco.csv`
- **Validation:** Data validation gates were implemented (via `assert` statements) to guarantee schema integrity, zero data leakage, and correct data types before exporting the dataset to the Feature Engineering team.
