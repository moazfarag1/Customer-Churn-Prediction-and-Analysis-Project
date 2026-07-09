# Phase 3 Baseline Model Comparison

## Source of Truth
- Input used: `data/cleaned/processed_telco.csv`
- Shape: 7,043 rows × 28 columns
- Target: `Churn Label` where 1 = Churned and 0 = Not Churned
- No raw data, cleaned data, preprocessing recreation, encoding recreation, or feature engineering recreation was used.
- Leakage columns excluded: `Churn Score`, `Churn Value`, `Churn Reason`, `CLTV`

## Split Summary
| Dataset | Rows | Class_0_Count | Class_1_Count | Churn_Rate |
| --- | --- | --- | --- | --- |
| Full | 7043 | 5174 | 1869 | 0.2654 |
| Train | 4225 | 3104 | 1121 | 0.2653 |
| Validation | 1409 | 1035 | 374 | 0.2654 |
| Untouched Test | 1409 | 1035 | 374 | 0.2654 |

Split rule: deterministic 64% train / 16% validation / 20% untouched test using `random_state=42` and `stratify`.

## Evaluation Priority
1. Recall
2. F1-score
3. ROC-AUC
4. Precision
5. Accuracy

## Baseline Metrics
| Model | Recall | F1 | ROC_AUC | Precision | Accuracy |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.5695 | 0.6192 | 0.8499 | 0.6783 | 0.8141 |
| Random Forest | 0.4866 | 0.5482 | 0.8334 | 0.6276 | 0.7871 |

## Classification Reports
### Logistic Regression
```text
              precision    recall  f1-score   support

 Not Churned       0.85      0.90      0.88      1035
     Churned       0.68      0.57      0.62       374

    accuracy                           0.81      1409
   macro avg       0.77      0.74      0.75      1409
weighted avg       0.81      0.81      0.81      1409

```

### Random Forest
```text
              precision    recall  f1-score   support

 Not Churned       0.83      0.90      0.86      1035
     Churned       0.63      0.49      0.55       374

    accuracy                           0.79      1409
   macro avg       0.73      0.69      0.70      1409
weighted avg       0.78      0.79      0.78      1409

```

## Redundancy and Multicollinearity Validation
This analysis was performed after the three-way split and used `X_train` for correlation review. Baseline comparison uses validation only; the untouched test is reserved for Phase 4 final reporting.

Required redundancy features evaluated:
- `Total Charges`
- `Tenure Months`
- `Avg_Monthly_Spend`
- `Is_Long_Term_Contract`

### Training-Only Correlation Matrix
| Feature | Total Charges | Tenure Months | Avg_Monthly_Spend | Is_Long_Term_Contract | Contract | Monthly Charges |
| --- | --- | --- | --- | --- | --- | --- |
| Total Charges | 1.0 | 0.834 | 0.651 | 0.458 | 0.461 | 0.65 |
| Tenure Months | 0.834 | 1.0 | 0.26 | 0.653 | 0.677 | 0.259 |
| Avg_Monthly_Spend | 0.651 | 0.26 | 1.0 | -0.052 | -0.064 | 0.996 |
| Is_Long_Term_Contract | 0.458 | 0.653 | -0.052 | 1.0 | 0.917 | -0.054 |
| Contract | 0.461 | 0.677 | -0.064 | 0.917 | 1.0 | -0.066 |
| Monthly Charges | 0.65 | 0.259 | 0.996 | -0.054 | -0.066 | 1.0 |

### High-Correlation Pairs
| Feature_A | Feature_B | Abs_Correlation |
| --- | --- | --- |
| Total Charges | Tenure Months | 0.8340 |
| Avg_Monthly_Spend | Monthly Charges | 0.9960 |
| Is_Long_Term_Contract | Contract | 0.9170 |

### Feature-Set Variant Results
| Model | Feature_Set | Feature_Count | Dropped_Features | Recall | F1 | ROC_AUC | Precision | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | drop_avg_monthly_spend | 26 | Avg_Monthly_Spend | 0.5722 | 0.6212 | 0.8499 | 0.6794 | 0.8148 |
| Logistic Regression | drop_derived_redundant | 25 | Avg_Monthly_Spend, Is_Long_Term_Contract | 0.5695 | 0.6192 | 0.8501 | 0.6783 | 0.8141 |
| Logistic Regression | all_features | 27 | None | 0.5695 | 0.6192 | 0.8499 | 0.6783 | 0.8141 |
| Logistic Regression | drop_is_long_term_contract | 26 | Is_Long_Term_Contract | 0.5695 | 0.6183 | 0.8499 | 0.6762 | 0.8133 |
| Logistic Regression | drop_financial_contract_redundant_combo | 24 | Total Charges, Avg_Monthly_Spend, Is_Long_Term_Contract | 0.5668 | 0.6172 | 0.8494 | 0.6773 | 0.8133 |
| Logistic Regression | drop_tenure_months | 26 | Tenure Months | 0.5588 | 0.6174 | 0.8495 | 0.6898 | 0.8162 |
| Logistic Regression | drop_total_charges | 26 | Total Charges | 0.5588 | 0.6120 | 0.8491 | 0.6764 | 0.8119 |
| Random Forest | drop_financial_contract_redundant_combo | 24 | Total Charges, Avg_Monthly_Spend, Is_Long_Term_Contract | 0.4947 | 0.5564 | 0.8302 | 0.6357 | 0.7906 |
| Random Forest | drop_total_charges | 26 | Total Charges | 0.4866 | 0.5540 | 0.8321 | 0.6431 | 0.7921 |
| Random Forest | all_features | 27 | None | 0.4866 | 0.5482 | 0.8334 | 0.6276 | 0.7871 |
| Random Forest | drop_tenure_months | 26 | Tenure Months | 0.4840 | 0.5510 | 0.8295 | 0.6396 | 0.7906 |
| Random Forest | drop_avg_monthly_spend | 26 | Avg_Monthly_Spend | 0.4813 | 0.5438 | 0.8289 | 0.6250 | 0.7857 |
| Random Forest | drop_is_long_term_contract | 26 | Is_Long_Term_Contract | 0.4759 | 0.5477 | 0.8302 | 0.6449 | 0.7913 |
| Random Forest | drop_derived_redundant | 25 | Avg_Monthly_Spend, Is_Long_Term_Contract | 0.4679 | 0.5319 | 0.8306 | 0.6162 | 0.7814 |

### Redundancy Decision
- The official exported Phase 3 baselines are trained on the full processed Phase 2 schema for a stable handoff.
- The feature-set variants above show whether removing `Total Charges`, `Tenure Months`, `Avg_Monthly_Spend`, or `Is_Long_Term_Contract` improves or harms Recall/F1/ROC-AUC.
- Do not modify `processed_telco.csv`; use this evidence for Phase 4 model comparison and discussion.

## Random Forest Feature Importance Note
Random Forest feature importance can be unstable when highly correlated or derived features exist, so the plot is interpretive guidance only.

- Probability calibration was not performed in Phase 3 and may be evaluated in Phase 4.
## Streamlit Compatibility Notes
- Saved artifacts are fitted sklearn Pipelines.
- Training-only median imputation and Logistic Regression scaling are inside `logistic_regression_pipeline.pkl`.
- Random Forest has no scaler.
- Future Streamlit inference must provide the same 27 feature columns in the same order.
- No manual inference scaling should be implemented.

## Exported Artifacts
- `models/logistic_regression_pipeline.pkl`
- `models/random_forest_pipeline.pkl`
- `assets/plots/models/confusion_matrix_lr.png`
- `assets/plots/models/confusion_matrix_rf.png`
- `assets/plots/models/roc_curve.png`
- `assets/plots/models/feature_importance_rf.png`

## Final Verdict
Phase 3 baseline modeling artifacts are ready for Phase 4 comparison.