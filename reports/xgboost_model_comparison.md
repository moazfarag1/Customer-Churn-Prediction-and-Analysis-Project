# Phase 4 XGBoost Optimization Report

## Source of Truth
- Data: `data/cleaned/processed_telco.csv`
- Feature schema: `models/feature_schema.pkl`
- Phase 3 baseline: `models/logistic_regression_pipeline.pkl`
- No raw data, preprocessing recreation, encoding recreation, or feature engineering recreation was used.

## Optimization Strategy
- Model: `XGBClassifier` inside an sklearn `Pipeline`
- Search: very small `GridSearchCV`
- CV: 3-fold stratified
- Refit metric: Recall
- Parameter grid size: 8 candidates
- No SMOTE, resampling, threshold tuning, calibration, SHAP, Optuna, or ensembling.

## Best XGBoost Parameters
```text
{'model__learning_rate': 0.1, 'model__max_depth': 3, 'model__n_estimators': 100}
```

## Validation Model Comparison
| Model | Recall | F1 | ROC_AUC | Precision | Accuracy |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.5695 | 0.6192 | 0.8499 | 0.6783 | 0.8141 |
| XGBoost Optimized | 0.5642 | 0.6125 | 0.8563 | 0.6698 | 0.8105 |
| XGBoost Baseline | 0.5455 | 0.5982 | 0.8569 | 0.6623 | 0.8055 |

## Untouched Final Holdout
| Model | Recall | F1 | ROC_AUC | Precision | Accuracy |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression Final Holdout | 0.5722 | 0.6054 | 0.8483 | 0.6426 | 0.8020 |

## Final Model Decision
- Selected final model: `Logistic Regression`
- Selection used validation metrics only. The untouched final holdout was evaluated once after refitting on train + validation.
- Selection priority: Recall, then F1, then ROC-AUC.
- If Logistic Regression remains best, that is an acceptable and honest modeling result.

## Streamlit Handoff Notes
- Use `models/final_model_pipeline.pkl` for the app unless the team explicitly chooses the XGBoost-only artifact.
- Use `models/feature_schema.pkl` to create the exact 27 input columns in the exact order.
- Do not manually scale inputs. Any needed transformations are inside saved pipelines.
- Keep `processed_telco.csv` and `feature_schema.pkl` unchanged.

## Exported Artifacts
- `models/xgboost_pipeline.pkl`
- `models/final_model_pipeline.pkl`
- `assets/plots/models/confusion_matrix_xgb.png`
- `assets/plots/models/roc_curve_xgb_vs_lr.png`
- `assets/plots/models/feature_importance_xgb.png`

## Final Verdict
Phase 4 is complete and ready for the Streamlit application phase.