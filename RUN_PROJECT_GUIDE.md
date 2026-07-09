# Run Project Guide

Updated for the `GITHUB_VERSION` package.

## Purpose

This guide explains the shortest reliable path to install dependencies, verify the frozen runtime artifacts, and launch the Streamlit churn prediction app.

The project is already trained. You do not need to rerun notebooks to use the app.

## 1. Requirements

- Python `3.10` to `3.12`
- Internet access for the first dependency installation
- A terminal such as PowerShell, Command Prompt, Bash, or zsh

## 2. Open The Project Folder

Run commands from the `GITHUB_VERSION` folder, the folder that contains `requirements.txt`.

```powershell
cd "GITHUB_VERSION"
```

If you are already in the parent workspace, this command is enough.

## 3. Create A Virtual Environment

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it on Windows Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

## 4. Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Expected `pip check` result:

```text
No broken requirements found.
```

## 5. Verify Required Runtime Files

The Streamlit app requires these files:

```text
app/app.py
models/final_model_pipeline.pkl
models/feature_schema.pkl
assets/plots/eda/01_churn_distribution.png
assets/plots/eda/02_tenure_distribution.png
assets/plots/eda/04_contract_vs_churn.png
assets/plots/models/roc_curve_xgb_vs_lr.png
requirements.txt
```

Run this smoke check from the `GITHUB_VERSION` folder:

```powershell
python -c "from pathlib import Path; import joblib, pandas as pd; root=Path('.'); model=joblib.load(root/'models/final_model_pipeline.pkl'); schema=joblib.load(root/'models/feature_schema.pkl'); row={c:0 for c in schema}; row.update({'Gender':1,'Senior Citizen':0,'Partner':0,'Dependents':0,'Tenure Months':12,'Phone Service':1,'Multiple Lines':0,'Online Security':0,'Online Backup':0,'Device Protection':0,'Tech Support':0,'Streaming TV':0,'Streaming Movies':0,'Contract':0,'Paperless Billing':1,'Monthly Charges':70.0,'Total Charges':840.0,'Tenure_Group':0,'Num_Add_On_Services':0,'Has_Online_Services':0,'Avg_Monthly_Spend':70.0,'Is_Long_Term_Contract':0,'Internet Service_Fiber optic':1,'Internet Service_No':0,'Payment Method_Credit card (automatic)':0,'Payment Method_Electronic check':1,'Payment Method_Mailed check':0}); X=pd.DataFrame([row])[schema]; print('schema_len=', len(schema)); print('prediction=', int(model.predict(X)[0])); print('churn_probability=', round(float(model.predict_proba(X)[0,1]), 6))"
```

Expected behavior:

- The model loads.
- The schema loads with `27` features.
- A prediction and churn probability are printed.

## 6. Start The Streamlit App

```powershell
python -m streamlit run app/app.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

The app contains:

- Prediction App
- Dashboard & Insights
- Project & Model Summary

## 7. Normal Usage

Use the form in the Prediction App tab to enter a customer profile, then click `Predict Churn Risk`.

The app will:

- Validate impossible input combinations.
- Rebuild the required engineered features.
- Order inputs according to `models/feature_schema.pkl`.
- Run the frozen `models/final_model_pipeline.pkl`.
- Show the prediction and churn probability.

## 8. Optional Notebook Reproduction

The notebooks are included for study and reproduction, not for normal app usage.

If you need to reproduce the development workflow, run them in this order:

```text
notebooks/01_moaz_eda_preprocessing.ipynb
notebooks/02_mohy_feature_engineering.ipynb
notebooks/03_mahmoud_modeling_baseline.ipynb
notebooks/04_ali_xgboost_optimization.ipynb
```

After rerunning notebooks, confirm that `models/final_model_pipeline.pkl` and `models/feature_schema.pkl` still load and the smoke check in section 5 still succeeds.

## 9. Troubleshooting

### `ModuleNotFoundError`

The virtual environment is inactive or dependencies were not installed. Activate `.venv`, then rerun:

```powershell
python -m pip install -r requirements.txt
```

### PowerShell blocks activation

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

This changes policy only for the current terminal session.

### Model or schema load fails

Restore these files from the package:

```text
models/final_model_pipeline.pkl
models/feature_schema.pkl
```

Do not rename them. `app/app.py` loads those exact paths.

### Dashboard image is missing

Restore the referenced image under `assets/plots/`. The app can still predict, but the dashboard tab will show a missing-image message.

### Streamlit opens but prediction fails

Run the smoke check in section 5. If that succeeds, restart Streamlit. If it fails, the model, schema, or installed dependency versions are inconsistent.

## 10. Expected Successful State

- `python -m pip check` reports no broken requirements.
- The smoke check prints `schema_len= 27`.
- Streamlit starts without a traceback.
- The prediction form produces a churn prediction and probability.
