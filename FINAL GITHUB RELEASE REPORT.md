# Final GitHub Release Report

Certification date: 2026-06-19

Target directory: `GITHUB_VERSION/`

## Certification Decision

`GITHUB_VERSION CERTIFIED FOR PUBLIC RELEASE`

The repository was validated as portable, reproducible, runnable, and suitable for public GitHub release after one required execution repair.

## Tests Executed

### Clean Machine Simulation

Executed on a fresh disposable copy of `GITHUB_VERSION/` outside the target repository.

- Created a fresh virtual environment.
- Installed from `requirements.txt`.
- Verified installed package compatibility.
- Re-ran notebooks from the copied repository.
- Launched Streamlit from the copied repository.
- Loaded model and schema from the copied repository.
- Ran multiple prediction scenarios.

Final documented install command path passed:

```text
python -m venv .pip_venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Result:

```text
No broken requirements found.
PIP_INSTALL_VERIFICATION=PASS 2.2.0 1.26.4 1.7.2 1.31.0 1.3.2 2.0.3
```

### Notebook Execution

Executed all required notebooks in order with `jupyter nbconvert --execute`:

| Notebook | Result | Runtime |
| --- | --- | --- |
| `notebooks/01_moaz_eda_preprocessing.ipynb` | PASS | 9.48 seconds |
| `notebooks/02_mohy_feature_engineering.ipynb` | PASS | 11.82 seconds |
| `notebooks/03_mahmoud_modeling_baseline.ipynb` | PASS | 15.60 seconds |
| `notebooks/04_ali_xgboost_optimization.ipynb` | PASS | 12.57 seconds |

### Runtime Inference Verification

Loaded:

- `models/final_model_pipeline.pkl`
- `models/feature_schema.pkl`

Verified:

- schema length is 27;
- pipeline steps are `imputer`, `scaler`, `model`;
- predictions and probabilities work;
- required dashboard image assets open successfully.

Prediction scenarios:

| Scenario | Prediction | Churn Probability |
| --- | --- | --- |
| `fiber_month_to_month` | 1 | 0.661757 |
| `long_contract_low_risk` | 0 | 0.008581 |
| `zero_tenure_edge` | 0 | 0.475294 |

Result:

```text
FINAL_INFERENCE=PASS
```

### Streamlit Application Verification

Launched:

```text
python -m streamlit run app/app.py --server.headless=true
```

Result:

```text
FINAL_STREAMLIT=PASS status=HTTP_200
```

Application logs were scanned for:

- `Traceback`
- `ModuleNotFoundError`
- `ImportError`
- `FileNotFoundError`
- `Image not found`
- `Critical Error`
- `No such file`

No blocking runtime errors were found.

### Path Portability Verification

Scanned `GITHUB_VERSION/` for machine-specific Windows drive roots, user profile folders, temporary Linux paths, old local workspace names, and other local-only path markers.

No local filesystem paths were found.

The only matches were public GitHub clone URLs in `README.md` and `DEPLOYMENT_GUIDE.md`, which are not machine-specific paths.

### Repository Cleanliness Verification

Scanned for generated cache artifacts:

- `.venv`
- `__pycache__`
- `*.pyc`
- `.ipynb_checkpoints`
- `.DS_Store`

No such artifacts were found inside `GITHUB_VERSION/`.

Required runtime and reproduction files were present and non-empty:

| File | Status |
| --- | --- |
| `app/app.py` | PASS |
| `models/final_model_pipeline.pkl` | PASS |
| `models/feature_schema.pkl` | PASS |
| `data/raw/telco_customer_churn.csv` | PASS |
| `data/cleaned/cleaned_telco.csv` | PASS |
| `data/cleaned/processed_telco.csv` | PASS |
| `requirements.txt` | PASS |
| `runtime_audit_utils.py` | PASS |
| `notebooks/01_moaz_eda_preprocessing.ipynb` | PASS |
| `notebooks/02_mohy_feature_engineering.ipynb` | PASS |
| `notebooks/03_mahmoud_modeling_baseline.ipynb` | PASS |
| `notebooks/04_ali_xgboost_optimization.ipynb` | PASS |

## Issues Found

### Issue 1: Missing Notebook Utility Module

Initial notebook execution failed at notebook 01 with:

```text
ModuleNotFoundError: No module named 'runtime_audit_utils'
```

Cause:

The notebooks import:

```python
from runtime_audit_utils import backup_if_overwriting, record_regeneration_step
```

but `runtime_audit_utils.py` was not included in `GITHUB_VERSION/`.

Impact:

A fresh clone could launch the Streamlit app, but could not reproduce notebooks from scratch. This blocked public-release certification.

Repair:

Added `GITHUB_VERSION/runtime_audit_utils.py` using the validated project utility behavior:

- `sha256_file`
- `backup_if_overwriting`
- `record_regeneration_step`

This repair preserves methodology and only restores the missing notebook dependency.

### Issue 2: README Structure Drift

The README listed:

```text
notebooks/            # Sequential Jupyter notebooks (01 to 06)
tests/                # Validation scripts
```

but the public `GITHUB_VERSION/` package contains notebooks 01 to 04 and does not contain a `tests/` directory.

Impact:

This did not affect runtime, but it made the public README inaccurate.

Repair:

Updated the README project structure to:

```text
notebooks/            # Sequential Jupyter notebooks (01 to 04)
runtime_audit_utils.py # Shared notebook utility for backups and regeneration manifests
```

No methodology, model, app behavior, dependencies, or metrics were changed.

## Repairs Performed

| File | Repair |
| --- | --- |
| `runtime_audit_utils.py` | Added missing notebook support module required for reproducible notebook execution. |
| `README.md` | Corrected public repository structure description. |
| `FINAL GITHUB RELEASE REPORT.md` | Added this required certification report. |

## Files Modified

- `GITHUB_VERSION/runtime_audit_utils.py`
- `GITHUB_VERSION/README.md`
- `GITHUB_VERSION/FINAL GITHUB RELEASE REPORT.md`

## Dependencies Updated

None.

`requirements.txt` was not modified because dependency installation and compatibility checks passed.

## Runtime Verification

Runtime verification passed:

- model loads;
- schema loads;
- predictions work;
- probability outputs are valid;
- required dashboard assets open;
- Streamlit launches and returns HTTP 200.

## Notebook Verification

Notebook verification passed after restoring `runtime_audit_utils.py`:

- notebook 01 passed;
- notebook 02 passed;
- notebook 03 passed;
- notebook 04 passed.

The notebook methodology was not changed.

## Application Verification

Application verification passed:

- app startup passed;
- model loading passed;
- schema loading passed;
- prediction passed;
- preprocessing contract passed through schema-ordered inference;
- no missing asset messages or import errors were found in Streamlit logs.

## GitHub Portability Verification

The repository is portable because:

- paths are project-relative;
- no local absolute filesystem paths remain;
- `.gitignore` excludes common generated artifacts;
- dependency installation from `requirements.txt` passed in a fresh environment;
- notebooks execute from the repository root;
- Streamlit resolves artifacts relative to `app/app.py`;
- required model, schema, data, notebooks, and assets are included.

## Final Statement

All required components executed successfully after the minimal repair:

- dependencies install;
- notebooks execute;
- model loads;
- schema loads;
- predictions succeed;
- Streamlit launches;
- no broken paths remain;
- no missing dependencies remain;
- repository is reproducible and portable.

`GITHUB_VERSION CERTIFIED FOR PUBLIC RELEASE`
