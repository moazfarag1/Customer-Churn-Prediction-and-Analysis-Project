# Known Limitations

Created During Phase 7 Final Engineering Readiness Audit  
Creation date: 2026-06-02

## Why This File Exists
This file gives professors, reviewers, and future maintainers an honest boundary around the release. It should be read with `MODEL_CARD.md`.

## Model Limitations
- The model was trained on a static historical dataset. Concept drift is not monitored.
- Recall is prioritized, but the default classification threshold still leaves some churners undetected.
- Probability calibration was not separately benchmarked.
- Highly correlated engineered features improve handoff stability but make individual Logistic Regression coefficient interpretation less reliable.
- The dataset omits richer behavioral signals such as support interactions and recent product usage.

## Deployment Limitations
- The application is a local Streamlit interface, not a multi-user production service.
- Pickle artifacts must come from a trusted source. Compatibility checks do not make untrusted pickle loading safe.
- The adapter is strict for normal UI use but is not a public API validation layer.
- The pinned Streamlit version is `1.31.0`; future upgrades should retest `st.image` compatibility.

## Reproducibility Limitations
- Direct dependencies are pinned in `requirements.txt`, but a transitive lock file was not generated during Phase 7 because the audit environment could not approve uv cache access.
- Validation must run from the repository root because it intentionally verifies repository-relative layout.

