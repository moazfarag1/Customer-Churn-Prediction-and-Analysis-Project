# Release Notes

Created During Phase 7 Final Engineering Readiness Audit  
Creation date: 2026-06-02

## Why This File Exists
This public release note summarizes the final stabilization work for reviewers without exposing internal audit prompts or scratch history.

## Release Candidate
Phase 7 packages the verified post-Phase 6A state.

## Stabilized Behavior
- Training-only median imputation lives inside the fitted Logistic Regression pipeline.
- Candidate selection uses validation data and preserves an untouched final holdout.
- Notebook replay uses deterministic single-process settings where Windows worker spawning was fragile.
- Notebook output writes are backup-aware and recorded in a regeneration manifest.
- App preprocessing uses NumPy-compatible rounding and matches training features across all 7,043 rows.
- Streamlit image rendering is compatible with pinned `streamlit==1.31.0`.
- Startup rejects stale model contracts and reordered schemas.

## Unchanged Release Artifacts
Phase 7 did not retrain the model, regenerate datasets, modify the schema, or change inference behavior.

