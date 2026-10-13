# 04_hgbr

## Question / hypothesis

Does treating NaN as signal (no imputation) with HistGradientBoostingRegressor
lower RMSE below Ridge's 10.60 on the same patient-grouped 5-fold CV?

## Motivation

- **Sourcing strategy:** user (Day 2 guide, step 4)
- **Source(s):**
  - Day 2 guide §4: "In this table, a missing OFF often means the visit was ON-only
    (uncomfortable OFF exams are skipped). That is information about the patient and
    the protocol, not noise to impute away."
- **Why this matters:** `03_grouped_cv` showed Ridge still needs to impute NaNs with
  the median, erasing the clinical signal carried by missingness. HGBR natively routes
  NaN rows to a learned branch at each tree split — the pattern "OFF was not measured"
  can directly predict a higher true OFF score.

## Method

- **Files touched:** `experiments/04_hgbr.py` (self-contained script)
- **Change versus 03_grouped_cv:** replace the `make_pipeline(SimpleImputer, Ridge)`
  with a bare `HistGradientBoostingRegressor(random_state=0)`. No imputation step.
  Same `feature_cols`, same `cv_splits` (GroupKFold 5-fold on patient_id).
- **Cross-validation:** GroupKFold(n_splits=5) on patient_id — identical to 03.
- **Hub key:** `04_hgbr`.
- **Out of scope for this experiment:** string columns (gene/cohort), skrub
  TableVectorizer, hyperparameter tuning.

## Risks / things that could invalidate the result

- HGBR is a non-linear ensemble; if the true OFF signal is close to linear, the gain
  over Ridge may be modest.
- `random_state=0` ensures reproducibility but does not eliminate stochasticity in
  the gradient boosting path.

## Status

- **State:** done
- **Approved by user on:** 2025-07-11
- **Headline result:** RMSE 7.75 ± 0.15 (GroupKFold 5-fold, patient holdout) — 27 % below Ridge 10.60
- **Implication for next iteration:** Treating NaN as signal is a major win. The model learned that a missing OFF predicts a higher true OFF score. Next: include string columns (gene/cohort) via skrub tabular_pipeline to see if they add further signal (B3).
