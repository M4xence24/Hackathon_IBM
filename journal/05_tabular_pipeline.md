# 05_tabular_pipeline

## Question / hypothesis

Does including string columns (`gene`, `cohort`) via skrub's `tabular_pipeline`
further lower RMSE below HGBR's 7.75 on the same patient-grouped 5-fold CV?

## Motivation

- **Sourcing strategy:** user (Day 2 guide, step 5)
- **Source(s):**
  - Day 2 guide §5: "Ridge and the numeric HGBR above never saw `gene` or `cohort`.
    Something has to turn text into numbers first. `tabular_pipeline('regressor')` is
    a ready-made two-step recipe: (1) `TableVectorizer` turns a messy table into a
    numeric matrix, (2) `HistGradientBoostingRegressor` predicts."
- **Why this matters:** `04_hgbr` used only numeric columns. Genetics (`gene`) and
  clinical cohort (`cohort`) may encode disease subtype or protocol differences that
  correlate with the true OFF score. `TableVectorizer` picks an encoder per column
  automatically (one-hot for low cardinality, StringEncoder for high cardinality).

## Method

- **Files touched:** `experiments/05_tabular_pipeline.py` (self-contained script)
- **Change versus 04_hgbr:** use `skrub.tabular_pipeline("regressor")` and expand
  features to all columns except `Index`, `patient_id`, and `target`. No manual
  column selection. Same `cv_splits` (GroupKFold 5-fold on patient_id).
- **Cross-validation:** GroupKFold(n_splits=5) on patient_id — identical to 03 and 04.
- **Hub key:** `05_tabular_pipeline`.
- **Out of scope for this experiment:** custom encoders, hyperparameter tuning,
  skrub DataOps graph (step 6 of the guide).

## Risks / things that could invalidate the result

- `patient_id` and `Index` must be dropped before fitting; leaving them in would
  allow the model to memorise patient identity (leakage).
- `TableVectorizer` may choose a high-cardinality encoder for `gene` that adds
  many columns; if the signal is weak, this could hurt via variance.
- Test features must use the same column drops as training.

## Status

- **State:** done
- **Approved by user on:** 2025-07-11
- **Headline result:** RMSE 7.53 ± 0.14 (GroupKFold 5-fold, patient holdout) — 3 % below HGBR 7.75
- **Implication for next iteration:** Adding gene/cohort via TableVectorizer gives a modest but consistent gain. The full feature set is now in use. Next steps could explore temporal features (disease duration, visit ordering) or skrub DataOps to attach groups directly on the graph (Day 2 step 6).
