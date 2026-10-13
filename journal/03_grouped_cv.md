# 03_grouped_cv

## Question / hypothesis

Does the Ridge vs Dummy RMSE gap survive when we use patient-grouped 5-fold CV
instead of the random row holdout — i.e., does the numeric signal generalise
to unseen patients?

## Motivation

- **Sourcing strategy:** user (Day 2 guide, step 3)
- **Source(s):**
  - Day 2 guide §3: "The same person has many visits. A random row split is leakage:
    visits from one patient sit on both sides, so the model memorizes that person
    instead of generalizing."
- **Why this matters:** `02_ridge` row-holdout RMSE (10.49) is optimistic. GroupKFold
  holds out whole patients — mirroring the Kaggle evaluation exactly. Every subsequent
  experiment uses these same `cv_splits` so comparisons are honest.

## Method

- **Files touched:** `experiments/03_grouped_cv.py` (self-contained script)
- **Change versus 02_ridge:** replace `splitter=0.2` with
  `cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=visits["patient_id"]))`.
  Both dummy and ridge are evaluated with this splitter. Each report is pushed
  individually to the hub (keys `03_dummy_cv`, `03_ridge_cv`).
- **Cross-validation:** GroupKFold(n_splits=5) on patient_id — whole patients held out.
- **Out of scope for this experiment:** changing the model, adding features, or
  changing alpha.

## Risks / things that could invalidate the result

- RMSE will increase versus the row holdout — that is expected and correct, not a
  regression. A higher number here means the row holdout was too optimistic.
- GroupKFold does not stratify on any outcome; fold sizes may vary if patient visit
  counts are unequal.

## Status

- **State:** done
- **Approved by user on:** 2025-07-11
- **Headline result:** Ridge RMSE 10.60 ± 0.27 vs Dummy 16.50 ± 0.34 (GroupKFold 5-fold, patient holdout)
- **Implication for next iteration:** Ridge generalises — the gap (36 %) survives the honest split. The RMSE barely moved from the row holdout (10.49→10.60), suggesting the numeric features encode visit-level rather than patient-specific patterns. Next: test HGBR which treats NaN as signal instead of imputing (B2).
