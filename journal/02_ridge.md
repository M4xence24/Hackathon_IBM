# 02_ridge

## Question / hypothesis

Does a Ridge regression on the numeric clinical features (age, on, off, ledd, delays)
produce a lower RMSE than the dummy mean baseline?

## Motivation

- **Sourcing strategy:** user (Day 2 guide, step 1)
- **Source(s):**
  - Day 2 guide §1: "Numeric clinical variables may carry a simple additive signal.
    A first check that the numbers matter at all."
- **Why this matters:** The dummy model ignores all features and always predicts the
  training mean. If Ridge cannot beat it, the numeric columns carry no usable signal
  and more complex models are not worth pursuing. The guide also asks us to compare
  both models in a single `ComparisonReport` pushed to the hub.

## Method

- **Files touched:** `experiments/02_ridge.py` (self-contained script, same style as `dummy.py`)
- **Change versus 01_dummy:** replace `DummyRegressor` with
  `make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0))`. Numeric columns
  need median imputation because `on`, `off`, `ledd` have missing values. Pass both
  `dummy` and `ridge` as a dict to `skore.evaluate` to get a `ComparisonReport`.
- **Cross-validation:** row holdout, `splitter=0.2` (random 20 % test split — same as
  dummy; visit-level leakage is the acknowledged limitation, addressed in B1).
- **Hub key:** `02_ridge` — pushed via `project.put("02_ridge", report)`.
- **Kaggle submission:** `submissions/02_ridge.csv` — Ridge refit on all training rows,
  predict on `X_test`, header `Index,target`.
- **Out of scope for this experiment:** grouped CV, string columns (gene/cohort),
  alpha tuning, any tree-based model.

## Risks / things that could invalidate the result

- The row holdout (`splitter=0.2`) puts the same patients on both sides, so RMSE
  will look better than the Kaggle grouped holdout. This is expected and documented
  in the guide; the comparison to dummy is still meaningful as a relative check.
- Ridge assumes a linear relationship; if the true OFF score is highly non-linear,
  the improvement over dummy will be small even if signal exists.

## Status

- **State:** done
- **Approved by user on:** 2025-01-01
- **Headline result:** RMSE 10.49 (row holdout, splitter=0.2) vs Dummy RMSE 16.48 — 36 % reduction
- **Implication for next iteration:** Ridge extracts real signal from the numeric features, but the row holdout leaks same-patient visits into both sides. Next step: GroupKFold on patient_id to get an honest estimate and check if the gap survives the stricter split (B1).
