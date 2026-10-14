# 01_dummy

## Question / hypothesis

What RMSE does a constant-mean predictor achieve on the Parkinson OFF motor score
data? This establishes the floor that all subsequent models must beat.

## Motivation

- **Sourcing strategy:** baseline (mandatory first step in any regression competition)
- **Why this matters:** Without a concrete floor we cannot quantify how much each
  modelling choice is worth. A DummyRegressor predicting the training mean is the
  simplest estimable model — if a proposed model cannot beat it, the model is useless.

## Method

- **Files touched:** `experiments/01_dummy.py`
- **Architecture:** `sklearn.dummy.DummyRegressor(strategy="mean")` — always predicts
  the mean of the training target.
- **Features:** 8 numeric clinical columns: `sexM`, `age_at_diagnosis`, `age`, `ledd`,
  `time_since_intake_on`, `time_since_intake_off`, `on`, `off`. The dummy ignores them.
- **Splitter:** row holdout (default `splitter=0.2` inside `skore.evaluate`).
  Patient-grouped CV is introduced in `03_grouped_cv`.
- **Hub key:** `01_dummy`.
- **Submission:** none (dummy is not submitted to Kaggle).

## Risks / things that could invalidate the result

- Row holdout can place multiple visits of the same patient on both sides of the split,
  making the RMSE estimate slightly optimistic compared with the Kaggle leaderboard
  (which holds out whole patients). This is intentional — the honest estimate is
  measured in `03_grouped_cv`.

## Status

- **State:** done
- **Approved by user on:** 2025-01-01
- **Headline result:** RMSE ≈ 16.5 (row holdout — consistent with SD of target ≈ 16.5)
- **Implication for next iteration:** Any model that learns even a weak signal should
  beat 16.5. Ridge on the same 8 features is the natural next step (`02_ridge`).
