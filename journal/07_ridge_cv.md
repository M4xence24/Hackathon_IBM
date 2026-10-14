# 07_ridge_cv

## Question / hypothesis

Does searching over a broad log-spaced alpha grid with honest `GroupKFold(5)` reveal a
significantly better Ridge regularisation level than the default `alpha=1.0` used in
`02_ridge`, and how close can an optimally tuned linear model get to the HGBR baseline
of RMSE 7.53?

## Motivation

- **Sourcing strategy:** user (improve-results-plan.md Sub-Task 2)
- **Source(s):** improve-results-plan.md
- **Why this matters:** `02_ridge` fixed `alpha=1.0` with a leaky row holdout.
  EDA found `age_at_diagnosis` ↔ `age` collinearity (r=0.94) that could inflate Ridge
  coefficients without strong enough regularisation. We have never searched over alpha
  values nor evaluated Ridge under honest GroupKFold. Until this is done, we cannot
  claim to know whether a linear model has been given a fair chance.

## Method

- **Files touched:** `experiments/07_ridge_cv.py` (self-contained DataOps script)
- **Change versus 06_dataops:** same DataOps architecture and GroupKFold(5), but
  the estimator chain is `SimpleImputer("median") + RidgeCV(alphas=np.logspace(-3, 4, 30))`.
  Features are restricted to the 8 numeric `feature_cols` from `02_ridge` so the result
  is directly comparable to that baseline.
- **Cross-validation:** GroupKFold(n_splits=5), groups=patient_id — baked into the graph.
- **Hub key:** `07_ridge_cv`.
- **Submission:** `submissions/07_ridge_cv.csv` (Ridge refit on all training rows with
  best alpha from RidgeCV).

## Risks / things that could invalidate the result

- `RidgeCV` uses LOO/GCV internally — its internal alpha selection is not the same as
  the outer GroupKFold, so the best alpha visible from the report is the LOO-selected
  one, not a cross-validated grid. This is still informative for knowing whether 1.0
  was already near-optimal.
- Restricting to 8 numeric features means we cannot compare directly to 06_dataops
  (which uses all columns). The comparison is with 03_grouped_cv (Ridge + GroupKFold).

## Status

- **State:** done
- **Approved by user on:** 2025-07-11
- **Headline result:** RMSE 10.60 ± 0.27 (GroupKFold 5-fold, RidgeCV 30-alpha grid) — identical to 03_grouped_cv with alpha=1.0
- **Implication for next iteration:** Alpha tuning gives no benefit for Ridge on these 8 numeric features. The linear model ceiling is ~10.6 RMSE regardless of regularisation. This confirms HGBR (7.53) gains come from non-linearity + NaN routing, not from a Ridge calibration deficit. Hub: https://skore.probabl.ai/barthon/bobathon-esilv/cross-validations/74431
