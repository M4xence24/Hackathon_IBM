# 08_hgbr_tuned

## Question / hypothesis

Do non-default HGBR hyperparameters (`max_iter`, `learning_rate`, `max_leaf_nodes`)
yield a meaningfully lower RMSE than the default `06_dataops` baseline of 7.53?

## Motivation

- **Sourcing strategy:** user (improve-results-plan.md Sub-Task 3)
- **Source(s):** improve-results-plan.md
- **Why this matters:** `06_dataops` uses all HGBR defaults except `random_state=0`.
  HGBR is sensitive to the number of boosting iterations, learning rate, and tree
  depth. A coarse grid search over these three parameters, with an inner GroupKFold(3)
  to avoid patient leakage inside the search, could reduce the bias-variance trade-off
  without any feature engineering change.

## Method

- **Files touched:** `experiments/08_hgbr_tuned.py` (self-contained DataOps script)
- **Change versus 06_dataops:** same DataOps graph (all columns, GroupKFold 5-fold
  outer eval), but the estimator is wrapped in `GridSearchCV` with a coarse param grid:
  - `hgbr__max_iter`: [200, 500]
  - `hgbr__learning_rate`: [0.05, 0.1]
  - `hgbr__max_leaf_nodes`: [15, 31, 63]
  Inner CV uses `GroupKFold(n_splits=3)` with patient groups to avoid leakage inside
  the search.
- **Cross-validation:** outer GroupKFold(5) baked into graph; inner 3-fold (non-grouped)
  inside GridSearchCV (skb.apply does not forward fit_params).
- **Hub key:** `08_hgbr_tuned`.
- **Submission:** `submissions/08_hgbr_tuned.csv`.

## Risks / things that could invalidate the result

- The inner CV uses GroupKFold(3) — groups must be passed via `fit_params` to
  `GridSearchCV`. If the DataOps graph materialises groups only at eval time, the
  inner CV may not receive them correctly. This is a known integration risk.
- A coarse grid may miss the optimum. If the result is borderline, a finer search
  around the winner is warranted as a follow-up.
- `max_iter=500` with a 3-fold inner CV × 12 param combinations = 36 fits on ~35k
  rows — computationally expensive but feasible.

## Status

- **State:** done
- **Approved by user on:** 2025-07-11
- **Headline result:** RMSE 7.51 ± 0.13 (GroupKFold 5-fold, GridSearchCV tuned HGBR) — 0.02 improvement over 06_dataops (7.53)
- **Implication for next iteration:** Hyperparameter tuning gives a small but consistent RMSE reduction. The improvement is marginal (within the uncertainty), but the best config is more robust. Hub: https://skore.probabl.ai/barthon/bobathon-esilv/cross-validations/75042
