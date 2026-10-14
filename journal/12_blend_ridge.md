# 12_blend_ridge

## Question / hypothesis

Does blending the residual HGBR model (`11_residual_patient` architecture) with a Ridge
regression on numeric features further reduce RMSE, and what is the optimal blend weight?

## Motivation

- **Sourcing strategy:** user (written outside the Bob iteration loop)
- **Source(s):** `experiments/improve_rmse.py`
- **Why this matters:** HGBR captures non-linear interactions and missingness patterns
  well but can have higher variance than a linear model on smooth numeric signals (age,
  ledd, disease duration). Ridge is a smoothed linear predictor — blending the two at
  inference time exploits their complementary strengths. The blend weight is selected from
  OOF predictions on the same 5 GroupKFold splits, so no data leakage enters the choice.

## Method

- **Files touched:** `experiments/improve_rmse.py` (flat jupytext script)
- **Architecture:** `BlendRegressor` — `w * HGBR_pred + (1 − w) * Ridge_pred`.
  - HGBR component: `ResidualRegressor` wrapping `HistGradientBoostingRegressor` with
    early stopping (`max_iter=2000`, `learning_rate=0.03`, `early_stopping=True`,
    `validation_fraction=0.1`, `n_iter_no_change=50`), `max_leaf_nodes=31`,
    `min_samples_leaf=40`, `l2_regularization=1.0`. `rater_col=None` (rater bias
    encoding was tested and found not to help).
  - Ridge component: `SimpleImputer(median) + StandardScaler + Ridge(alpha=10)` on
    numeric columns only (no categorical codes).
  - Blend weight `w` chosen from `{0.0, 0.25, 0.5, 0.75, 1.0}` via OOF RMSE on the
    same GroupKFold(5) splits. **Best w = 0.75**.
- **Feature engineering:** same `build_features` as `11_test.py`, plus temporal
  neighbourhood features:
  - `off_prev`, `off_next`, `on_prev` — previous/next visit OFF/ON scores (shift by
    age order within patient)
  - `dt_prev` — time delta to previous visit
  - `off_delta_prev` — change in OFF from previous visit
  - `p_off_slope` — per-patient OLS slope of OFF vs age (disease progression rate)
- **Cross-validation:** `GroupKFold(n_splits=5)`, `groups=patient_id`.
- **Hub key:** `09_blend_ridge`.
- **Submission:** `submissions/09_blend_ridge.csv`.

## OOF weight search results

| w (HGBR fraction) | OOF RMSE |
|---|---|
| 0.00 (Ridge only) | 5.0187 |
| 0.25 | 4.6424 |
| 0.50 | 4.3786 |
| **0.75** | **4.2483** ← selected |
| 1.00 (HGBR only) | 4.2638 |

Ridge alone scores 5.02; the pure HGBR is 4.26; blending at w=0.75 squeezes out an
additional 0.02 points over pure HGBR.

## Risks / things that could invalidate the result

- Same train+test combined feature building risk as `10_test.py`/`11_test.py`.
- `off_next` / `off_prev` use values from adjacent visits sorted by age. At test time,
  the model may use test-set neighbour visits for the combined-concat feature build,
  which is mildly optimistic.
- The blend weight is selected on the same OOF folds used for the final `evaluate` call.
  This is not a held-out selection — the weight could be slightly overfitted to these 5
  folds. The gain over w=1.0 is small (0.02 RMSE), so this risk is low.
- Early stopping uses a random 10 % internal validation split inside each training fold,
  which introduces seed-dependent noise in the number of iterations. See `step6_seeds.py`
  for a quantification of this noise.

## Status

- **State:** done
- **Approved by user on:** 2025-07-11 (run by user outside Bob iteration loop)
- **Headline result:** RMSE 4.25 ± 0.09 (GroupKFold 5-fold, BlendRegressor w=0.75) — **new best**; 7 % below 11_residual_patient (4.58)
- **Implication for next iteration:** Temporal neighbourhood features (prev/next visit OFF, OFF slope) and the Ridge blend together yield another 0.33 RMSE reduction. The blend gain over pure HGBR is marginal (0.02); the main driver is the richer feature set. Next direction: more temporal features or a finer weight grid.
