# improve_rmse_skore

## Question / hypothesis

Can a `ResidualRegressor` (HGBR on `y − clinical_baseline`) with levodopa decay
features and patient-level aggregates beat `09_patient_features` (RMSE 5.38)?

## Motivation

- **Sourcing strategy:** user (written outside the Bob iteration loop as a rapid
  prototype; later superseded by `10_residual_patient.py` and `11_residual_patient.py`)
- **Why this matters:** The clinical `off` score correlates at r=0.87 with the target.
  Anchoring predictions on a clinically meaningful baseline (off if observed, else
  on + median offset) and learning only the small residual should reduce model variance.
  Levodopa decay features model pharmacokinetics explicitly.

## Method

- **Files touched:** `experiments/improve_rmse_skore.py`
- **Architecture:** `ResidualRegressor` sklearn wrapper:
  1. Baseline = `off` if observed, else `on + median(off − on)`, else global median.
  2. Fits `HistGradientBoostingRegressor` on `y − baseline`.
  3. Predict: `baseline + hgbr.predict(X)`.
- **Feature engineering (via `build_features`):**
  - Missingness flags: `off_missing`, `on_missing`
  - Difference: `on_minus_off`
  - Levodopa decay: `ledd_decay_2.0`, `ledd_decay_4.0` (exponential half-lives 2h/4h)
  - Patient-level aggregates: `p_n_visits`, `p_mean_on`, `p_mean_off`, `p_mean_ledd`,
    `p_mean_age`, `age_since_first`, `visit_rank`
  - Categorical codes: `gene` NaN → "unknown"; `gene`, `cohort`, `rater_id` ordinal-encoded.
- **HGBR params:** `learning_rate=0.05`, `max_iter=500`, `max_leaf_nodes=31`,
  `min_samples_leaf=40`, `l2_regularization=1.0`, `random_state=0`.
- **Cross-validation:** `GroupKFold(n_splits=5)`, `groups=patient_id`.
- **Hub key:** `07_residual_patient` (shared with `10_residual_patient` and
  `11_residual_patient` — last run overwrites on hub).
- **Submission:** `submissions/07_residual_patient.csv`.

## Risks / things that could invalidate the result

- Train + test combined feature building: patient aggregates are computed over the
  full concat of train + test rows — test-patient statistics bleed into training folds,
  making the CV RMSE slightly optimistic.
- Hub key `07_residual_patient` is overwritten by `10_residual_patient.py` and
  `11_residual_patient.py` on subsequent runs.

## Status

- **State:** done (prototype — superseded by `10_residual_patient` and `11_residual_patient`)
- **Approved by user on:** 2025-07-11 (run outside Bob iteration loop)
- **Headline result:** RMSE ≈ 4.9 (GroupKFold 5-fold, ResidualRegressor + levodopa decay
  + basic patient aggregates) — improvement over `09_patient_features` (5.38).
- **Implication for next iteration:** This prototype established the residual approach.
  `10_residual_patient` and `11_residual_patient` extend it with a richer feature set
  and are the canonical versions to reference.
