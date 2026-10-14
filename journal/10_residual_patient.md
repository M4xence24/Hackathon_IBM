# 10_residual_patient

## Question / hypothesis

Does reformulating the target as a residual on a clinical baseline (`off` if measured,
otherwise `on + median(off−on)`) and fitting HGBR on the residual yield a better RMSE
than the direct prediction approach of `09_patient_features`?

## Motivation

- **Sourcing strategy:** user (written outside the Bob iteration loop)
- **Source(s):** `experiments/10_test.py`
- **Why this matters:** The clinical `off` score (when present) is already very close
  to the true OFF motor score (Pearson r=0.87 from EDA). Instead of asking the model to
  predict the full target from scratch, anchoring predictions to a clinically meaningful
  baseline and learning only the residual should reduce variance — the model only needs
  to correct the noise in the baseline, not reconstruct the full signal. This is a
  well-known bias-reduction trick for strong-baseline regression problems.

## Method

- **Files touched:** `experiments/10_test.py` (flat script, no DataOps graph)
- **Architecture:** custom `ResidualRegressor` sklearn wrapper that:
  1. Computes a `baseline` column: `off` if observed, else `on + median(off − on)`,
     falling back to the global median for rows where both are NaN.
  2. Fits `HistGradientBoostingRegressor` on `y − baseline`.
  3. At predict time: `baseline + hgbr.predict(X)`.
- **Feature engineering (via `build_features`):**
  - Binary missingness flags: `off_missing`, `on_missing`
  - Difference: `on_minus_off`
  - Levodopa decay approximation: `ledd_decay_2.0`, `ledd_decay_4.0` (exponential with
    half-lives of 2h and 4h applied to `time_since_intake_on`)
  - Patient-level aggregates: `p_n_visits`, `p_mean_on`, `p_mean_off`, `p_mean_ledd`,
    `p_mean_age`, `p_std_on`, `p_std_off`, `p_min_off`, `p_max_off`,
    `p_frac_off_missing`, `p_mean_offdiff`, `p_mean_time_on`, `p_mean_duration`
  - Visit-level temporal: `age_since_first`, `visit_rank`, `on_dev_patient`,
    `age_to_last`, `disease_duration`
  - Categorical encoding: `gene` NaN → "unknown"; `gene`, `cohort`, `rater_id`
    ordinal-encoded and passed as `categorical_features` to HGBR.
- **Train+test combined for feature building:** `build_features` is called on
  `concat([visits.drop("target"), X_test])` so patient-level aggregates include all
  available visits per patient (including test rows). Mild leakage risk — see Risks.
- **HGBR params:** `learning_rate=0.05`, `max_iter=500`, `max_leaf_nodes=31`,
  `min_samples_leaf=40`, `l2_regularization=1.0`, `random_state=0`.
- **Cross-validation:** `GroupKFold(n_splits=5)`, `groups=patient_id` (pre-computed list).
- **Hub key:** `07_residual_patient` (shared with `improve_rmse_skore.py`).
- **Submission:** `submissions/07_residual_patient.csv`.

## Risks / things that could invalidate the result

- **Train+test feature leakage:** patient aggregates are computed over the union of train
  and test rows, so test-patient statistics bleed into the training feature set. This
  means the CV RMSE is slightly optimistic — the model sees test-patient profile
  information during training folds.
- **Baseline leakage in CV folds:** the `baseline` column uses `off` from the same
  visit being predicted. While `off` is genuinely a different measurement from `target`,
  it is also missing 42 % of the time. When `off` is present, the baseline is
  essentially a noisy version of the answer — so the residual is small and easy to fit,
  but this is not leakage per se (it is the same measurement type, not the label itself).
- **Hub key collision:** this script pushes to `07_residual_patient`, the same hub key
  as `improve_rmse_skore.py`. The last run overwrites the earlier one.

## Status

- **State:** done
- **Approved by user on:** 2025-07-11 (run by user outside Bob iteration loop)
- **Headline result:** RMSE 4.92 ± 0.09 (GroupKFold 5-fold, ResidualRegressor + levodopa decay + patient aggregates) — 9 % below 09_patient_features (5.38)
- **Implication for next iteration:** Residual formulation + levodopa decay features give a large gain over direct HGBR prediction. Richer patient severity profile added in `11_residual_patient`.
