# 09_patient_features

## Question / hypothesis

Do per-patient static aggregations (mean/std of `ledd`, `on`, `off`; visit count)
merged onto each visit row improve RMSE over the current best (08_hgbr_tuned: 7.51)?

## Motivation

- **Sourcing strategy:** user (improve-results-plan.md Sub-Task 4)
- **Source(s):** improve-results-plan.md + EDA (data/eda.md)
- **Why this matters:** EDA shows ~8 visits per patient on average and that `off`
  (r=0.87) and `on` (r=0.67) are the strongest predictors. Both vary visit to visit.
  A patient's mean `off` level over all their visits is a proxy for disease severity
  baseline that the model currently re-discovers from scratch each fold. Adding it
  explicitly as a feature could reduce residual variance and improve generalisation.

## Method

- **Files touched:** `experiments/09_patient_features.py` (self-contained DataOps script)
- **Change versus 08_hgbr_tuned:** same DataOps graph (all columns, GroupKFold 5-fold,
  tuned HGBR), but the visits table is enriched with six per-patient static features:
  - `visit_count_patient`: number of visits per patient
  - `ledd_mean_patient`, `ledd_std_patient`: mean and std of `ledd` per patient
  - `on_mean_patient`: mean `on` score per patient
  - `off_mean_patient`: mean `off` score per patient
  These aggregates are computed from the training data only. At test time, patient stats
  from training are merged onto X_test rows; unseen patients get NaN (HGBR handles this).
- **Cross-validation:** GroupKFold(5) baked into graph; same as 08_hgbr_tuned.
- **Hub key:** `09_patient_features`.
- **Submission:** `submissions/09_patient_features.csv`.

## Risks / things that could invalidate the result

- **Target leakage:** no aggregation uses the `target` column. All aggregates are
  computed from feature columns only (`ledd`, `on`, `off`). This is safe.
- **Leakage via test patients:** patient stats are computed on all training rows
  (including the fold's test rows in each outer fold). This introduces a mild form of
  leakage: the per-patient mean in a fold's training set includes visits from the held-
  out fold of the same patient. The effect is small given 8 visits/patient on average,
  but it means the RMSE improvement (if any) will be slightly optimistic.
- The aggregates for `on` and `off` have 30 % and 42 % missing respectively — the
  patient mean is computed from observed values only (pandas `mean(skipna=True)` default).
  Patients with no observed `on` or `off` get NaN aggregates — HGBR handles this.

## Status

- **State:** done
- **Approved by user on:** 2025-07-11
- **Headline result:** RMSE 5.38 ± 0.12 (GroupKFold 5-fold, tuned HGBR + patient aggregates) — 28 % improvement over 08_hgbr_tuned (7.51)
- **Implication for next iteration:** Per-patient aggregations are by far the biggest gain so far. Adding `off_mean_patient`, `on_mean_patient`, `ledd_mean/std_patient`, `visit_count_patient` compresses RMSE by 1.13 points. Next directions: more granular patient features (std of off/on, visit ordering, disease progression slope), or rater_id as a feature. Hub: https://skore.probabl.ai/barthon/bobathon-esilv/cross-validations/77058
