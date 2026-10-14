# JOURNAL

## Status

- **Project / dataset:** Parkinson's disease multi-cohort synthetic visits — regression
- **Goal:** minimize RMSE for the true OFF MDS-UPDRS motor score, on patients held out from training
- **Last experiment:** 12_blend_ridge - done
- **Last result:** RMSE 4.25 ± 0.09 (GroupKFold 5-fold, BlendRegressor w=0.75: residual HGBR + Ridge) — **new best**
- **Latest submission:** `submissions/09_blend_ridge.csv` (11 013 rows) — hub key `09_blend_ridge`

- **Workspace decisions** (immutable unless the user pivots):
  - tabular library: pandas - recorded: 2025-01-01
  - env manager: pip+venv - recorded: 2025-01-01
  - agent feature: not installed
  - optional features: none
  - package name (`src/<pkg>/`): n/a - flat scripts
  - skore mode: hub - recorded: 2025-01-01
  - skore hub workspace: leo.rabiat-humbert - recorded: 2025-01-01
  - skore mlflow tracking uri: n/a
  - student prior: comfortable - recorded: 2025-01-01
  - CV splitter family: row holdout (splitter=0.2) for 01_dummy and 02_ridge; GroupKFold later

## Data understanding (EDA)

- **Status:** done - 2025-07-11
- **Summary:** 44 590 visits × 14 columns, 5 576 patients (~8 visits/patient). Target
  (true OFF MDS-UPDRS) is approximately Normal (mean 37.5, SD 16.5). Strongest
  predictors: `off` (r=0.87) and `on` (r=0.67), both heavily missing (42 % and 30 %) —
  explains HGBR's advantage over Ridge via NaN routing. No datetime columns; GroupKFold
  is the correct splitter. `rater_id` (50 raters, Cramér V 0.019) is a candidate
  feature not yet used. `time_since_intake_off` is 79 % missing — its absence is itself
  signal. `age_at_diagnosis` ↔ `age` collinear (r=0.94): tree models are immune, Ridge
  needs alpha tuning.
- **Report:** [data/eda.md](../data/eda.md)

## History

| Stem | Intent (one line) | Status | Headline result | Design note |
|---|---|---|---|---|
| `01_dummy` | DummyRegressor mean baseline, row holdout, push to hub as `01_dummy` | done | n/a - script exists, result on hub | n/a |
| `02_ridge` | Ridge + median imputer on numeric features, compare to dummy, push to hub as `02_ridge` | done | RMSE 10.49 (row holdout) vs Dummy 16.48 — 36 % lower | [design note](02_ridge.md) |
| `03_grouped_cv` | Same Ridge + Dummy with GroupKFold(5) on patient_id — honest patient holdout | done | Ridge 10.60 ± 0.27 vs Dummy 16.50 ± 0.34 (grouped CV) | [design note](03_grouped_cv.md) |
| `04_hgbr` | HistGradientBoostingRegressor — NaN as signal, no imputation, grouped CV | done | RMSE 7.75 ± 0.15 (grouped CV) — 27 % below Ridge | [design note](04_hgbr.md) |
| `05_tabular_pipeline` | skrub tabular_pipeline — all columns including gene/cohort, grouped CV | done | RMSE 7.53 ± 0.14 (grouped CV) — 3 % below HGBR | [design note](05_tabular_pipeline.md) |
| `06_dataops` | skrub DataOps graph — GroupKFold baked in, same model as 05, preferred architecture | done | RMSE 7.53 ± 0.15 (grouped CV, DataOps) — matches 05; **final submission** | [design note](06_dataops.md) |
| `07_ridge_cv` | RidgeCV 30-alpha log-spaced grid, 8 numeric features, GroupKFold(5) DataOps | done | RMSE 10.60 ± 0.27 — identical to 03_grouped_cv; alpha=1.0 was already optimal | [design note](07_ridge_cv.md) |
| `08_hgbr_tuned` | GridSearchCV HGBR hyperparameter tuning (max_iter/learning_rate/max_leaf_nodes), GroupKFold(5) DataOps | done | RMSE 7.51 ± 0.13 — marginal 0.02 improvement over 06_dataops | [design note](08_hgbr_tuned.md) |
| `09_patient_features` | Per-patient aggregates (visit count, ledd/on/off mean+std) merged onto visits, tuned HGBR, GroupKFold(5) DataOps | done | RMSE 5.38 ± 0.12 — **28 % improvement** over 08_hgbr_tuned; best CV result | [design note](09_patient_features.md) |
| `10_residual_patient` | ResidualRegressor (HGBR on y−baseline): clinical baseline + levodopa decay features + patient aggregates; GroupKFold(5) | done | RMSE 4.92 ± 0.09 — 9 % below 09_patient_features (5.38) | [design note](10_residual_patient.md) |
| `11_residual_patient` | Same residual architecture with richer patient severity profile (std/min/max off, disease duration, visit rank, on_dev_patient); GroupKFold(5) | done | RMSE 4.58 ± 0.11 — 7 % below 10_residual_patient | [design note](11_residual_patient.md) |
| `step6_seeds` *(diagnostic)* | Seed noise + early-stopping diagnostic: 2 feature variants × 3 seeds × 2 stopping modes; no hub push, no submission | done | See script output — purpose was to confirm feature variant signal exceeds seed variance | n/a |
| `12_blend_ridge` | BlendRegressor (w=0.75 HGBR + 0.25 Ridge) with temporal neighbourhood features (prev/next OFF, OFF slope); early-stopping HGBR; GroupKFold(5) | done | RMSE 4.25 ± 0.09 — **new best**; 7 % below 11_residual_patient | [design note](12_blend_ridge.md) |

## Backlog

| # | Item | Source |
|---|---|---|
| ~~B1~~ | ~~Evaluate with patient-grouped CV (GroupKFold on patient_id)~~ — resolved in 03_grouped_cv | user (Day 2 step 3) |
| ~~B2~~ | ~~HistGradientBoostingRegressor — keeps NaN as signal~~ — resolved in 04_hgbr | user (Day 2 step 4) |
| ~~B3~~ | ~~tabular_pipeline (skrub) — includes gene/cohort string columns~~ — resolved in 05_tabular_pipeline | user (Day 2 step 5) |
