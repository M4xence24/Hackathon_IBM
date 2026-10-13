# JOURNAL

## Status

- **Project / dataset:** Parkinson's disease multi-cohort synthetic visits — regression
- **Goal:** minimize RMSE for the true OFF MDS-UPDRS motor score, on patients held out from training
- **Last experiment:** 06_dataops - done
- **Last result:** RMSE 7.53 ± 0.15 (GroupKFold 5-fold, DataOps graph) — Day 2 complete
- **Final submission:** `submissions/06_dataops.csv` (11 013 rows) — hub URL: https://skore.probabl.ai/barthon/bobathon-esilv/cross-validations/67956

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

- **Status:** skipped - 2025-01-01
- **Summary:** n/a
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

## Backlog

| # | Item | Source |
|---|---|---|
| ~~B1~~ | ~~Evaluate with patient-grouped CV (GroupKFold on patient_id)~~ — resolved in 03_grouped_cv | user (Day 2 step 3) |
| ~~B2~~ | ~~HistGradientBoostingRegressor — keeps NaN as signal~~ — resolved in 04_hgbr | user (Day 2 step 4) |
| ~~B3~~ | ~~tabular_pipeline (skrub) — includes gene/cohort string columns~~ — resolved in 05_tabular_pipeline | user (Day 2 step 5) |
