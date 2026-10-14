# Improve Results Plan — Parkinson's True OFF Prediction

## Overview

Current best: **RMSE 7.53 ± 0.15** (GroupKFold 5-fold, `06_dataops`:
`TableVectorizer + HistGradientBoostingRegressor`, all columns, DataOps architecture).

Three independent improvement directions are planned, in priority order:

1. **EDA** — understand the data before modelling; surface feature distributions,
   missingness patterns, patient-level structure, and target skew.
2. **Ridge alpha search** — try `RidgeCV` (grid of alphas) with honest GroupKFold to
   find whether the default `alpha=1.0` was optimal; establish a proper linear baseline.
3. **Model improvements on top of `06_dataops`** — hyperparameter tuning of HGBR,
   patient-level aggregation features, and/or a stronger ensemble (LightGBM / XGBoost).

Each sub-task is self-contained and can be implemented independently.

---

## Sub-Task 1 — EDA: data understanding before modelling

### Intent

No EDA has been done. Before proposing new models or features, we need to understand
the raw data: column types, missingness patterns, target distribution, patient-level
structure, cardinality of string columns, and feature correlations with `target`.
This gives evidence for every modelling decision that follows.

### Expected Outcomes

- `data/eda.py` (jupytext `# %%` script) executed successfully.
- `data/eda.md` written with findings: shape, dtypes, missingness, cardinality,
  target statistics, notable correlations.
- `data/eda_visits.html` skrub `TableReport` linked from `eda.md`.
- `journal/JOURNAL.md` `## Data understanding (EDA)` section updated.
- Key findings that influence experiment backlog are surfaced explicitly.

### Todo List

- [ ] Create `data/eda.py` as a jupytext `# %%` script:
  - Load `X_train.csv`, `y_train.csv`; merge into `visits`.
  - Print shape, dtypes, missing-value counts per column.
  - Print `visits["target"].describe()` + skewness.
  - Print per-patient visit count distribution (`visits.groupby("patient_id").size()`).
  - Print cardinality of `gene` and `cohort`.
  - Print correlation of numeric features with `target`.
  - Generate `skrub.TableReport(visits)` → save as `data/eda_visits.html`.
- [ ] Execute `data/eda.py` via the in-process runner; capture the digest.
- [ ] Write `data/eda.md` with structured findings from the digest.
- [ ] Update `journal/JOURNAL.md` `## Data understanding (EDA)` section.

### Relevant Context

- Data columns: `Index`, `patient_id`, `cohort` (low-cardinality), `sexM`, `gene`
  (high-cardinality), `age_at_diagnosis`, `age`, `ledd`, `time_since_intake_on`,
  `time_since_intake_off`, `rater_id`, `on`, `off`; target: `target`.
- Known missingness: `on`, `off`, `ledd`, `time_since_intake_off` (from row preview).
- `rater_id` has never been used as a feature — EDA should check its cardinality and
  whether it correlates with `target`.
- EDA runner: `audit-ml-pipeline/scripts/run_cells.py`.
- `skrub.TableReport` for HTML overview.

### Status

[ ] pending

---

## Sub-Task 2 — Ridge alpha search with honest grouped CV

### Intent

`02_ridge` used `alpha=1.0` with a row holdout (visit-level leakage). We never searched
over alpha values, and the CV was dishonest. This sub-task runs `RidgeCV` with a broad
log-spaced alpha grid under honest `GroupKFold(5)` to find the best linear regularisation
level, and records whether any alpha materially beats the `alpha=1.0` HGBR comparison.
This also closes the question of whether linear models have been given a fair chance.

### Expected Outcomes

- `experiments/07_ridge_cv.py` runs to completion.
- `journal/07_ridge_cv.md` design note filed and status = done.
- Hub key `07_ridge_cv` pushed to the project.
- `submissions/07_ridge_cv.csv` written.
- RMSE reported per alpha (via `RidgeCV`) and the best alpha identified.
- JOURNAL.md History row added for `07_ridge_cv`.

### Todo List

- [ ] Write `journal/07_ridge_cv.md` design note (state: approved by user before coding).
- [ ] Write `experiments/07_ridge_cv.py` as a DataOps graph (same architecture as `06_dataops`):
  - Load `X_train`, `y_train`, `X_test`; build `visits`.
  - Use same 8 numeric `feature_cols` as `02_ridge` (for direct comparison).
  - Build DataOps graph: `skrub.var` → drop non-features → `mark_as_X` with
    `GroupKFold(n_splits=5)` and `groups=patient_id`.
  - Use `make_pipeline(SimpleImputer("median"), RidgeCV(alphas=np.logspace(-3, 4, 30)))`
    as the estimator — `RidgeCV` selects alpha internally via LOO/GCV.
  - `evaluate(pred, data={"visits": visits})` → `project.put("07_ridge_cv", report)`.
  - Print best alpha chosen by `RidgeCV` after fitting.
  - Submission CSV: `SkrubLearner` fit + predict on `X_test` → `submissions/07_ridge_cv.csv`.
- [ ] Run the script; record best alpha and RMSE.
- [ ] Update `journal/07_ridge_cv.md` Status block.
- [ ] Add History row to `journal/JOURNAL.md`.

### Relevant Context

- `feature_cols` from `experiments/02_ridge.py`: `sexM`, `age_at_diagnosis`, `age`,
  `ledd`, `time_since_intake_on`, `time_since_intake_off`, `on`, `off`.
- `RidgeCV` from `sklearn.linear_model` — performs efficient LOO CV internally to
  select alpha; best alpha accessible via `.best_estimator_[-1].alpha_` after fit.
- DataOps graph pattern: `experiments/06_dataops.py`.
- Hub login pattern: copy from `experiments/06_dataops.py`.

### Status

[ ] pending

---

## Sub-Task 3 — HGBR hyperparameter tuning on the DataOps graph

### Intent

`06_dataops` used default HGBR hyperparameters (`random_state=0`, everything else
default). HGBR exposes `max_iter`, `learning_rate`, `max_leaf_nodes`, and
`min_samples_leaf` — all of which affect the bias-variance trade-off. A coarse grid
search over these, still inside the honest GroupKFold, may yield a meaningful RMSE drop
without any architectural changes.

### Expected Outcomes

- `experiments/08_hgbr_tuned.py` runs to completion.
- `journal/08_hgbr_tuned.md` design note filed and status = done.
- Hub key `08_hgbr_tuned` pushed.
- `submissions/08_hgbr_tuned.csv` written.
- Best hyperparameter set and resulting RMSE recorded.
- JOURNAL.md History row added.

### Todo List

- [ ] Write `journal/08_hgbr_tuned.md` design note (state: approved by user before coding).
- [ ] Write `experiments/08_hgbr_tuned.py` as a DataOps graph extending `06_dataops`:
  - Same DataOps graph structure as `06_dataops` (all columns, GroupKFold 5-fold).
  - Wrap `TableVectorizer + HistGradientBoostingRegressor` in a `Pipeline`.
  - Wrap the pipeline in `HalvingGridSearchCV` (or `GridSearchCV`) with a param grid:
    - `hgbr__max_iter`: [200, 500]
    - `hgbr__learning_rate`: [0.05, 0.1]
    - `hgbr__max_leaf_nodes`: [15, 31, 63]
  - Use `cv=GroupKFold(n_splits=3)` inside the search (inner CV), outer eval still
    GroupKFold(5) for the `evaluate` call.
  - `evaluate(pred, data={"visits": visits})` → `project.put("08_hgbr_tuned", report)`.
  - Print best params after run.
  - Submission CSV → `submissions/08_hgbr_tuned.csv`.
- [ ] Run the script; record best params and outer CV RMSE.
- [ ] Update `journal/08_hgbr_tuned.md` Status block.
- [ ] Add History row to `journal/JOURNAL.md`.

### Relevant Context

- `06_dataops.py` is the starting point — copy the graph, add the search wrapper.
- `HalvingGridSearchCV` from `sklearn.model_selection` is faster than full grid for
  HGBR; falls back to `GridSearchCV` if halving causes instability with grouped CV.
- Inner CV must also be `GroupKFold` to avoid patient leakage inside the search.
  Pass `groups` via `fit_params` in the inner CV: `fit_params={"groups": groups_array}`.
- Keep the outer eval honest: `evaluate` uses the baked-in GroupKFold(5) from the graph.

### Status

[ ] pending

---

## Sub-Task 4 — Patient-level aggregation features

### Intent

The current feature set is purely visit-level. But many clinical signals are meaningful
relative to a patient's own baseline: e.g. how much the `on` score has changed since
the patient's first visit, or the standard deviation of `off` scores across past visits.
These patient-level features could help HGBR capture within-patient trajectories.

**Important caveat**: patient aggregations must be computed using only past visits
(or all visits if used as a static summary — leakage-safe only if the aggregation does
not look at the target column). We use target-free aggregations only.

### Expected Outcomes

- `experiments/09_patient_features.py` runs to completion.
- `journal/09_patient_features.md` design note filed and status = done.
- Hub key `09_patient_features` pushed.
- `submissions/09_patient_features.csv` written.
- RMSE compared to `06_dataops` baseline.
- JOURNAL.md History row added.

### Todo List

- [ ] Write `journal/09_patient_features.md` design note (state: approved by user before coding).
- [ ] Write `experiments/09_patient_features.py`:
  - Load `visits` as before.
  - Compute per-patient static aggregates (no target column involved):
    - `age_at_diagnosis` (constant per patient — already present).
    - Mean and std of `ledd` per patient → `ledd_mean_patient`, `ledd_std_patient`.
    - Visit count per patient → `visit_count_patient`.
    - Mean `on` per patient → `on_mean_patient`.
    - Mean `off` per patient → `off_mean_patient`.
  - Merge aggregates back onto `visits` by `patient_id`.
  - Build DataOps graph (same as `06_dataops`) with the enriched `visits` table.
  - `evaluate(pred, data={"visits": visits_enriched})` → `project.put("09_patient_features", report)`.
  - Submission: apply same aggregations to `X_test` (merge test rows with train-derived
    patient stats) → `submissions/09_patient_features.csv`.
- [ ] Run the script; record RMSE vs `06_dataops`.
- [ ] Update `journal/09_patient_features.md` Status block.
- [ ] Add History row to `journal/JOURNAL.md`.

### Relevant Context

- `visits` already has `patient_id` — groupby is straightforward.
- For test-time aggregations: patient stats are computed from training data only
  (not from `X_test`), then merged onto `X_test` by `patient_id`. Patients in the test
  set but not in training get NaN aggregates — HGBR handles this natively.
- Do NOT use `target` in any aggregation (that would be target leakage).
- `rater_id` is a potential confounder — EDA (Sub-Task 1) will clarify whether it
  should be used or dropped.

### Status

[ ] pending

---

## Ordering

```
Sub-Task 1 (EDA)  →  Sub-Task 2 (RidgeCV)  →  Sub-Task 3 (HGBR tuning)  →  Sub-Task 4 (patient features)
```

Sub-Task 1 should run first because EDA findings may sharpen the feature engineering
decisions in Sub-Tasks 3 and 4. Sub-Tasks 2, 3, and 4 are otherwise independent and
could be run in any order if EDA is already done.
