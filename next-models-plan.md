# Next Models Plan — Parkinson's True OFF Prediction

## Overview

Two experiments are done: `01_dummy` (RMSE 16.48) and `02_ridge` (RMSE 10.49, row holdout).
The next three experiments follow the Day 2 guide in order. Each is a self-contained
`experiments/NN_*.py` script in the same style as `02_ridge.py`. Each run pushes its report
to Skore Hub under a stable key and writes a Kaggle CSV to `submissions/`.

The workspace is `barthon` on `https://api.skore.probabl.ai`, project `bobathon-esilv`.

Hub login pattern (copy from `02_ridge.py`):
```python
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace="barthon")
```

**Key constraint**: `project.put()` only accepts `EstimatorReport` or
`CrossValidationReport` — not `ComparisonReport`. Evaluate each model individually
then push. Use `skore.compare(...)` locally for side-by-side display only.

---

## Sub-Task 1 — Grouped CV on Ridge + Dummy (Day 2 §3)

### Intent
Replace the random row holdout with patient-grouped 5-fold CV. This is the honest
evaluation that mirrors the Kaggle split (whole patients held out, never split across
folds). The guide explicitly says `splitter=cv_splits` where `cv_splits` is
`list(GroupKFold(n_splits=5).split(X, y, groups=groups))` with
`groups = visits["patient_id"]`.

**Why now**: `02_ridge` row-holdout RMSE (10.49) is optimistic because the same patient's
visits appear on both sides. GroupKFold gives the number we should trust from here on.
All subsequent experiments use `cv_splits` — defining it here locks the comparison baseline.

### Expected Outcomes
- `experiments/03_grouped_cv.py` runs to completion.
- `journal/03_grouped_cv.md` design note filed and status = done.
- Hub keys `03_dummy_cv` and `03_ridge_cv` appear in the project.
- `submissions/03_grouped_cv.csv` written (Ridge refit on all training rows).
- JOURNAL.md History row added.

### Todo List
- [ ] Write `journal/03_grouped_cv.md` (design note, state: approved).
- [ ] Write `experiments/03_grouped_cv.py`:
  - Load `X_train`, `y_train`, `X_test`; build `visits`, `y`, `X` with same `feature_cols` as `02_ridge`.
  - Build `groups = visits["patient_id"]`.
  - Precompute `cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))`.
  - `DummyRegressor(strategy="mean")` → `evaluate(dummy, X, y, splitter=cv_splits)` → `project.put("03_dummy_cv", report_dummy)`.
  - `make_pipeline(SimpleImputer("median"), Ridge(alpha=1.0))` → `evaluate(ridge, X, y, splitter=cv_splits)` → `project.put("03_ridge_cv", report_ridge)`.
  - Local `compare(...)` for display.
  - Submission CSV: Ridge refit on all training rows → `submissions/03_grouped_cv.csv`.
- [ ] Run the script; capture RMSE for both models.
- [ ] Update `journal/03_grouped_cv.md` Status block with headline result.
- [ ] Add History row to `journal/JOURNAL.md`.

### Relevant Context
- Day 2 guide §3: `docs/DAY2.md` lines 101–132.
- `feature_cols` and hub login: `experiments/02_ridge.py`.
- `project.put` constraint: accepts only `EstimatorReport` / `CrossValidationReport`.
- Hub workspace: `barthon`, project name: `bobathon-esilv`.

### Status
[x] done — Ridge 10.60 ± 0.27 / Dummy 16.50 ± 0.34 (GroupKFold 5-fold)

---

## Sub-Task 2 — HistGradientBoostingRegressor: missingness as signal (Day 2 §4)

### Intent
Add `HistGradientBoostingRegressor(random_state=0)` — no imputation, NaNs treated as
signal. Evaluated with the same `cv_splits` from Sub-Task 1 so scores are directly
comparable to Ridge. The guide says "do NOT median-fill NaNs" for HGBR.

**Why now**: Ridge had to fill NaN with the median, throwing away the information that
"off was not measured" (often means the visit was ON-only). HGBR natively routes NaN
to a learned branch. This is the first test of non-linear + missingness-aware modelling.

### Expected Outcomes
- `experiments/04_hgbr.py` runs to completion.
- `journal/04_hgbr.md` design note filed and status = done.
- Hub key `04_hgbr` appears in the project.
- `submissions/04_hgbr.csv` written.
- JOURNAL.md History row added.

### Todo List
- [ ] Write `journal/04_hgbr.md` (design note, state: approved).
- [ ] Write `experiments/04_hgbr.py`:
  - Load data identically to `03_grouped_cv.py`; recompute `cv_splits`.
  - `HistGradientBoostingRegressor(random_state=0)` — **no** `SimpleImputer`.
  - `evaluate(hgbr, X, y, splitter=cv_splits)` → `project.put("04_hgbr", report)`.
  - Local `compare({"ridge_cv": report_ridge, "hgbr": report_hgbr})` for display.
  - Submission CSV: HGBR refit on all training rows → `submissions/04_hgbr.csv`.
- [ ] Run the script; capture RMSE.
- [ ] Update `journal/04_hgbr.md` Status block with headline result.
- [ ] Add History row to `journal/JOURNAL.md`.

### Relevant Context
- Day 2 guide §4: `docs/DAY2.md` lines 143–172.
- `cv_splits` pattern: Sub-Task 1 / `experiments/03_grouped_cv.py`.
- HGBR handles NaN natively — no imputer in the pipeline.
- String columns (`gene`, `cohort`) are NOT added here; that is Sub-Task 3.

### Status
[x] done — HGBR RMSE 7.75 ± 0.15 (GroupKFold 5-fold)

---

## Sub-Task 3 — skrub `tabular_pipeline`: mixed types including strings (Day 2 §5)

### Intent
Use `skrub.tabular_pipeline("regressor")` which bundles `TableVectorizer` +
`HistGradientBoostingRegressor`. This includes string columns (`gene`, `cohort`) that
Ridge and HGBR could not use. Drop only `Index`, `patient_id`, and `target` from `X`;
pass everything else.

**Why now**: `gene` and `cohort` are strings — Ridge can't read them; HGBR Sub-Task 2
also skipped them. `TableVectorizer` chooses an encoder per column automatically
(one-hot for low cardinality, `StringEncoder` for high cardinality). This is the first
model to use the full feature set.

**Important**: `patient_id` must be dropped from `X` or the model memorises patient
identity (leakage). `Index` is an identifier. Both must be excluded.

### Expected Outcomes
- `experiments/05_tabular_pipeline.py` runs to completion.
- `journal/05_tabular_pipeline.md` design note filed and status = done.
- Hub key `05_tabular_pipeline` appears in the project.
- `submissions/05_tabular_pipeline.csv` written.
- JOURNAL.md History row added.

### Todo List
- [ ] Write `journal/05_tabular_pipeline.md` (design note, state: approved).
- [ ] Write `experiments/05_tabular_pipeline.py`:
  - Load `X_train`, `y_train`, `X_test`.
  - Build `visits`; define `y = visits["target"]`.
  - `X_full = visits.drop(columns=["Index", "patient_id", "target"])` — all remaining columns.
  - Recompute `cv_splits` with `groups = visits["patient_id"]`.
  - `model = tabular_pipeline("regressor")` (from `skrub`).
  - `evaluate(model, X_full, y, splitter=cv_splits)` → `project.put("05_tabular_pipeline", report)`.
  - Submission CSV: model refit on `X_full` → predict on `X_test.drop(columns=["Index", "patient_id"], errors="ignore")` → `submissions/05_tabular_pipeline.csv`.
- [ ] Run the script; capture RMSE.
- [ ] Update `journal/05_tabular_pipeline.md` Status block with headline result.
- [ ] Add History row to `journal/JOURNAL.md`.

### Relevant Context
- Day 2 guide §5: `docs/DAY2.md` lines 177–209.
- `from skrub import tabular_pipeline` — already in `.venv`.
- Drop `Index` and `patient_id` from both train and test features.
- `X_test` for submission: same drops as training (`errors="ignore"` if `target` absent).

### Status
[x] done — RMSE 7.53 ± 0.14 (GroupKFold 5-fold)

---

## Ordering

```
03_grouped_cv  →  04_hgbr  →  05_tabular_pipeline
```

Each depends on `cv_splits` (patient-grouped 5-fold). Sub-Task 1 defines it first.
Sub-Tasks 2 and 3 recompute it from the same source so they remain self-contained scripts.
