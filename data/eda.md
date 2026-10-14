# EDA: Parkinson's Disease Multi-Cohort Visits

Exploratory data analysis of the training dataset (`X_train.csv` merged with
`y_train.csv`). Run before modelling; raw data is read-only.

Rich interactive report: [data/eda_visits.html](eda_visits.html)

---

## Dataset at a glance

| | |
|---|---|
| **Rows** | 44 590 visits |
| **Columns** | 14 (13 features + target) |
| **Patients** | 5 576 unique `patient_id` values |
| **Target** | `target` — true OFF MDS-UPDRS motor score (continuous, regression) |
| **Test set** | 11 013 rows (`X_test.csv`) |

Average visits per patient: **~8** (44 590 / 5 576).

---

## Per-column findings

| Column | dtype | Missing % | Unique | Notes |
|---|---|---|---|---|
| `Index` | Int64 | 0 % | 44 590 | Row identifier — must be dropped from features |
| `patient_id` | String | 0 % | 5 576 | Patient identifier — must be in groups for GroupKFold, not in features |
| `cohort` | String | 0 % | 2 | Low cardinality (A / B) — safe to one-hot encode |
| `sexM` | Int64 | 0 % | 2 | Binary (0/1) |
| `gene` | String | **32 %** | 4 | High cardinality relative to 4 values but 32 % missing — `StringEncoder` handles both |
| `age_at_diagnosis` | Float64 | 5 % | 563 | Nearly constant within a patient (slow drift) |
| `age` | Float64 | 0 % | 1 082 | `r=0.94` with `age_at_diagnosis` — collinear |
| `ledd` | Float64 | **37 %** | 1 320 | Levodopa equivalent daily dose — missing when medication data unavailable; Pearson r=0.30 with target |
| `time_since_intake_on` | Float64 | **46 %** | 64 | Time (hours) from last ON-state dose — very sparse |
| `time_since_intake_off` | Float64 | **79 %** | 178 | Time from last OFF-state dose — almost entirely missing (79 %) |
| `rater_id` | String | 0 % | 50 | 50 unique raters — medium cardinality; weak Cramér V (0.019) with target; may capture rater bias |
| `on` | Float64 | **30 %** | 83 | MDS-UPDRS ON-state motor score; Pearson r=0.67 with target |
| `off` | Float64 | **42 %** | 101 | MDS-UPDRS OFF-state motor score; Pearson r=0.87 with target — strongest predictor |
| `target` | Float64 | 0 % | 867 | True OFF motor score (label) — mean 37.5, SD 16.5 |

---

## Target distribution

- **Mean**: 37.5 — **Median**: 37.3 — **SD**: 16.5
- **Range**: 0 to 109.5 — **IQR**: 23.7 (Q1=25.6, Q3=49.3)
- **Shape**: approximately bell-shaped (histogram bin counts peak at the 33–44 range);
  slight right tail (a few patients above 80).
- **No stratification needed**: RMSE is the metric; the distribution is near-Normal.
  `GroupKFold` (no stratification) is appropriate.

---

## Structure signals

- **No datetime columns** detected. There is no explicit visit timestamp — temporal
  ordering within a patient is implicit in the data ordering / `age` field only.
- **`patient_id`**: unique ratio 0.125 — clearly a group identifier. Multiple visits
  per patient; GroupKFold is the right splitter (already used in experiments 03–06).
- **`Index`**: unique ratio 1.0 — pure row identifier, must be dropped.
- No other group-like column with high unique ratio found beyond `patient_id` and `Index`.

---

## Associations

**Strongest feature↔target links (Pearson):**

| Feature | Pearson r | Cramér V |
|---|---|---|
| `off` | **0.871** | 0.406 | ← strongest predictor; also 42 % missing |
| `on` | 0.669 | 0.368 | ← second; 30 % missing |
| `ledd` | 0.298 | 0.236 | moderate |
| `age` | 0.310 | 0.114 | moderate |
| `age_at_diagnosis` | 0.133 | 0.051 | weak |
| `time_since_intake_on` | ~0 | 0.177 | non-linear signal only |
| `time_since_intake_off` | ~0 | 0.092 | very weak |
| `cohort` | n/a | 0.090 | small categorical signal |
| `gene` | n/a | 0.052 | small categorical signal |
| `rater_id` | n/a | 0.019 | negligible |
| `sexM` | ~0 | 0.023 | negligible |

**Feature↔feature collinearity:**

- `age_at_diagnosis` ↔ `age`: Pearson r = 0.94 — strongly collinear (age grows
  linearly from diagnosis age). Tree models (HGBR) are immune; Ridge should be aware.
- `on` ↔ `off`: Pearson r = 0.87 — correlated (disease severity shows in both states).
- `on` ↔ `target`: r = 0.67; `off` ↔ `target`: r = 0.87. Both are predictors.

**Potential leakage:** `off` (Pearson 0.87 with target) is a very strong predictor.
However, `off` is the measured OFF-state score at visit, while `target` is the *true*
OFF-state score — a different measurement on the same visit. This is genuine signal,
not leakage: the measured score is noisy, the true score is latent. **No leakage
concern.**

---

## Modelling implications

1. **`off` and `on` are by far the strongest features** — any model that ignores them
   leaves the most signal on the table. They are also the most missing (42 % and 30 %),
   so NaN-aware models (HGBR) outperform imputation strategies. This explains the 27 %
   RMSE gap between Ridge (median imputation) and HGBR (experiment 04).

2. **`rater_id` (50 unique raters)** has negligible Cramér V (0.019). It has never been
   used as a feature. Adding it via `TableVectorizer` could capture systematic rater
   bias. Worth including in a future experiment — but its signal is weak.

3. **`time_since_intake_off` is 79 % missing.** Most of the signal from this column is
   the *fact that it is missing* (the visit was not a controlled OFF state), not the
   numeric value. HGBR handles this well via NaN routing.

4. **`age_at_diagnosis` ↔ `age` collinearity (r=0.94)** is irrelevant for tree models
   but may hurt Ridge without careful regularisation. This reinforces the importance of
   the alpha search (Sub-Task 2).

5. **No temporal order** column found — `TimeSeriesSplit` is not warranted. `GroupKFold`
   remains the correct splitter.

6. **Patient-level aggregation potential:** with ~8 visits per patient on average, and
   `off` / `on` both varying visit to visit, per-patient means and variances could add
   useful baseline signal (Sub-Task 4).

7. **`gene` missingness (32 %):** these NaNs should stay as NaN (not imputed as a
   separate category), as the missing pattern may correlate with cohort or disease
   subtype.

---

## Open questions

- Does the ordering of visits within a patient follow chronological order in the CSV, or
  is it arbitrary? If chronological, disease progression features (visit number, time
  delta) could be derived.
- Are visits from `cohort=A` and `cohort=B` from different clinical centres, or
  different study arms? The Cramér V signal (0.09) is small but present.
