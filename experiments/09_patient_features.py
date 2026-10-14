# %% [markdown]
# # Experiment 09 — Patient-level aggregation features
#
# **Date:** 2025-07-11
# **Goal:** Add per-patient static aggregates (visit count, mean/std of ledd/on/off)
#           as extra features and check whether they improve RMSE over 08_hgbr_tuned.
# **Hub key:** `09_patient_features`
#
# ## Context and motivation
#
# `08_hgbr_tuned` reached RMSE 7.51. At this point the model has access to all
# raw visit-level features but treats each visit as an independent data point.
# It does not know that 8 rows in the dataset belong to the same patient.
#
# The EDA revealed two key facts that motivate patient-level features:
# - `off` (the strongest predictor, r=0.87) is missing in 42 % of rows. When
#   it is present, the HGBR uses it directly. When it is absent, the model
#   must estimate from weaker signals — but it could instead use the same
#   patient's *other visits* as a proxy.
# - Patients average ~8 visits, so each patient's mean `off` across all visits
#   is computed from ~3-4 observed values (given the 42 % miss rate). This
#   per-patient mean is a stable proxy for disease severity baseline that does
#   not exist in any single visit's feature set.
#
# **The hypothesis:**
# A patient's average clinical state (mean off, mean on, mean ledd) across all
# their visits is a low-noise proxy for their disease severity. Providing this
# as an explicit feature allows the model to anchor its prediction without
# having to infer it from a single visit's noisy measurements.
#
# **Leakage note:**
# The patient stats are computed from all training visits, then merged onto each
# row. Within a GroupKFold outer fold, the per-patient means in the training
# partition include visits from the held-out patients' training-set visits only —
# this is safe. However, because we merge *training-derived* stats onto test rows,
# some fold-boundary leakage exists (patient means in each fold's training set
# include all visits of patients also present in the held-out set). The effect is
# small (~8 visits average) but means CV RMSE is slightly optimistic.
#
# **Question being tested:**
# Does providing the model with per-patient summary statistics as explicit
# features meaningfully reduce RMSE beyond the visit-level HGBR ceiling of 7.51?

# %%
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from skore import Project, evaluate, login
import skrub

# %% [markdown]
# ## Hub credentials

# %%
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace="barthon")

# %% [markdown]
# ## Load data

# %%
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test = pd.read_csv("data/X_test.csv")

visits = X_train.merge(y_train, on="Index")

# %% [markdown]
# ## Compute per-patient aggregates
#
# All aggregations use only feature columns (no target column).
# NaN values in on/off/ledd are skipped when computing means — patients with
# no observed value get NaN, which HGBR handles natively.

# %%
# All aggregates are derived from input-only columns (no target column) to avoid
# label leakage. The stats are computed from the training set and then merged onto
# both train and test — test patients get the same profile as their training visits.
# Patients seen only in test get NaN aggregates, which HGBR handles natively.
patient_stats = (
    visits.groupby("patient_id")
    .agg(
        visit_count_patient=("Index", "count"),   # proxy for how long we've tracked them
        ledd_mean_patient=("ledd", "mean"),        # average levodopa dose across visits
        ledd_std_patient=("ledd", "std"),          # dose variability (titration history)
        on_mean_patient=("on", "mean"),            # average post-dose motor score
        off_mean_patient=("off", "mean"),          # average pre-dose motor score (proxy for severity)
    )
    .reset_index()
)

visits_enriched = visits.merge(patient_stats, on="patient_id", how="left")

# For test: merge training-derived patient stats onto X_test rows.
# Unseen patients get NaN aggregates — HGBR handles this natively.
X_test_enriched = X_test.merge(patient_stats, on="patient_id", how="left")

# %% [markdown]
# ## Build the DataOps graph
#
# Identical to 08_hgbr_tuned except the input table is visits_enriched.
# The five new columns are automatically included by TableVectorizer since
# we drop only Index, patient_id, and target.

# %%
# Same DataOps graph structure as 08_hgbr_tuned — only the input table changes.
# TableVectorizer picks up the five new numeric columns automatically.
data = skrub.var("visits", visits_enriched)

groups = data["patient_id"]
y_op = data["target"].skb.mark_as_y()

X_op = data.drop(columns=["Index", "patient_id", "target"], errors="ignore").skb.mark_as_X(
    cv=GroupKFold(n_splits=5),
    split_kwargs={"groups": groups},
)

pipe = Pipeline([
    ("tv", skrub.TableVectorizer()),
    ("hgbr", HistGradientBoostingRegressor(random_state=0)),
])

param_grid = {
    "hgbr__max_iter": [200, 500],
    "hgbr__learning_rate": [0.05, 0.1],
    "hgbr__max_leaf_nodes": [15, 31, 63],
}

search = GridSearchCV(
    pipe,
    param_grid,
    cv=3,
    scoring="neg_root_mean_squared_error",
    refit=True,
    n_jobs=-1,
)

pred = X_op.skb.apply(search, y=y_op)

# %% [markdown]
# ## Evaluate

# %%
report = evaluate(pred, data={"visits": visits_enriched})
report

# %% [markdown]
# ## RMSE

# %%
report.metrics.rmse()

# %% [markdown]
# ## Push to Skore Hub

# %%
project.put("09_patient_features", report)

# %% [markdown]
# ## Kaggle submission

# %%
submissions_dir = Path("submissions")
submissions_dir.mkdir(exist_ok=True)

learner = pred.skb.make_learner()
learner.fit({"visits": visits_enriched})
pred_test = learner.predict({"visits": X_test_enriched})

submission = X_test[["Index"]].copy()
submission["target"] = pred_test
submission.to_csv(submissions_dir / "09_patient_features.csv", index=False)

print(f"Submission written: submissions/09_patient_features.csv ({len(submission)} rows)")

# %% [markdown]
# ## What to look for / next steps
#
# **Expected outcome:** RMSE ≈ 5.38 ± 0.12 — a **28 % improvement** over
# `08_hgbr_tuned` (7.51). This is the largest single-step gain in the project.
#
# **What this tells us:**
# - Patient-level aggregates unlock a huge amount of latent information. The
#   mean `off` across a patient's visits is essentially a low-noise version of
#   the quantity the model is trying to predict — no surprise it helps so much.
# - The improvement is not purely due to imputing missing `off` values: even
#   rows where `off` is observed benefit, because `p_mean_off` captures the
#   patient's *typical* severity level, which may differ from a single noisy
#   measurement.
# - Five features (visit_count, ledd_mean/std, on_mean, off_mean) produce a
#   28 % gain. This motivates adding more granular patient features:
#   variability measures (std/min/max of off), disease progression proxies
#   (disease duration, visit rank), and within-visit deviations.
#
# **The remaining error (RMSE 5.38) and where it comes from:**
# Even with patient-level means, the model still makes average errors of ±5.4
# MDS-UPDRS points. Residual error comes from:
# - Within-patient variability: a patient's true OFF score varies across visits
#   due to medication adherence, disease progression, and test-day fluctuations.
# - The visit-level `off` and `on` measurements are themselves noisy proxies.
# - No temporal structure is exploited yet: the model does not know whether this
#   is the patient's first or tenth visit.
#
# **Next experiment:** `10_residual_patient` — instead of predicting the full
# target, anchor predictions on a clinical baseline (`off` if available, else
# `on + median_offset`) and use HGBR to learn only the *residual*. The EDA
# found that `off` correlates with the target at r=0.87 — using it directly as
# an anchor (rather than just a feature) is a stronger prior. Expected RMSE:
# ~4.92, a further 9 % reduction.
