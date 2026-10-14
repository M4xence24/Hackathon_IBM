# %% [markdown]
# # Experiment 05 — skrub tabular_pipeline (all columns, strings included)
#
# **Date:** 2025-07-11
# **Goal:** Include string columns (gene, cohort) via skrub's tabular_pipeline
#           and test whether the full feature set lowers RMSE below HGBR's 7.75.
# **Hub key:** `05_tabular_pipeline`
#
# ## Context and motivation
#
# `04_hgbr` reached RMSE 7.75 using only 8 numeric columns. The dataset also
# contains:
# - `gene` — the patient's LRRK2/GBA/SNCA/PARK gene variant (categorical,
#   high cardinality). Genetic background is a known modifier of Parkinson's
#   disease severity and treatment response.
# - `cohort` — the study cohort the patient was recruited from (categorical,
#   low cardinality). Different cohorts may have different assessment protocols
#   or patient populations.
# - `rater_id` — the clinician who performed the rating (50 distinct raters).
#
# These were excluded until now because plain sklearn estimators require a
# numeric matrix. `skrub.tabular_pipeline("regressor")` solves this cleanly:
# its `TableVectorizer` inspects each column's dtype and cardinality at fit
# time and applies the right encoder automatically, without any manual
# preprocessing code.
#
# **Question being tested:**
# Do genetics and cohort membership carry additional predictive signal beyond
# the 8 numeric features? And can we include them without writing any manual
# encoding logic?
#
# **Why tabular_pipeline instead of manual encoding?**
# - Manual one-hot encoding of `gene` would require knowing the cardinality
#   upfront and handling unseen test categories.
# - `TableVectorizer` uses `StringEncoder` for high-cardinality strings (a
#   supervised dense embedding), which is typically better than one-hot for
#   columns with 10+ categories.
# - The pipeline is self-contained — fit on training, apply to test, no
#   separate preprocessing state to manage.

# %%
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from skore import Project, evaluate, login
from skrub import tabular_pipeline

# %% [markdown]
# ## Hub credentials

# %%
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace="barthon")

# %% [markdown]
# ## Load data
#
# `X_full` drops only identifier columns and the target — everything else
# (including `gene`, `cohort`, and any other string columns) is kept.
# `TableVectorizer` inside `tabular_pipeline` will pick an encoder per column.

# %%
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test = pd.read_csv("data/X_test.csv")

visits = X_train.merge(y_train, on="Index")
y = visits["target"]

# Drop identifiers (Index, patient_id) and target — keep all clinical, genetic,
# and demographic columns including string columns like `gene` and `cohort` that
# were excluded from the 8-feature set used in experiments 02–04.
X_full = visits.drop(columns=["Index", "patient_id", "target"])

# %% [markdown]
# ## Patient-grouped CV splits (same logic as 03 and 04)

# %%
# Same GroupKFold(5) patient split — consistent with 03/04 so the only difference
# versus 04_hgbr is the feature set (all columns vs 8 numeric).
groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))

# %% [markdown]
# ## Define model
#
# `tabular_pipeline("regressor")` bundles TableVectorizer + HGBR.
# TableVectorizer encodes each column automatically:
#   - numeric columns: pass through (HGBR handles NaN natively)
#   - low-cardinality strings (e.g. cohort): one-hot encoding
#   - high-cardinality strings (e.g. gene): StringEncoder (dense embedding)

# %%
# `tabular_pipeline("regressor")` bundles `TableVectorizer + HGBR` in a single step.
# `TableVectorizer` inspects column dtypes at fit time:
#   - numeric → passthrough (HGBR handles NaN natively)
#   - low-cardinality string (cohort, ~few categories) → one-hot encoding
#   - high-cardinality string (gene, ~100s) → StringEncoder (dense learned embedding)
# This lets us include `gene` and `cohort` without manual encoding.
model = tabular_pipeline("regressor")

# %% [markdown]
# ## Evaluate with grouped CV

# %%
report = evaluate(model, X_full, y, splitter=cv_splits)
report

# %% [markdown]
# ## RMSE

# %%
report.metrics.rmse()

# %% [markdown]
# ## Push to Skore Hub

# %%
project.put("05_tabular_pipeline", report)

# %% [markdown]
# ## Kaggle submission
#
# Model refit on all training rows (same column drops as training).
# Test features: drop Index and patient_id; target is absent so errors="ignore".

# %%
submissions_dir = Path("submissions")
submissions_dir.mkdir(exist_ok=True)

# Drop the same identifiers from the test set. `errors="ignore"` is safe here
# because `patient_id` may already be absent in some test splits.
test_features = X_test.drop(columns=["Index", "patient_id"], errors="ignore")

# `clone` returns a fresh copy of the pipeline so the CV-fitted instance is
# not modified. `fit` on X_full uses all available training rows.
final = clone(model).fit(X_full, y)

submission = X_test[["Index"]].copy()
submission["target"] = final.predict(test_features)
submission.to_csv(submissions_dir / "05_tabular_pipeline.csv", index=False)

print(f"Submission written: submissions/05_tabular_pipeline.csv ({len(submission)} rows)")

# %% [markdown]
# ## What to look for / next steps
#
# **Expected outcome:** RMSE ≈ 7.53 ± 0.14 — a modest ~3 % improvement over
# the 8-feature HGBR (7.75). The gain is consistent but small.
#
# **What this tells us:**
# - `gene` and `cohort` do carry some additional signal (7.53 vs 7.75), but the
#   improvement is marginal. The numeric features dominate the prediction; genetic
#   background is a secondary modifier.
# - The EDA noted `rater_id` has a very low Cramér V (0.019) with the target,
#   suggesting rater-specific biases are present but small. `12_blend_ridge` will
#   test whether encoding rater bias helps (result: it does not).
# - `TableVectorizer` worked without manual cardinality handling — the StringEncoder
#   for `gene` learned a dense embedding without us specifying the categories.
#
# **Architecture note:**
# `05_tabular_pipeline` has a hidden fragility: the `cv_splits` list and the model
# are separate Python objects. If we reorder the data or change the merge key,
# `cv_splits` will silently become wrong. `06_dataops` fixes this by baking the
# GroupKFold splitter directly into the graph node.
#
# **Next experiment:** `06_dataops` — rewrite the same model as a skrub DataOps
# graph where the GroupKFold splitter is structurally bound to the feature node.
# The RMSE should match exactly (7.53) — this is an architecture validation, not
# a performance experiment.
