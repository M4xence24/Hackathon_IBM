# %% [markdown]
# # Experiment 05 — skrub tabular_pipeline (all columns, strings included)
#
# **Date:** 2025-07-11
# **Goal:** Include string columns (gene, cohort) via skrub's tabular_pipeline
#           and test whether the full feature set lowers RMSE below HGBR's 7.75.
# **Hub key:** `05_tabular_pipeline`

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

# Drop identifiers and target — keep all clinical / genetic / demographic columns.
X_full = visits.drop(columns=["Index", "patient_id", "target"])

# %% [markdown]
# ## Patient-grouped CV splits (same logic as 03 and 04)

# %%
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

test_features = X_test.drop(columns=["Index", "patient_id"], errors="ignore")

final = clone(model).fit(X_full, y)

submission = X_test[["Index"]].copy()
submission["target"] = final.predict(test_features)
submission.to_csv(submissions_dir / "05_tabular_pipeline.csv", index=False)

print(f"Submission written: submissions/05_tabular_pipeline.csv ({len(submission)} rows)")
