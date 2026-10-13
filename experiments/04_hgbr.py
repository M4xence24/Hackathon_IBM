# %% [markdown]
# # Experiment 04 — HistGradientBoostingRegressor (NaN as signal)
#
# **Date:** 2025-07-11
# **Goal:** Test whether keeping NaN as signal (no imputation) beats Ridge's
#           RMSE 10.60 on the same patient-grouped 5-fold CV.
# **Hub key:** `04_hgbr`

# %%
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from skore import Project, compare, evaluate, login

# %% [markdown]
# ## Hub credentials

# %%
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace="barthon")

# %% [markdown]
# ## Load data
#
# Same numeric feature columns as the previous experiments.
# HGBR handles NaN natively — no SimpleImputer needed.

# %%
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test = pd.read_csv("data/X_test.csv")

visits = X_train.merge(y_train, on="Index")
y = visits["target"]

feature_cols = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]
X = visits[feature_cols]

# %% [markdown]
# ## Patient-grouped CV splits (same as 03_grouped_cv)

# %%
groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

# %% [markdown]
# ## Define model
#
# No imputer: a missing `off` or `on` is routed by a learned branch in each
# tree split. `random_state=0` makes the run reproducible.

# %%
hgbr = HistGradientBoostingRegressor(random_state=0)

# %% [markdown]
# ## Evaluate with grouped CV

# %%
report_hgbr = evaluate(hgbr, X, y, splitter=cv_splits)
report_hgbr

# %% [markdown]
# ## RMSE

# %%
report_hgbr.metrics.rmse()

# %% [markdown]
# ## Push to Skore Hub

# %%
project.put("04_hgbr", report_hgbr)

# %% [markdown]
# ## Kaggle submission
#
# HGBR refit on all training rows, predict on test visits.

# %%
submissions_dir = Path("submissions")
submissions_dir.mkdir(exist_ok=True)

final = clone(hgbr).fit(X, y)

submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test[feature_cols])
submission.to_csv(submissions_dir / "04_hgbr.csv", index=False)

print(f"Submission written: submissions/04_hgbr.csv ({len(submission)} rows)")
