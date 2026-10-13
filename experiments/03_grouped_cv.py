# %% [markdown]
# # Experiment 03 — Grouped CV: Ridge vs Dummy (patient holdout)
#
# **Date:** 2025-07-11
# **Goal:** Replace the random row holdout with patient-grouped 5-fold CV so
#           the RMSE estimate mirrors the Kaggle patient holdout evaluation.
# **Hub keys:** `03_dummy_cv`, `03_ridge_cv`

# %%
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from skore import Project, compare, evaluate, login

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
# ## Patient-grouped CV splits
#
# `GroupKFold` holds out whole patients — no patient appears on both sides of
# any fold. `splitter.split(X, y)` in skore does not accept `groups=`, so we
# precompute the index pairs once and reuse them for every model.

# %%
groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

# %% [markdown]
# ## Define models

# %%
dummy = DummyRegressor(strategy="mean")
ridge = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=1.0),
)

# %% [markdown]
# ## Evaluate with grouped CV

# %%
report_dummy = evaluate(dummy, X, y, splitter=cv_splits)
report_ridge = evaluate(ridge, X, y, splitter=cv_splits)

# %% [markdown]
# ## Side-by-side comparison (local display only)

# %%
comparison = compare({"dummy": report_dummy, "ridge": report_ridge})
comparison

# %% [markdown]
# ## RMSE — dummy (grouped CV)

# %%
report_dummy.metrics.rmse()

# %% [markdown]
# ## RMSE — ridge (grouped CV)

# %%
report_ridge.metrics.rmse()

# %% [markdown]
# ## Push to Skore Hub

# %%
project.put("03_dummy_cv", report_dummy)
project.put("03_ridge_cv", report_ridge)

# %% [markdown]
# ## Kaggle submission
#
# Ridge refit on all training rows, predict on test visits.

# %%
submissions_dir = Path("submissions")
submissions_dir.mkdir(exist_ok=True)

final = clone(ridge).fit(X, y)

submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test[feature_cols])
submission.to_csv(submissions_dir / "03_grouped_cv.csv", index=False)

print(f"Submission written: submissions/03_grouped_cv.csv ({len(submission)} rows)")
