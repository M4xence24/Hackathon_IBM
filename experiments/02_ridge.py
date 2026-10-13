# %% [markdown]
# # Experiment 02 — Ridge vs Dummy
#
# **Date:** 2025-01-01
# **Goal:** Check whether a Ridge regression on numeric clinical features
#           beats the dummy mean baseline on RMSE (row holdout, splitter=0.2).
# **Hub key:** `02_ridge`

# %%
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from skore import Project, compare, evaluate, login

# %% [markdown]
# ## Hub credentials
#
# Reuse the same login pattern as dummy.py — no browser needed.

# %%
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace="barthon")

# %% [markdown]
# ## Load data
#
# Same feature columns as `dummy.py`. `on`, `off`, and `ledd` have missing
# values — Ridge cannot handle NaNs, so we median-impute inside the pipeline.

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
# ## Define models
#
# `dummy` always predicts the training mean — our floor.
# `ridge` median-imputes then fits a Ridge (L2-regularised linear model).
# `alpha=1.0` pulls weights toward zero so one noisy column cannot dominate.

# %%
dummy = DummyRegressor(strategy="mean")
ridge = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=1.0),
)

# %% [markdown]
# ## Evaluate each model (row holdout, splitter=0.2)
#
# `splitter=0.2` draws a random 20 % row holdout — the same split style used
# for the Day 1 dummy. Visit-level leakage (same patient on both sides) is
# expected here; grouped CV is addressed in the next experiment (B1).

# %%
report_dummy = evaluate(dummy, X, y, splitter=0.2)
report_ridge = evaluate(ridge, X, y, splitter=0.2)

# %% [markdown]
# ## Compare both models in one view

# %%
comparison = compare({"dummy": report_dummy, "ridge": report_ridge})
comparison

# %% [markdown]
# ## RMSE — dummy

# %%
report_dummy.metrics.rmse()

# %% [markdown]
# ## RMSE — ridge

# %%
report_ridge.metrics.rmse()

# %% [markdown]
# ## Push to Skore Hub
#
# `put` accepts individual `EstimatorReport` / `CrossValidationReport` objects.
# Both are stored under the same project so Skore Hub can compare them side by side.

# %%
project.put("01_dummy", report_dummy)
project.put("02_ridge", report_ridge)

# %% [markdown]
# ## Kaggle submission
#
# Refit Ridge on **all** training rows (no CV split), predict on test visits,
# write `submissions/02_ridge.csv` with header `Index,target`.

# %%
submissions_dir = Path("submissions")
submissions_dir.mkdir(exist_ok=True)

final = clone(ridge).fit(X[feature_cols], y)

submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test[feature_cols])
submission.to_csv(submissions_dir / "02_ridge.csv", index=False)

print(f"Submission written: submissions/02_ridge.csv ({len(submission)} rows)")
