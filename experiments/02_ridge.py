# %% [markdown]
# # Experiment 02 — Ridge vs Dummy
#
# **Date:** 2025-01-01
# **Goal:** Check whether a Ridge regression on numeric clinical features
#           beats the dummy mean baseline on RMSE (row holdout, splitter=0.2).
# **Hub key:** `02_ridge`
#
# ## Context and motivation
#
# `01_dummy` established the performance floor at RMSE ≈ 16.5 — the model
# predicts the training mean for every visit regardless of what the features say.
# The question now is whether the raw numeric clinical features carry any
# predictive signal at all.
#
# We choose **Ridge** (L2-regularised least squares) as the first real model
# because:
# 1. It is the simplest model that actually reads the features.
# 2. Its RMSE is interpretable: if Ridge beats dummy, the numeric features
#    carry a linear signal; if it doesn't, no linear model is worth pursuing.
# 3. The EDA found a strong collinearity between `age` and `age_at_diagnosis`
#    (Pearson r = 0.94). Ridge's L2 penalty naturally handles this without
#    manual column removal.
#
# **Question being tested:**
# Does a linear combination of the 8 numeric clinical features (after median
# imputation) beat the constant-mean baseline? And by how much?
#
# **Known limitation of this experiment:**
# `splitter=0.2` draws a random 20 % row holdout. Because the dataset has
# ~8 visits per patient, the same patient can appear in both the training and
# test sets — this is data leakage. The RMSE here will be *optimistic* compared
# to the Kaggle leaderboard (which withholds whole patients). This limitation
# is fixed in `03_grouped_cv`.

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
# `dummy` always predicts the training mean — our performance floor (RMSE ≈ 16.5).
dummy = DummyRegressor(strategy="mean")

# `SimpleImputer(median)` fills missing `on`/`off`/`ledd` values before Ridge.
# Median is more robust to the heavy right-skew in `ledd` than mean imputation.
# `Ridge(alpha=1.0)` adds L2 regularisation to prevent `age` and `age_at_diagnosis`
# (r=0.94 collinearity) from inflating coefficient norms.
ridge = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=1.0),
)

# %% [markdown]
# ## Evaluate each model (row holdout, splitter=0.2)
#
# `splitter=0.2` draws a random 20 % row holdout — the same split style used
# for the Day 1 dummy. Visit-level leakage (same patient on both sides) is
# expected here; grouped CV is addressed in the next experiment (03_grouped_cv).

# %%
# `splitter=0.2` is a random 20% row holdout — same as experiment 01 so results
# are directly comparable. Grouped CV (honest patient holdout) follows in
# 03_grouped_cv.
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

# `clone` creates a fresh unfitted copy — prevents any CV state leaking into the
# final model that will be used for test-set predictions.
final = clone(ridge).fit(X[feature_cols], y)

submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test[feature_cols])
submission.to_csv(submissions_dir / "02_ridge.csv", index=False)

print(f"Submission written: submissions/02_ridge.csv ({len(submission)} rows)")

# %% [markdown]
# ## What to look for / next steps
#
# **Expected outcome:** Ridge RMSE ≈ 10.5 — a ~36 % reduction from the dummy
# baseline of 16.5. This confirms that the numeric features carry real linear
# signal.
#
# **What this tells us:**
# - A ~36 % gap proves the features are informative. This is enough to justify
#   investing in more complex models.
# - The row holdout RMSE will look slightly better than the grouped CV RMSE
#   because same-patient visits sit on both sides of the split. The true
#   generalisation gap to unseen patients is measured in `03_grouped_cv`.
# - `alpha=1.0` was chosen without tuning. `07_ridge_cv` will confirm whether
#   this matters (spoiler: it does not — alpha tuning yields no improvement on
#   these features).
#
# **Next experiment:** `03_grouped_cv` — swap the row holdout for a patient-
# grouped 5-fold CV. This is the honest evaluation that mirrors how Kaggle
# withholds entire patients for scoring. The RMSE will increase slightly (10.49
# → 10.60) confirming the row holdout was mildly optimistic.
