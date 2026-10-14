# %% [markdown]
# # Experiment 01 — DummyRegressor mean baseline
#
# **Date:** 2025-01-01
# **Goal:** Establish the performance floor before any real modelling.
# **Hub key:** `01_dummy`
#
# ## Context and motivation
#
# Before trying any machine learning model we need a concrete, reproducible
# reference point. A `DummyRegressor(strategy="mean")` always predicts the
# training mean regardless of the input — it captures *zero* information from
# the features. Its RMSE equals the standard deviation of the target, giving
# us the worst-case baseline every subsequent experiment must beat.
#
# For this Parkinson's disease dataset, the target is the true OFF MDS-UPDRS
# motor score (a clinical severity score). The distribution has mean ≈ 37.5
# and SD ≈ 16.5, so we expect a dummy RMSE around 16.5. Any model that cannot
# beat this number extracts no usable signal from the data.
#
# **Question being tested:**
# What is the unconditional variance of the target? This sets the ceiling on
# how bad our models are allowed to be.

# %%
import pandas as pd
from sklearn.dummy import DummyRegressor
from skore import evaluate
from skore import Project, login

# %% [markdown]
# ## Load data
#
# `X_train` and `y_train` share an `Index` column used as the join key.
# We merge them so features and target are in a single DataFrame.

# %%
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test = pd.read_csv("data/X_test.csv")

visits = X_train.merge(y_train, on="Index")

# %% [markdown]
# ## Select features
#
# These 8 numeric clinical columns are the raw signals available per visit.
# `on` and `off` are MDS-UPDRS motor scores measured after and before levodopa,
# both heavily missing (~30 % and ~42 %). `ledd` is the daily
# levodopa-equivalent dose. The dummy ignores all features but we define the
# column list here so later experiments can reuse it for a direct comparison.

# %%
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
y = visits["target"]

# %% [markdown]
# ## Define and evaluate model
#
# `strategy="mean"` always predicts the mean of the training target — this is
# the simplest possible regressor and gives the variance of the target as its
# MSE. The default `splitter=0.2` draws a random 20 % row holdout.
# Patient-grouped CV (the honest evaluation) is introduced in `03_grouped_cv`.

# %%
dummy = DummyRegressor(strategy="mean")
report = evaluate(dummy, X, y)
report.metrics.rmse()

# %% [markdown]
# ## Push to Skore Hub

# %%
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace="barthon")
project.put("01_dummy", report)
# The console prints: Consult your report at https://skore.probabl.ai/…

# %% [markdown]
# ## What to look for / next steps
#
# **Expected outcome:** RMSE ≈ 16.5 (the SD of the target). Any deviation from
# this means either the holdout set has a different mean or the data has been
# resampled.
#
# **What this tells us:**
# - If a future model's RMSE is > 16.5 it is actively harmful — it makes
#   predictions worse than "always predict the mean".
# - The 36 % gap between this and `02_ridge` (RMSE ≈ 10.5) confirms that
#   even simple linear features carry real signal.
#
# **Next experiment:** `02_ridge` — replace the dummy with a Ridge regression
# on these same 8 numeric features to see how much signal a linear model
# can extract. The row holdout is kept identical so the comparison is fair.
