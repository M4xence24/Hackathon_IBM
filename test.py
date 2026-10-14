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

report = evaluate(
    {"dummy": dummy, "ridge": ridge},
    X,
    y,
)

project.put("Dummy_Ridge_Comparison", report)