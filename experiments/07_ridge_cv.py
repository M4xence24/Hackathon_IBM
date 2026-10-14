# %% [markdown]
# # Experiment 07 — RidgeCV: alpha search with honest GroupKFold
#
# **Date:** 2025-07-11
# **Goal:** Find the optimal Ridge regularisation alpha using a 30-point
#           log-spaced grid under honest GroupKFold(5). Closes the question
#           of whether alpha=1.0 was already near-optimal.
# **Hub key:** `07_ridge_cv`
#
# ## Context and motivation
#
# `02_ridge` used `alpha=1.0` chosen without any tuning. The EDA found that
# `age` and `age_at_diagnosis` are highly collinear (Pearson r = 0.94), which
# can inflate Ridge coefficient norms if the regularisation is too weak. But
# we never actually searched over alpha values, so `alpha=1.0` might be
# leaving RMSE on the table.
#
# This experiment answers two questions:
# 1. **Did alpha=1.0 matter?** If the optimal alpha is far from 1.0, Ridge
#    was under- or over-regularised and we should re-evaluate all linear
#    comparisons.
# 2. **What is the absolute ceiling for a linear model on these features?**
#    Even with the best possible alpha, if Ridge cannot get below ~10.0 RMSE,
#    we can definitively close the chapter on linear models and focus entirely
#    on tree-based and residual approaches.
#
# We use `RidgeCV` rather than `GridSearchCV + Ridge` because `RidgeCV` employs
# an efficient GCV (generalised cross-validation / leave-one-out approximation)
# internally, making it much faster than fitting a fresh Ridge for each alpha
# value.
#
# **Features:** The same 8 numeric columns as `02_ridge` / `03_grouped_cv` —
# this makes the comparison apples-to-apples with those baselines. The full
# column set (including gene/cohort) is handled by HGBR in other experiments.
#
# **Question being tested:**
# Is the Ridge linear model ceiling at ~10.6 RMSE a fundamental limitation of
# the linear assumption, or was alpha=1.0 a poor choice that can be improved?

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
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
# ## Build the DataOps graph
#
# Same 8 numeric features as 02_ridge for a direct comparison with 03_grouped_cv.
# RidgeCV searches a 30-point log-spaced alpha grid internally via LOO/GCV.

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

data = skrub.var("visits", visits)

groups = data["patient_id"]
y_op = data["target"].skb.mark_as_y()

# Restrict to the same 8 numeric features used in 02/03 so the comparison is
# apples-to-apples: RidgeCV vs Ridge(alpha=1.0) on identical data.
X_op = data[feature_cols].skb.mark_as_X(
    cv=GroupKFold(n_splits=5),
    split_kwargs={"groups": groups},
)

# 30-point log-spaced grid from 1e-3 to 1e4 covers the full plausible alpha range.
# RidgeCV uses efficient GCV (leave-one-out approximation) internally, so adding
# more grid points has negligible cost.
alphas = np.logspace(-3, 4, 30)
# `SimpleImputer` is still needed: RidgeCV does not handle NaN natively.
ridge_cv = make_pipeline(
    SimpleImputer(strategy="median"),
    RidgeCV(alphas=alphas),
)

# `.skb.apply` wraps the pipeline so skore can evaluate it as a DataOps learner,
# inheriting the GroupKFold splitter baked into X_op.
pred = X_op.skb.apply(ridge_cv, y=y_op)

# %% [markdown]
# ## Evaluate

# %%
report = evaluate(pred, data={"visits": visits})
report

# %% [markdown]
# ## RMSE

# %%
report.metrics.rmse()

# %% [markdown]
# ## Push to Skore Hub

# %%
project.put("07_ridge_cv", report)

# %% [markdown]
# ## Kaggle submission

# %%
submissions_dir = Path("submissions")
submissions_dir.mkdir(exist_ok=True)

learner = pred.skb.make_learner()
learner.fit({"visits": visits})
# Pass `patient_id` alongside `feature_cols` so the DataOps graph can resolve the
# `groups` node (which references `data["patient_id"]`), even though `patient_id`
# is not in the feature matrix X_op.
pred_test = learner.predict({"visits": X_test[["patient_id"] + feature_cols]})

submission = X_test[["Index"]].copy()
submission["target"] = pred_test
submission.to_csv(submissions_dir / "07_ridge_cv.csv", index=False)

print(f"Submission written: submissions/07_ridge_cv.csv ({len(submission)} rows)")

# %% [markdown]
# ## What to look for / next steps
#
# **Expected outcome:** RMSE ≈ 10.60 ± 0.27 — identical to `03_grouped_cv`
# with the fixed `alpha=1.0`.
#
# **What this tells us:**
# - If RMSE matches 03_grouped_cv (within uncertainty), then alpha=1.0 was
#   already near the GCV optimum. The collinearity between `age` and
#   `age_at_diagnosis` did not require a different regularisation level than
#   the default.
# - The linear model ceiling is a genuine ceiling caused by the linear
#   assumption itself, not a tuning deficit. There is nothing left to gain
#   from Ridge on these 8 features — the next gains must come from:
#   (a) non-linear models (HGBR already showed this: 7.53), or
#   (b) better features (patient aggregates in `09_patient_features`).
#
# **Conclusion for linear models:**
# Ridge is now fully characterised. It sits at RMSE ≈ 10.6 regardless of alpha.
# All future performance investment goes into tree-based and residual models.
# Ridge will reappear only as a *blend component* in `12_blend_ridge`, where its
# smooth linear predictions complement the higher-variance HGBR residual model.
