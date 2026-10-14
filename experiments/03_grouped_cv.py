# %% [markdown]
# # Experiment 03 — Grouped CV: Ridge vs Dummy (patient holdout)
#
# **Date:** 2025-07-11
# **Goal:** Replace the random row holdout with patient-grouped 5-fold CV so
#           the RMSE estimate mirrors the Kaggle patient holdout evaluation.
# **Hub keys:** `03_dummy_cv`, `03_ridge_cv`
#
# ## Context and motivation
#
# `02_ridge` showed a ~36 % RMSE reduction over the dummy, but used a random
# row holdout (`splitter=0.2`). This is **leaky**: because the dataset has
# ~8 visits per patient, the same patient's visits can appear on both the
# training and test sides of the split. The model memorises patient-level
# patterns from earlier visits and applies them to later ones from the same
# patient — this is not the scenario Kaggle is testing, where whole patients
# are held out.
#
# `GroupKFold(n_splits=5)` assigns **all visits of a patient to the same fold**.
# No patient ever appears on both sides of any split. This mirrors exactly how
# Kaggle withholds test patients.
#
# **Question being tested:**
# Does the ~36 % Ridge vs Dummy advantage survive the honest patient holdout?
# More precisely: is the signal in the numeric features patient-agnostic
# (generalises to unseen patients), or is it mostly patient-memorisation that
# vanished under a leaky row split?
#
# **Why 5 folds?**
# With 5 576 patients and 5 folds, each held-out fold contains ≈ 1115 patients
# (~8 900 visits). This is large enough for a stable RMSE estimate while
# keeping training folds at 80 % of the data.
#
# **What changes vs 02_ridge:**
# Only the splitter changes — same 8 features, same Ridge + median-imputer
# pipeline. Every downstream experiment from 03 onwards uses this same
# GroupKFold(5) protocol so RMSE numbers are directly comparable.

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
# `GroupKFold` ensures no patient appears in both the training and validation sides
# of any fold — this mirrors how Kaggle withholds entire patients for scoring.
# 5 folds gives a good bias-variance tradeoff with ~1115 patients per held-out fold.
groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

# %% [markdown]
# ## Define models

# %%
# Same models as `02_ridge` — only the splitter changes, so the RMSE difference
# between experiments 02 and 03 isolates the effect of the CV strategy.
dummy = DummyRegressor(strategy="mean")
ridge = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=1.0),
)

# %% [markdown]
# ## Evaluate with grouped CV

# %%
# Passing the pre-computed index pairs directly to `splitter=` bypasses skore's
# default splitter and uses our patient-grouped folds for every model.
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

# Refit on all training rows — no held-out set, so the submission model sees the
# full training distribution.
final = clone(ridge).fit(X, y)

submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test[feature_cols])
submission.to_csv(submissions_dir / "03_grouped_cv.csv", index=False)

print(f"Submission written: submissions/03_grouped_cv.csv ({len(submission)} rows)")

# %% [markdown]
# ## What to look for / next steps
#
# **Expected outcome:** Ridge RMSE ≈ 10.60 ± 0.27 — slightly worse than the
# row holdout (10.49) because the leakage is now removed.
#
# **What this tells us:**
# - The tiny increase from 10.49 → 10.60 is reassuring: the row holdout was
#   mildly optimistic but not dramatically so. The numeric features encode
#   visit-level patterns that generalise to unseen patients, not just
#   patient-specific memorisation.
# - The 36 % gap over the dummy (16.50) survives the honest split — the signal
#   is real and patient-agnostic.
# - The cross-validation uncertainty (±0.27) tells us how stable the estimate
#   is. A good next model should show improvement clearly outside this band.
#
# **Ceiling of the linear model:**
# Ridge on 8 numeric features plateaus at ~10.6 RMSE. Even with perfect alpha
# tuning (`07_ridge_cv`), this floor does not move. The bottleneck is the
# linear assumption and the forced imputation of NaN values.
#
# **Next experiment:** `04_hgbr` — replace the Ridge + imputer with
# `HistGradientBoostingRegressor`. HGBR can learn that a *missing* `off` value
# is itself a signal (patients with no `off` measurement tend to have higher
# true OFF scores), something median imputation actively destroys. Expected
# RMSE: ~7.75, a 27 % improvement.
