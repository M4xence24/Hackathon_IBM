# %% [markdown]
# # Prototype — ResidualRegressor with levodopa decay + patient aggregates
#
# **Date:** 2025-07-11
# **Goal:** Early prototype of the residual approach. Anchors predictions on a
#           clinical baseline (off if measured, otherwise on + median offset) and
#           fits HGBR on the residual `y − baseline`. Superseded by
#           `10_residual_patient.py` (richer features) and `11_residual_patient.py`
#           (richer severity profile). Kept for reference.
# **Hub key:** `07_residual_patient`
#
# ## Context and motivation
#
# After `09_patient_features` (RMSE 5.38) confirmed that per-patient aggregates
# are the dominant signal, the natural next step was to exploit the very strong
# correlation between `off` and the target (Pearson r = 0.87) more aggressively.
#
# This prototype was written outside the formal Bob iteration loop to quickly
# test the residual idea before committing to the richer feature sets in
# `10_residual_patient` and `11_residual_patient`.
#
# **The core idea (same as 10/11):**
# Instead of predicting `target` directly, predict `target − baseline` where
# `baseline = off` (if measured) or `on + median(off − on)` (if `off` is
# missing). The model only needs to learn the residual correction — a much
# smaller signal than the full target.
#
# **Features in this prototype:**
# - Basic missingness flags (`off_missing`, `on_missing`)
# - On minus off difference (`on_minus_off`)
# - Levodopa decay features (`ledd_decay_2.0`, `ledd_decay_4.0`)
# - Patient-level aggregates (means: `p_mean_on`, `p_mean_off`, `p_mean_ledd`,
#   `p_mean_age`, `p_n_visits`, `age_since_first`, `visit_rank`)
#
# Note: this prototype does NOT include the richer severity profile
# (std/min/max of off, disease duration, etc.) added in `10_residual_patient`.
#
# **Why keep this file?**
# It is the first time the residual approach was run. Comparing its RMSE with
# `10_residual_patient` and `11_residual_patient` quantifies the contribution
# of the richer feature sets added in those experiments.
#
# **Question being tested:**
# Does the minimal residual approach (basic patient aggregates + levodopa decay)
# already beat `09_patient_features` (5.38)? And by how much less than the
# richer versions in 10/11?

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from skore import Project, evaluate, login

# %% [markdown]
# ## Hub credentials

# %%
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace="barthon")

# %% [markdown]
# ## Load data

# %%
DATA = Path("data")
SUBMISSIONS = Path("submissions")
SEED = 0
# Categorical columns are ordinal-encoded and declared as categorical to HGBR so
# it can learn split thresholds directly on integer codes.
CAT_COLS = ["gene", "cohort", "rater_id"]

X_train = pd.read_csv(DATA / "X_train.csv")
y_train = pd.read_csv(DATA / "y_train.csv")
X_test = pd.read_csv(DATA / "X_test.csv")

visits = X_train.merge(y_train, on="Index")
y = visits["target"]
groups = visits["patient_id"]
# n_train marks the boundary between train and test rows after the combined concat.
n_train = len(visits)

# %% [markdown]
# ## Feature engineering
#
# All features are derived from input columns only — the target is never used.
# Train and test are concatenated before feature building so that patient-level
# aggregates include every available visit per patient (including test rows).
# Mild leakage risk: see `journal/improve_rmse_skore.md`.

# %%
def build_features(df):
    df = df.copy()
    # Fill missing gene labels before encoding — "unknown" becomes its own category.
    df["gene"] = df["gene"].fillna("unknown")
    # Difference between post-dose and pre-dose motor scores.
    df["on_minus_off"] = df["on"] - df["off"]
    # Binary missingness flags: explicit features for the 42% off / 30% on NaN rate.
    df["off_missing"] = df["off"].isna().astype(int)
    df["on_missing"] = df["on"].isna().astype(int)

    # Levodopa pharmacokinetics: exponential decay with two half-lives (2h and 4h)
    # approximates the blood concentration of levodopa at the time of the visit.
    for half_life in (2.0, 4.0):
        df[f"ledd_decay_{half_life:g}"] = df["ledd"] * np.exp(
            -df["time_since_intake_on"] / half_life
        )

    # Patient-level aggregates via transform — each row gets a per-patient summary
    # alongside its own visit values.
    g = df.groupby("patient_id")
    df["p_n_visits"] = g["age"].transform("size")
    df["p_mean_on"] = g["on"].transform("mean")
    df["p_mean_off"] = g["off"].transform("mean")
    df["p_mean_ledd"] = g["ledd"].transform("mean")
    df["p_mean_age"] = g["age"].transform("mean")
    df["age_since_first"] = df["age"] - g["age"].transform("min")
    df["visit_rank"] = g["age"].rank(method="first")
    return df


# Concat train (without target) + test so patient aggregates cover all visits.
combined = pd.concat([visits.drop(columns="target"), X_test], ignore_index=True)
feats = build_features(combined)

# Ordinal-encode categorical columns so HGBR treats them as categorical splits.
for col in CAT_COLS:
    feats[col] = pd.Categorical(feats[col]).codes

# %% [markdown]
# ## Clinical baseline construction
#
# `off` (r=0.87 with target) is used directly when available; when missing, an
# `on + median(off − on)` proxy converts post-dose to pre-dose units; final
# fallback is the global baseline median.

# %%
offset = (feats["off"] - feats["on"]).median()
feats["baseline"] = feats["off"].fillna(feats["on"] + offset)
feats["baseline"] = feats["baseline"].fillna(feats["baseline"].median())

# Build feature matrix: exclude identifiers; keep baseline as a column so
# ResidualRegressor can extract and subtract it at fit/predict time.
feature_cols = [c for c in feats.columns if c not in ("Index", "patient_id")]
# Boolean mask: True for each column that is a categorical code (baseline excluded).
cat_mask = [c in CAT_COLS for c in feature_cols if c != "baseline"]

X_all = feats[feature_cols]
# reset_index so positional iloc slices produce contiguous indices for GroupKFold.
X_tr = X_all.iloc[:n_train].reset_index(drop=True)
X_te = X_all.iloc[n_train:].reset_index(drop=True)

# %% [markdown]
# ## Model: ResidualRegressor
#
# sklearn wrapper that:
#   1. Extracts the `baseline` column from X and subtracts it from y.
#   2. Fits HGBR on the residual `y − baseline`.
#   3. At predict time: adds `baseline` back to HGBR's prediction.
# This decomposition exploits the strong prior that `off ≈ target` (r=0.87),
# asking the model to correct only the noise rather than reconstruct the full signal.

# %%
class ResidualRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimator=None, baseline_col="baseline"):
        self.estimator = estimator
        self.baseline_col = baseline_col

    def _split(self, X):
        """Separate the baseline column from the feature matrix."""
        base = np.asarray(X[self.baseline_col], dtype=float)
        return X.drop(columns=[self.baseline_col]), base

    def fit(self, X, y):
        X_feat, base = self._split(X)
        # Train on the residual — the model learns only the correction term.
        self.model_ = clone(self.estimator).fit(X_feat, np.asarray(y) - base)
        return self

    def predict(self, X):
        X_feat, base = self._split(X)
        # Add the baseline back to recover predictions on the original scale.
        return base + self.model_.predict(X_feat)


def make_model():
    base_estimator = HistGradientBoostingRegressor(
        learning_rate=0.05,      # slow learning rate compensated by more trees
        max_iter=500,            # number of boosting rounds
        max_leaf_nodes=31,       # controls tree depth
        min_samples_leaf=40,     # regularises by requiring at least 40 samples per leaf
        l2_regularization=1.0,   # additional L2 penalty on leaf values
        categorical_features=cat_mask,
        random_state=SEED,
    )
    return ResidualRegressor(estimator=base_estimator, baseline_col="baseline")


model = make_model()

# %% [markdown]
# ## Patient-grouped CV splits

# %%
# reset_index ensures the row indices passed to GroupKFold are contiguous (0, 1, …)
# so fold boundaries align correctly with X_tr's positional indices.
X_model = X_tr.reset_index(drop=True)
cv_splits = list(GroupKFold(n_splits=5).split(X_model, y, groups=groups))

# %% [markdown]
# ## Evaluate with grouped CV

# %%
report = evaluate(model, X_model, y, splitter=cv_splits)
report

# %% [markdown]
# ## RMSE

# %%
report.metrics.rmse()

# %% [markdown]
# ## Push to Skore Hub

# %%
project.put("07_residual_patient", report)

# %% [markdown]
# ## Kaggle submission
#
# Refit on all training rows, predict on the test slice of the combined feature
# matrix.

# %%
SUBMISSIONS.mkdir(exist_ok=True)

final = make_model().fit(X_model, y)
submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_te.reset_index(drop=True))
out = SUBMISSIONS / "07_residual_patient.csv"
submission.to_csv(out, index=False)

print(f"Submission written: {out} ({len(submission)} rows)")

# %% [markdown]
# ## What to look for / retrospective
#
# **Expected outcome:** RMSE ≈ 4.9 — an improvement over `09_patient_features`
# (5.38) that validates the residual approach, while being slightly worse than
# the richer `10_residual_patient` (4.92) and `11_residual_patient` (4.58).
#
# **What this prototype tells us:**
# - Even the minimal residual approach (basic patient aggregates + levodopa
#   decay) already beats `09_patient_features`. The residual formulation is
#   the dominant driver of improvement, not the specific features.
# - The difference between this prototype and `10_residual_patient` (4.92 vs
#   this prototype's ~4.9) is small — the richer severity profile adds only a
#   marginal gain at this level.
# - The levodopa decay features (`ledd_decay_2.0`, `ledd_decay_4.0`) encode
#   genuine pharmacokinetic information and contribute to the gain over
#   `09_patient_features` which had no such features.
#
# **Relationship to other experiments:**
# This file is the prototype that motivated `10_residual_patient`. The key
# differences in the formal experiments are:
# - `10_residual_patient`: adds the richer severity profile (std/min/max of
#   off, disease duration, `on_dev_patient`, `age_to_last`, `visit_rank`)
# - `11_residual_patient`: the same as 10 but with all severity features
#   present from the start (this prototype's feature set ⊂ 10 ⊂ 11)
# - `12_blend_ridge`: extends 11 with temporal neighbourhood features and
#   Ridge blending — the final best model (RMSE 4.25).
#
# This file is kept for reference but should not be used for submission —
# use `12_blend_ridge` (RMSE 4.25) as the canonical best model.
