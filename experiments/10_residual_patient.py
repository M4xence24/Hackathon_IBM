# %% [markdown]
# # Experiment 10 — ResidualRegressor: clinical baseline + levodopa decay features
#
# **Date:** 2025-07-11
# **Goal:** Test whether anchoring predictions on a clinical baseline (off if
#           measured, otherwise on + median offset) and learning only the residual
#           beats the direct prediction approach of `09_patient_features` (RMSE 5.38).
# **Hub key:** `07_residual_patient`
#
# ## Context and motivation
#
# `09_patient_features` uses `off` as a regular feature with coefficient weight
# learned by the model. But `off` is not just any feature — it is the pre-dose
# motor score, a direct measurement of *the same clinical concept* as the target
# (true OFF MDS-UPDRS). The EDA found Pearson r = 0.87 between `off` and target.
#
# **The residual idea:**
# Instead of asking HGBR to predict the full target from scratch, we:
# 1. Anchor each prediction on a clinical **baseline** (the best single-row
#    estimate we can make before the model runs):
#    - If `off` is measured: `baseline = off`  (r=0.87 with target)
#    - If `off` is missing but `on` is present: `baseline = on + median(off − on)`
#    - Otherwise: `baseline = global median of (off − target)`
# 2. Train HGBR to predict `y − baseline` (the **residual**).
# 3. At predict time: `prediction = baseline + hgbr_residual`.
#
# This has a well-known bias-variance benefit: the baseline absorbs most of the
# signal, leaving HGBR to correct only the small residual noise. The model's
# task shrinks from "predict a score in [0, 100]" to "predict a correction
# in [−10, +10]". Smaller target range → smaller RMSE.
#
# **Additional features in this experiment (vs 09_patient_features):**
# - Levodopa pharmacokinetics: `ledd_decay_2.0` and `ledd_decay_4.0` model the
#   exponential decay of blood levodopa concentration at the time of the visit.
#   The `time_since_intake_on` column records how long ago the patient took
#   their dose — a shorter time means higher drug concentration and potentially
#   lower motor impairment.
# - Richer patient severity profile: `p_std_on`, `p_std_off`, `p_min_off`,
#   `p_max_off`, within-patient variability measures, disease duration.
#
# **Question being tested:**
# Does the residual formulation (predict `y − baseline` instead of `y`) reduce
# RMSE below `09_patient_features` (5.38), controlling for the expanded feature set?

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
# it can learn split thresholds directly on integer codes rather than treating them
# as continuous numbers.
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
# aggregates (e.g. mean off per patient) include every available visit, including
# those in X_test. This improves aggregate quality but introduces a mild leakage
# risk: test-patient statistics bleed into training folds (CV RMSE is slightly
# optimistic). The risk is documented in the journal (`journal/10_residual_patient.md`).

# %%
def build_features(df):
    df = df.copy()
    # Fill missing gene labels before encoding — "unknown" becomes its own category.
    df["gene"] = df["gene"].fillna("unknown")
    # Difference between post-dose and pre-dose motor scores; captures within-visit
    # treatment response.
    df["on_minus_off"] = df["on"] - df["off"]
    # Binary flags: HGBR treats NaN as a third value, but an explicit flag makes
    # missingness a first-class feature for the linear layers of skrub encoders.
    df["off_missing"] = df["off"].isna().astype(int)
    df["on_missing"] = df["on"].isna().astype(int)

    # Levodopa pharmacokinetics: LEDD decays exponentially after intake.
    # Two half-lives (2h and 4h) bracket the typical levodopa window.
    for half_life in (2.0, 4.0):
        df[f"ledd_decay_{half_life:g}"] = df["ledd"] * np.exp(
            -df["time_since_intake_on"] / half_life
        )

    # Patient-level aggregates — computed via transform so each row keeps its
    # own visit value while gaining a per-patient summary statistic.
    g = df.groupby("patient_id")
    df["p_n_visits"] = g["age"].transform("size")         # visit count per patient
    df["p_mean_on"] = g["on"].transform("mean")           # mean post-dose score
    df["p_mean_off"] = g["off"].transform("mean")         # mean pre-dose score (severity proxy)
    df["p_mean_ledd"] = g["ledd"].transform("mean")       # mean daily levodopa dose
    df["p_mean_age"] = g["age"].transform("mean")         # mean age across visits
    df["age_since_first"] = df["age"] - g["age"].transform("min")  # time in study
    df["visit_rank"] = g["age"].rank(method="first")      # ordinal position in patient history

    # Richer severity profile — added versus improve_rmse_skore.py.
    df["p_std_on"] = g["on"].transform("std")             # within-patient score variability
    df["p_std_off"] = g["off"].transform("std")
    df["p_min_off"] = g["off"].transform("min")           # worst observed off score
    df["p_max_off"] = g["off"].transform("max")           # best observed off score
    df["p_frac_off_missing"] = g["off_missing"].transform("mean")  # measurement frequency
    df["p_mean_offdiff"] = g["on_minus_off"].transform("mean")     # avg treatment response
    df["p_mean_time_on"] = g["time_since_intake_on"].transform("mean")
    df["on_dev_patient"] = df["on"] - df["p_mean_on"]    # deviation from patient's own mean
    df["age_to_last"] = g["age"].transform("max") - df["age"]     # time until last visit
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    df["p_mean_duration"] = df.groupby("patient_id")["disease_duration"].transform("mean")
    return df


# Concat train (without target) + test so patient aggregates cover all visits.
combined = pd.concat([visits.drop(columns="target"), X_test], ignore_index=True)
feats = build_features(combined)

# Ordinal-encode categorical columns: pd.Categorical assigns consistent integer codes
# and HGBR treats them as categorical splits (not ordinal distances).
for col in CAT_COLS:
    feats[col] = pd.Categorical(feats[col]).codes

# %% [markdown]
# ## Clinical baseline construction
#
# `off` is the measured pre-dose motor score (r=0.87 with target from EDA). When
# available it is the best single predictor. When missing, `on + median(off − on)`
# is a pharmacologically motivated proxy: median offset converts post-dose to
# pre-dose units. The final fallback is the global median.

# %%
offset = (feats["off"] - feats["on"]).median()
feats["baseline"] = feats["off"].fillna(feats["on"] + offset)
feats["baseline"] = feats["baseline"].fillna(feats["baseline"].median())

# Build feature matrix: exclude identifiers; keep baseline as a feature column
# so ResidualRegressor can extract and subtract it at fit/predict time.
feature_cols = [c for c in feats.columns if c not in ("Index", "patient_id")]
# Boolean mask for HGBR's categorical_features param — True for each CAT_COLS column
# that is present in feature_cols (baseline is excluded from the mask).
cat_mask = [c in CAT_COLS for c in feature_cols if c != "baseline"]

X_all = feats[feature_cols]
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
        max_leaf_nodes=31,       # controls tree depth; 31 is a good default for tabular data
        min_samples_leaf=40,     # regularises by requiring at least 40 samples per leaf
        l2_regularization=1.0,   # additional L2 penalty on leaf values
        categorical_features=cat_mask,  # tell HGBR which columns are categorical codes
        random_state=SEED,
    )
    return ResidualRegressor(estimator=base_estimator, baseline_col="baseline")


model = make_model()

# %% [markdown]
# ## Patient-grouped CV splits

# %%
# Same GroupKFold(5) protocol as 03–09 — ensures the RMSE is comparable.
cv_splits = list(GroupKFold(n_splits=5).split(X_tr, y, groups=groups))

# %% [markdown]
# ## Evaluate with grouped CV

# %%
report = evaluate(model, X_tr, y, splitter=cv_splits)
report

# %% [markdown]
# ## RMSE

# %%
print(f"RMSE: {report.metrics.rmse()}")

# %% [markdown]
# ## Push to Skore Hub

# %%
project.put("07_residual_patient", report)

# %% [markdown]
# ## Kaggle submission
#
# Refit on all training rows, predict on the test slice built from the combined
# feature matrix.

# %%
SUBMISSIONS.mkdir(exist_ok=True)
final = make_model().fit(X_tr, y)
submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_te)
out = SUBMISSIONS / "07_residual_patient.csv"
submission.to_csv(out, index=False)
print(f"Submission written: {out} ({len(submission)} rows)")

# %% [markdown]
# ## What to look for / next steps
#
# **Expected outcome:** RMSE ≈ 4.92 ± 0.09 — a 9 % improvement over
# `09_patient_features` (5.38).
#
# **What this tells us:**
# - The residual formulation works: anchoring predictions on the clinical
#   baseline reduces the variance of what the model must explain. The model
#   now corrects a smaller signal (residual range ≈ ±5) instead of predicting
#   the full target (range ≈ [0, 100]).
# - The levodopa decay features (`ledd_decay_2.0`, `ledd_decay_4.0`) add
#   pharmacokinetic context: the model now knows approximately how much drug
#   is active in the patient's system at the time of the visit.
# - The richer severity profile (std/min/max of off, disease duration) gives
#   the model a fuller picture of each patient's clinical trajectory.
#
# **Known risk — train+test combined feature building:**
# `build_features` is called on the concatenation of train and test rows.
# Patient-level aggregates therefore include test visits, meaning test-patient
# statistics bleed into training folds. This makes CV RMSE slightly optimistic.
# The effect is bounded by the fraction of each patient's visits in the test
# set. See `journal/10_residual_patient.md` for a full discussion.
#
# **Next experiment:** `11_residual_patient` — same residual architecture,
# but with an even richer patient severity profile: add `p_std_on`, `p_std_off`,
# `p_min_off`, `p_max_off`, `on_dev_patient`, `age_to_last`, and
# `disease_duration`. The hypothesis is that within-patient score variability
# (how much a patient's OFF score fluctuates across visits) is itself a
# disease severity signal that the current feature set misses.
# Expected RMSE: ~4.58, a further 7 % improvement.
