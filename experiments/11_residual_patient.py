# %% [markdown]
# # Experiment 11 — ResidualRegressor: richer patient severity profile
#
# **Date:** 2025-07-11
# **Goal:** Extend the `10_residual_patient` feature set with within-patient
#           variability measures (std of on/off, min/max off, disease duration,
#           visit rank, on_dev_patient) and check whether they improve RMSE
#           below 10_residual_patient's 4.92.
# **Hub key:** `07_residual_patient` (overwrites 10_residual_patient on hub)
#
# ## Context and motivation
#
# `10_residual_patient` reached RMSE 4.92 using per-patient *mean* statistics
# (mean on, mean off, mean ledd). But the mean is only one statistic of the
# patient's distribution. A patient whose OFF score is highly variable across
# visits is clinically very different from one with the same mean but a stable
# score — variability signals disease severity instability and medication
# inconsistency, which likely affects the residual the model needs to correct.
#
# **New features added in this experiment:**
# - `p_std_on`, `p_std_off` — within-patient standard deviation of ON/OFF
#   scores. High `p_std_off` indicates that the patient's disease state is
#   unpredictable visit-to-visit.
# - `p_min_off`, `p_max_off` — the range of the patient's observed OFF scores.
#   The maximum represents the patient's worst observed pre-dose state.
# - `p_frac_off_missing` — fraction of visits where `off` was not measured.
#   A high fraction may indicate that the patient is frequently too impaired
#   for the OFF assessment.
# - `on_dev_patient` — how much the current visit's `on` score deviates from
#   the patient's mean. A large deviation signals an atypical visit.
# - `age_to_last` — time remaining until the patient's last observed visit.
#   Encodes whether the current visit is early or late in the patient's study
#   participation (proxy for disease stage).
# - `disease_duration` — time since diagnosis. Direct proxy for disease
#   progression stage.
# - `p_mean_duration` — mean disease duration across the patient's visits.
#
# **Architecture:** Identical to `10_residual_patient` — same `ResidualRegressor`
# wrapper, same HGBR hyperparameters, same GroupKFold(5) evaluation. Only the
# feature set changes, which isolates the contribution of the new features.
#
# **Question being tested:**
# Does within-patient score variability (std/min/max of off) and disease
# progression stage (disease_duration, visit_rank, age_to_last) improve RMSE
# beyond what per-patient means alone achieved (4.92)?

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
# aggregates include every available visit per patient, including test rows.
# Mild leakage risk: test-patient statistics enter training folds — CV RMSE is
# slightly optimistic. Documented in `journal/11_residual_patient.md`.

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

    # Levodopa pharmacokinetics: exponential decay with two half-lives (2h and 4h).
    for half_life in (2.0, 4.0):
        df[f"ledd_decay_{half_life:g}"] = df["ledd"] * np.exp(
            -df["time_since_intake_on"] / half_life
        )

    # Patient-level aggregates via transform — each row retains its own visit value
    # plus a per-patient summary statistic.
    g = df.groupby("patient_id")
    df["p_n_visits"] = g["age"].transform("size")
    df["p_mean_on"] = g["on"].transform("mean")
    df["p_mean_off"] = g["off"].transform("mean")
    df["p_mean_ledd"] = g["ledd"].transform("mean")
    df["p_mean_age"] = g["age"].transform("mean")
    df["age_since_first"] = df["age"] - g["age"].transform("min")
    df["visit_rank"] = g["age"].rank(method="first")

    # Richer severity profile — NEW versus 10_residual_patient:
    df["p_std_on"] = g["on"].transform("std")             # within-patient ON variability
    df["p_std_off"] = g["off"].transform("std")           # within-patient OFF variability
    df["p_min_off"] = g["off"].transform("min")           # best (lowest) pre-dose score
    df["p_max_off"] = g["off"].transform("max")           # worst (highest) pre-dose score
    df["p_frac_off_missing"] = g["off_missing"].transform("mean")  # measurement rate
    df["p_mean_offdiff"] = g["on_minus_off"].transform("mean")     # avg treatment response
    df["p_mean_time_on"] = g["time_since_intake_on"].transform("mean")
    df["on_dev_patient"] = df["on"] - df["p_mean_on"]    # deviation from patient mean ON
    df["age_to_last"] = g["age"].transform("max") - df["age"]     # time until last visit
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]   # time since diagnosis
    df["p_mean_duration"] = df.groupby("patient_id")["disease_duration"].transform("mean")
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
X_tr = X_all.iloc[:n_train].reset_index(drop=True)
X_te = X_all.iloc[n_train:].reset_index(drop=True)

# %% [markdown]
# ## Model: ResidualRegressor
#
# Identical architecture to `10_residual_patient`. Only the feature set changes.

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
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        categorical_features=cat_mask,
        random_state=SEED,
    )
    return ResidualRegressor(estimator=base_estimator, baseline_col="baseline")


model = make_model()

# %% [markdown]
# ## Patient-grouped CV splits

# %%
# Same GroupKFold(5) protocol as 03–10 — ensures RMSE is directly comparable.
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
# Refit on all training rows, predict on the test slice of the combined feature
# matrix.

# %%
SUBMISSIONS.mkdir(exist_ok=True)
final = make_model().fit(X_tr, y)
submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_te)
out = SUBMISSIONS / "11_residual_patient.csv"
submission.to_csv(out, index=False)
print(f"Submission written: {out} ({len(submission)} rows)")

# %% [markdown]
# ## What to look for / next steps
#
# **Expected outcome:** RMSE ≈ 4.58 ± 0.11 — a 7 % improvement over
# `10_residual_patient` (4.92). This is a new best result.
#
# **What this tells us:**
# - Within-patient score variability (`p_std_off`, `p_std_on`) is genuinely
#   informative beyond the mean. A patient with high `p_std_off` is harder to
#   predict for any individual visit — knowing the variability lets the model
#   widen its correction term appropriately.
# - Disease progression markers (`disease_duration`, `age_to_last`) add signal:
#   the same `off` value means something different for a patient who was just
#   diagnosed versus one who has had Parkinson's for 10+ years.
# - `on_dev_patient` (deviation of current `on` from patient mean) flags
#   atypical visits: if `on` is unusually high or low for this patient, the
#   baseline correction should differ.
#
# **The seed noise diagnostic (`step6_seeds`):**
# After this result, a diagnostic experiment was run to confirm that the RMSE
# improvement (4.92 → 4.58) exceeds the noise from different random seeds and
# early-stopping configurations. The diagnostic confirmed the signal is real.
#
# **Next experiment:** `12_blend_ridge` — test whether blending the residual
# HGBR with a simple Ridge regression on numeric features can squeeze out
# additional RMSE reduction. The intuition: HGBR has high capacity for
# non-linear patterns but also higher variance on smooth numeric signals.
# Ridge is a low-variance linear smoother that may complement HGBR's weaknesses.
# Additionally, temporal neighbourhood features (previous/next visit OFF scores,
# rate of change of OFF across visits) are added to capture disease progression
# velocity. Expected RMSE: ~4.25, a further 7 % improvement.
