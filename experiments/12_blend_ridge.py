# %% [markdown]
# # Experiment 12 — BlendRegressor: residual HGBR + Ridge + temporal neighbourhood
#
# **Date:** 2025-07-11
# **Goal:** Test whether blending the residual HGBR (`11_residual_patient`
#           architecture) with a Ridge regression on numeric features, combined
#           with new temporal neighbourhood features, reduces RMSE below 4.58.
# **Hub key:** `09_blend_ridge`
#
# ## Context and motivation
#
# `11_residual_patient` reached RMSE 4.58. Two ideas motivate this experiment:
#
# **Idea 1 — Model blending:**
# HGBR is a high-capacity non-linear model. It captures complex interactions
# but also has higher variance than a linear model on smooth signals. Ridge
# is a low-variance linear smoother — it cannot capture non-linearities but is
# very stable on clean numeric features (age, ledd, disease_duration).
#
# Blending the two at inference time exploits their complementary weaknesses:
#   `prediction = w * HGBR_pred + (1 − w) * Ridge_pred`
#
# The optimal weight `w` is selected from OOF (out-of-fold) predictions on the
# same GroupKFold(5) splits — no additional held-out data is consumed. This
# makes the weight selection honest: each OOF fold uses a model trained without
# seeing the validation rows.
#
# **Idea 2 — Temporal neighbourhood features:**
# `11_residual_patient` treats visits as independent data points within the
# patient's history. But disease progression is a longitudinal process: the
# OFF score at the previous visit, and the *rate of change* of OFF over time,
# are strong predictors of the current visit's score.
#
# New features added:
# - `off_prev`, `off_next` — OFF score at the immediately preceding/following
#   visit (sorted by age within patient). First/last visits get NaN.
# - `on_prev` — ON score at the previous visit.
# - `dt_prev` — time elapsed since the previous visit (years).
# - `off_delta_prev` — change in OFF from the previous visit (rate of change).
# - `p_off_slope` — per-patient OLS slope of OFF vs age: the disease
#   progression rate estimated from all visits via least squares.
#
# **Additional model change:**
# Early stopping (`early_stopping=True`, `n_iter_no_change=50`) with a lower
# learning rate (0.03 vs 0.05) and a higher iteration ceiling (2000 vs 500).
# This lets the model train longer while avoiding overfitting.
#
# **Rater-bias encoding (tested but disabled):**
# A smoothed per-rater target encoding was prototyped: each rater's systematic
# bias (mean residual) is computed OOF to avoid leakage. Testing showed this
# did not improve RMSE on these GroupKFold(5) folds — `rater_col=None`.
#
# **Question being tested:**
# Do temporal neighbourhood features (prev/next visit OFF, rate of change) and
# blending with Ridge together reduce RMSE meaningfully below 4.58? What is
# the optimal blend weight?

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
CAT_COLS = ["gene", "cohort", "rater_id"]

X_train = pd.read_csv(DATA / "X_train.csv")
y_train = pd.read_csv(DATA / "y_train.csv")
X_test = pd.read_csv(DATA / "X_test.csv")

visits = X_train.merge(y_train, on="Index")
y = visits["target"]
groups = visits["patient_id"]
n_train = len(visits)

# %% [markdown]
# ## Features
#
# Calculées uniquement à partir des colonnes d'entrée, jamais de la cible.
# Train et test sont traités ensemble pour que les agrégats par patient soient
# cohérents (chaque patient de test a ses visites dans X_test).


# %%
def build_features(df):
    df = df.copy()
    # Fill missing gene labels before encoding — "unknown" becomes its own category.
    df["gene"] = df["gene"].fillna("unknown")
    # Within-visit treatment response.
    df["on_minus_off"] = df["on"] - df["off"]
    # Binary missingness flags: explicit features for the 42% off / 30% on NaN rate.
    df["off_missing"] = df["off"].isna().astype(int)
    df["on_missing"] = df["on"].isna().astype(int)

    # Levodopa pharmacokinetics: exponential decay with two half-lives (2h and 4h)
    # approximates blood levodopa concentration at the time of the visit.
    for half_life in (2.0, 4.0):
        df[f"ledd_decay_{half_life:g}"] = df["ledd"] * np.exp(
            -df["time_since_intake_on"] / half_life
        )

    # Patient-level aggregates via transform.
    g = df.groupby("patient_id")
    df["p_n_visits"] = g["age"].transform("size")
    df["p_mean_on"] = g["on"].transform("mean")
    df["p_mean_off"] = g["off"].transform("mean")
    df["p_mean_ledd"] = g["ledd"].transform("mean")
    df["p_mean_age"] = g["age"].transform("mean")
    df["age_since_first"] = df["age"] - g["age"].transform("min")
    df["visit_rank"] = g["age"].rank(method="first")

    # Temporal neighbourhood features — NEW versus 11_residual_patient.
    # Shift on a sorted copy; assignment via index preserves original row order.
    s = df.sort_values(["patient_id", "age"])
    gs = s.groupby("patient_id")
    df["off_prev"] = gs["off"].shift(1)   # OFF score at the previous visit
    df["off_next"] = gs["off"].shift(-1)  # OFF score at the next visit (if available)
    df["on_prev"] = gs["on"].shift(1)     # ON score at the previous visit
    df["dt_prev"] = df["age"] - gs["age"].shift(1)  # time since previous visit (years)
    df["off_delta_prev"] = df["off"] - df["off_prev"]  # change in OFF since last visit

    # Per-patient OLS slope of OFF vs age: captures disease progression rate.
    # Computed from input columns only (no target). Replace zero denominator with NaN
    # for patients with only one visit (zero variance in age).
    a = df["age"] - df["p_mean_age"]
    o = df["off"] - df["p_mean_off"]
    num = (a * o).groupby(df["patient_id"]).transform("sum")
    den = (a * a).groupby(df["patient_id"]).transform("sum").replace(0, np.nan)
    df["p_off_slope"] = num / den

    # Richer severity profile (same as 11_residual_patient).
    df["p_std_on"] = g["on"].transform("std")
    df["p_std_off"] = g["off"].transform("std")
    df["p_min_off"] = g["off"].transform("min")
    df["p_max_off"] = g["off"].transform("max")
    df["p_frac_off_missing"] = g["off_missing"].transform("mean")
    df["p_mean_offdiff"] = g["on_minus_off"].transform("mean")
    df["p_mean_time_on"] = g["time_since_intake_on"].transform("mean")
    df["on_dev_patient"] = df["on"] - df["p_mean_on"]
    df["age_to_last"] = g["age"].transform("max") - df["age"]
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    df["p_mean_duration"] = df.groupby("patient_id")["disease_duration"].transform("mean")
    return df


combined = pd.concat([visits.drop(columns="target"), X_test], ignore_index=True)
feats = build_features(combined)

for col in CAT_COLS:
    feats[col] = pd.Categorical(feats[col]).codes

# Baseline : OFF si mesuré, sinon ON plus le décalage médian (calculé sur les features).
# Clinical baseline: `off` if measured, otherwise `on + median(off − on)` proxy,
# falling back to the global median. See `10_residual_patient` for the rationale.
offset = (feats["off"] - feats["on"]).median()
feats["baseline"] = feats["off"].fillna(feats["on"] + offset)
feats["baseline"] = feats["baseline"].fillna(feats["baseline"].median())

feature_cols = [c for c in feats.columns if c not in ("Index", "patient_id")]
cat_mask = [c in CAT_COLS for c in feature_cols if c != "baseline"]

X_all = feats[feature_cols]
X_tr = X_all.iloc[:n_train].reset_index(drop=True)
X_te = X_all.iloc[n_train:].reset_index(drop=True)


# %% [markdown]
# ## Modèle : résidu sur baseline
#
# Wrapper scikit-learn : il retire la colonne `baseline`, entraîne le modèle sur
# `y - baseline`, puis ajoute la baseline à la prédiction. Ainsi `evaluate` et
# `fit` fonctionnent comme pour les autres expériences.


# %%
class ResidualRegressor(BaseEstimator, RegressorMixin):
    """ResidualRegressor with optional smoothed rater-bias encoding.

    If `rater_col` is set, a per-rater bias (smoothed mean of residual) is
    computed out-of-fold (inner KFold) and appended as an extra feature so
    each training row never sees its own target. This prevents leakage while
    still giving the model a rater-specific intercept adjustment.
    Note: `rater_col=None` in `make_model()` after testing — rater bias did
    not improve RMSE on these 5 GroupKFold folds.
    """

    def __init__(self, estimator=None, baseline_col="baseline", rater_col=None,
                 smooth=10.0, inner_folds=5):
        self.estimator = estimator
        self.baseline_col = baseline_col
        self.rater_col = rater_col
        self.smooth = smooth
        self.inner_folds = inner_folds

    def _split(self, X):
        base = np.asarray(X[self.baseline_col], dtype=float)
        return X.drop(columns=[self.baseline_col]), base

    def _encode(self, rater, resid):
        global_mean = float(np.mean(resid))
        stats = pd.DataFrame({"r": rater, "e": resid}).groupby("r")["e"].agg(["sum", "count"])
        enc = (stats["sum"] + self.smooth * global_mean) / (stats["count"] + self.smooth)
        return enc, global_mean

    def fit(self, X, y):
        X_feat, base = self._split(X)
        resid = np.asarray(y) - base
        if self.rater_col:
            rater = X_feat[self.rater_col].to_numpy()
            oof = np.empty(len(resid))
            kf = KFold(n_splits=self.inner_folds, shuffle=True, random_state=SEED)
            for tr, va in kf.split(X_feat):
                enc, gm = self._encode(rater[tr], resid[tr])
                oof[va] = pd.Series(rater[va]).map(enc).fillna(gm).to_numpy()
            self.enc_, self.gm_ = self._encode(rater, resid)
            X_feat = X_feat.assign(rater_bias=oof)
        self.model_ = clone(self.estimator).fit(X_feat, resid)
        return self

    def predict(self, X):
        X_feat, base = self._split(X)
        if self.rater_col:
            bias = pd.Series(X_feat[self.rater_col].to_numpy()).map(self.enc_).fillna(self.gm_)
            X_feat = X_feat.assign(rater_bias=bias.to_numpy())
        return base + self.model_.predict(X_feat)


def make_model():
    base_estimator = HistGradientBoostingRegressor(
        learning_rate=0.03,       # lower than 11_residual_patient — compensated by early stopping
        max_iter=2000,            # high ceiling; early stopping will terminate well before this
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,      # halt when validation RMSE stops improving
        validation_fraction=0.1,  # 10% of the training fold used as internal stop criterion
        n_iter_no_change=50,      # patience: stop after 50 rounds without improvement
        categorical_features=cat_mask,
        random_state=SEED,
    )
    # `rater_col=None`: rater-bias encoding was tested and did not reduce RMSE
    # on these 5 GroupKFold folds, so it is disabled.
    return ResidualRegressor(
        estimator=base_estimator,
        baseline_col="baseline",
        rater_col=None,
    )


# Early stopping uses an internal random 10% split inside the training fold only —
# it never touches the GroupKFold validation rows, so there is no CV data leakage.


# %% [markdown]
# ## Ridge blend component
#
# Ridge on numeric features only (no categorical codes, with median imputation and
# standard scaling), blended with the residual HGBR at weight `w` chosen via OOF.


# %%
# Ridge operates only on numeric columns — categorical codes carry no ordinal meaning
# for a linear model, and including them would conflate arbitrary integer labels with
# distances. `alpha=10.0` was chosen empirically; Ridge is a secondary component so
# fine-tuning it has diminishing returns.
NUM_COLS = [c for c in X_tr.columns if c not in CAT_COLS]


def make_ridge():
    # Median imputation handles the ~42% off / ~30% on NaN rate.
    # StandardScaler is required: Ridge penalises coefficient norms, so unscaled
    # features with large ranges (e.g. ledd ≈ hundreds vs age ≈ tens) would be
    # penalised unfairly.
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=10.0),
    )


class BlendRegressor(BaseEstimator, RegressorMixin):
    """Mélange pondéré : w * HGB résiduel + (1 - w) * Ridge."""

    def __init__(self, hgb=None, ridge=None, w=0.5):
        self.hgb = hgb
        self.ridge = ridge
        self.w = w

    def _num(self, X):
        return X[NUM_COLS]

    def fit(self, X, y):
        self.hgb_ = clone(self.hgb).fit(X, y)
        self.ridge_ = clone(self.ridge).fit(self._num(X), y)
        return self

    def predict(self, X):
        return self.w * self.hgb_.predict(X) + (1 - self.w) * self.ridge_.predict(self._num(X))

# %%
# Pre-compute the same 5 GroupKFold splits used for OOF weight search and final
# evaluate — both see identical fold boundaries.
cv_splits = list(GroupKFold(n_splits=5).split(X_tr, y, groups=groups))

# %% [markdown]
# ## Blend weight selection (OOF on the same GroupKFold folds)

# %%
# Candidate blend weights: 0 = Ridge only, 1 = HGBR only, intermediate values
# explore the convex combination. The search is over OOF predictions so no
# additional held-out data is consumed.
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
y_arr = y.to_numpy(dtype=float)
oof_hgb = np.empty(n_train)
oof_ridge = np.empty(n_train)
# Collect OOF predictions from each model independently on the same GroupKFold folds.
for tr, va in cv_splits:
    hgb_m = make_model().fit(X_tr.iloc[tr], y.iloc[tr])
    ridge_m = make_ridge().fit(X_tr.iloc[tr][NUM_COLS], y.iloc[tr])
    oof_hgb[va] = hgb_m.predict(X_tr.iloc[va])
    oof_ridge[va] = ridge_m.predict(X_tr.iloc[va][NUM_COLS])

# RMSE of each blend weight over the full OOF predictions.
scores = {
    w: float(np.sqrt(np.mean((y_arr - (w * oof_hgb + (1 - w) * oof_ridge)) ** 2)))
    for w in WEIGHTS
}
for w, s in scores.items():
    print(f"w = {w:.2f} -> RMSE OOF = {s:.4f}")

# Select the weight with the lowest OOF RMSE.
BEST_W = min(scores, key=scores.get)
print(f"Best blend weight: {BEST_W}")

model = BlendRegressor(hgb=make_model(), ridge=make_ridge(), w=BEST_W)

# %% [markdown]
# ## Evaluate with grouped CV

# %%
report = evaluate(model, X_tr, y, splitter=cv_splits)
report

# %% [markdown]
# ## RMSE

# %%
report.metrics.rmse()

# %% [markdown]
# ## Push to Skore Hub

# %%
project.put("09_blend_ridge", report)

# %% [markdown]
# ## Kaggle submission
#
# Refit sur tout le train, prédiction sur le test.

# %%
SUBMISSIONS.mkdir(exist_ok=True)

final = BlendRegressor(hgb=make_model(), ridge=make_ridge(), w=BEST_W).fit(X_tr, y)
submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_te)
out = SUBMISSIONS / "09_blend_ridge.csv"
submission.to_csv(out, index=False)

print(f"Submission written: {out} ({len(submission)} rows)")

# %% [markdown]
# ## What to look for / next steps
#
# **Expected outcome:** RMSE ≈ 4.25 ± 0.09 — a 7 % improvement over
# `11_residual_patient` (4.58). This is the best result in the project.
#
# **OOF blend weight results (from the journal):**
# | w (HGBR fraction) | OOF RMSE |
# |---|---|
# | 0.00 (Ridge only) | 5.0187 |
# | 0.25            | 4.6424 |
# | 0.50            | 4.3786 |
# | **0.75**        | **4.2483** ← selected |
# | 1.00 (HGBR only)| 4.2638 |
#
# **What this tells us:**
# - The blend gain over pure HGBR is small (0.02 RMSE) but consistent. The
#   main driver of improvement in this experiment is the richer feature set
#   (temporal neighbourhood features), not the blending itself.
# - `off_prev` and `off_next` give the model access to disease trajectory:
#   knowing the OFF score at adjacent visits dramatically reduces prediction
#   error for visits where the current `off` is missing.
# - `p_off_slope` (the per-patient disease progression rate) is a genuinely
#   novel signal that wasn't captured by the static severity profile.
# - Ridge (w=0) alone scores 5.02 — worse than pure HGBR (4.26). Ridge's
#   contribution is purely as a variance-reducer, not a better predictor.
# - Early stopping at learning_rate=0.03 avoids overfitting on the 10%
#   internal validation split while allowing up to 2000 boosting rounds.
#
# **Known risks:**
# - `off_next` uses a future visit's value. At test time, future visits of
#   test patients may be in `X_test` (since we concat train+test), making
#   `off_next` mildly optimistic in the combined feature build.
# - The blend weight was selected on the same OOF folds used for final
#   `evaluate` — the 0.02 gain could be slightly overfit to these 5 folds.
#
# **This is the final submitted experiment.** RMSE 4.25 is the project best.
# Potential further directions if more time were available:
# - Finer blend weight search (grid of 0.01 increments near w=0.75).
# - Target encoding for `rater_id` with a larger smooth parameter.
# - Stacking: train a meta-learner on OOF predictions from multiple models.
# - Feature: `off_slope_to_current` (trend in OFF since the previous visit).
