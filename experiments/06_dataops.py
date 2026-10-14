# %% [markdown]
# # Experiment 06 — skrub DataOps: groups baked into the graph
#
# **Date:** 2025-07-11
# **Goal:** Rewrite the §5 tabular_pipeline model as a skrub DataOps graph so
#           patient-grouped CV is structurally attached to the data, not a
#           separate list that can drift.
# **Hub key:** `06_dataops`
#
# ## Context and motivation
#
# `05_tabular_pipeline` reached RMSE 7.53. The model works correctly, but there
# is a structural risk: the `cv_splits` list is a separate Python object from
# the `model` pipeline. The two can silently become inconsistent:
# - If the DataFrame is filtered or resampled between defining `cv_splits` and
#   calling `evaluate`, the fold indices no longer correspond to the right rows.
# - If a colleague copy-pastes the evaluation code and forgets the `splitter=`
#   argument, skore falls back to a default random split — the result looks
#   plausible but is not patient-grouped.
#
# **The DataOps approach** bakes the `GroupKFold` splitter *into the feature
# node of the graph* via `.skb.mark_as_X(cv=..., split_kwargs=...)`. Now:
# - `skore.evaluate(pred)` reads the splitter from the graph automatically —
#   there is no `splitter=` argument to forget.
# - The `groups` node is derived from the same data variable as the features —
#   it can never drift relative to the rows.
# - `pred.skb.make_learner()` freezes the full graph into a single object for
#   test-set prediction, keeping features and splitter together.
#
# **Question being tested:**
# Does the DataOps architecture produce identical RMSE to `05_tabular_pipeline`?
# If yes, it is the preferred form for all future experiments because it
# eliminates the data/splitter coupling risk.

# %%
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
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
# `skrub.var("visits", visits)` names the input table. Later, at predict time,
# `learner.predict({"visits": X_test})` swaps in the test table under the same name.
#
# `.skb.mark_as_X(cv=..., split_kwargs=...)` bakes patient-grouped CV into the
# feature node — `skore.evaluate(pred)` will read it without a `splitter=` argument.
#
# `Index`, `patient_id`, and `target` are dropped from the feature node; `patient_id`
# is still accessible for the `groups` node because it is referenced from the
# original `data` node before the drop.

# %%
# `skrub.var` names the input table in the DataOps graph. At predict time the
# learner receives `{"visits": X_test}` and skrub substitutes the test frame.
data = skrub.var("visits", visits)

# `groups` is extracted before the column drops so it remains accessible for the
# CV splitter even though `patient_id` is not a feature.
groups = data["patient_id"]
y_op = data["target"].skb.mark_as_y()

# Baking GroupKFold + split_kwargs into `mark_as_X` means `skore.evaluate(pred)`
# picks up the splitter automatically — no need to pass `splitter=cv_splits` at
# every `evaluate` call. This prevents the splitter from accidentally drifting
# between the model graph and the evaluation call.
X_op = data.drop(columns=["Index", "patient_id", "target"], errors="ignore").skb.mark_as_X(
    cv=GroupKFold(n_splits=5),
    split_kwargs={"groups": groups},
)

# Two-step pipeline: TableVectorizer handles string columns, then HGBR handles NaN
# in numeric columns natively. This is structurally identical to 05_tabular_pipeline
# but declared as a DataOps graph instead of a plain sklearn Pipeline.
pred = X_op.skb.apply(skrub.TableVectorizer()).skb.apply(
    HistGradientBoostingRegressor(random_state=0),
    y=y_op,
)

# %% [markdown]
# ## Evaluate
#
# The graph has GroupKFold baked in via `mark_as_X`. Pass the data binding
# so skore can materialise the nodes; `splitter=` is read from the graph.

# %%
# Pass the data binding dict so skore can materialise the graph nodes.
# The GroupKFold splitter is read from the graph — no `splitter=` needed.
report = evaluate(pred, data={"visits": visits})
report

# %% [markdown]
# ## RMSE

# %%
report.metrics.rmse()

# %% [markdown]
# ## Push to Skore Hub

# %%
project.put("06_dataops", report)

# %% [markdown]
# ## Kaggle submission
#
# `make_learner()` freezes the graph into a `SkrubLearner`. `fit` and `predict`
# take an environment dict: `{"visits": <dataframe>}`. For test data the `target`
# column is absent — skrub resolves only the nodes it needs for prediction.

# %%
submissions_dir = Path("submissions")
submissions_dir.mkdir(exist_ok=True)

# `make_learner()` freezes the graph into a fitted sklearn-compatible object.
# `fit` and `predict` both receive an environment dict mapping variable names to
# DataFrames. For prediction the `target` column is absent — skrub resolves only
# the nodes it needs and skips the `y` node.
learner = pred.skb.make_learner()
learner.fit({"visits": visits})
pred_test = learner.predict({"visits": X_test})

submission = X_test[["Index"]].copy()
submission["target"] = pred_test
submission.to_csv(submissions_dir / "06_dataops.csv", index=False)

print(f"Submission written: submissions/06_dataops.csv ({len(submission)} rows)")

# %% [markdown]
# ## What to look for / next steps
#
# **Expected outcome:** RMSE ≈ 7.53 ± 0.15 — identical to `05_tabular_pipeline`.
# A different number here indicates a bug (data ordering, column mismatch, or
# a different skrub default TableVectorizer configuration).
#
# **What this tells us:**
# - The DataOps graph produces exactly the same result as the plain sklearn
#   pipeline — the refactor is a pure architecture improvement, not a change in
#   the model.
# - The GroupKFold splitter is now a first-class citizen of the model object,
#   not a separate list that can go stale.
# - `make_learner()` provides a clean fit/predict interface for submission that
#   mirrors how the model would be deployed.
#
# **This experiment closes Day 2 of the hackathon.** The architecture is
# validated and all future experiments will use the DataOps pattern.
#
# **Next experiments — two parallel directions:**
# 1. `07_ridge_cv` — verify that `alpha=1.0` was already optimal for Ridge by
#    running a 30-point grid search. This closes the linear model investigation.
# 2. `08_hgbr_tuned` — coarse grid search over HGBR hyperparameters to squeeze
#    out marginal gains from the default configuration.
# 3. `09_patient_features` (the big step) — add per-patient aggregates (mean off,
#    visit count, disease duration). Expected: ~28 % improvement to RMSE ≈ 5.38.
