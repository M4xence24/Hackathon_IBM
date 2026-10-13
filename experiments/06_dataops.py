# %% [markdown]
# # Experiment 06 — skrub DataOps: groups baked into the graph
#
# **Date:** 2025-07-11
# **Goal:** Rewrite the §5 tabular_pipeline model as a skrub DataOps graph so
#           patient-grouped CV is structurally attached to the data, not a
#           separate list that can drift.
# **Hub key:** `06_dataops`

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
data = skrub.var("visits", visits)

groups = data["patient_id"]
y_op = data["target"].skb.mark_as_y()

X_op = data.drop(columns=["Index", "patient_id", "target"], errors="ignore").skb.mark_as_X(
    cv=GroupKFold(n_splits=5),
    split_kwargs={"groups": groups},
)

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

learner = pred.skb.make_learner()
learner.fit({"visits": visits})
pred_test = learner.predict({"visits": X_test})

submission = X_test[["Index"]].copy()
submission["target"] = pred_test
submission.to_csv(submissions_dir / "06_dataops.csv", index=False)

print(f"Submission written: submissions/06_dataops.csv ({len(submission)} rows)")
