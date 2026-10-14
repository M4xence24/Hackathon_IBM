# %% [markdown]
# # EDA: Parkinson's Disease Multi-Cohort Visits
#
# Exploratory data analysis of the Parkinson's true OFF MDS-UPDRS motor score
# dataset, run before designing a model.
#
# - **Raw data** is read-only — `X_train.csv`, `y_train.csv` in `data/`.
#   This file never cleans or modifies the raw data.
# - **Outputs** go under `EDA_DIR` (the repo's `data/`): an
#   `eda_visits.html` report and `eda.md` narrative.

# %%
import json
from pathlib import Path

import pandas as pd
import skrub

# This file lives in data/, so parents[1] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# EDA outputs always land here (created if missing); the raw data lives here too.
EDA_DIR = PROJECT_ROOT / "data"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load the raw data
#
# Merge X_train and y_train on Index to get the full visits table with target.

# %%
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
RAW = X_train.merge(y_train, on="Index")
RAW.shape  # noqa: B018

# %% [markdown]
# ## Table overview
#
# Per-table report saved to `data/eda_visits.html`, plus a compact per-column
# summary: dtype, fraction missing, and number of unique values.

# %%
report = skrub.TableReport(RAW, title="visits", verbose=0)
report.write_html(EDA_DIR / "eda_visits.html")

summary = json.loads(report.json())
n_rows = summary.get("n_rows")
overview = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("n_unique"),
    }
    for col in summary.get("columns", [])
]
{"n_rows": n_rows, "n_columns": len(overview), "columns": overview}  # noqa: B018

# %% [markdown]
# ## Target
#
# Distribution of `target` (true OFF MDS-UPDRS motor score) — spread and skew
# shape the metric choice and whether CV should stratify.

# %%
TARGET = "target"
next((col for col in summary.get("columns", []) if col.get("name") == TARGET), None)  # noqa: B018

# %% [markdown]
# ## Structure signals
#
# Datetime columns (pointing to time-based validation) and high-cardinality
# id/group columns (pointing to grouped validation to avoid patient leakage).

# %%
datetime_cols = [
    col.get("name")
    for col in summary.get("columns", [])
    if "date" in str(col.get("dtype", "")).lower()
]
unique_ratio = sorted(
    (
        {
            "column": col.get("name"),
            "unique_ratio": (col.get("n_unique") or 0) / n_rows if n_rows else None,
        }
        for col in summary.get("columns", [])
    ),
    key=lambda r: (r["unique_ratio"] is not None, r["unique_ratio"]),
    reverse=True,
)
{"datetime_cols": datetime_cols, "top_unique_ratio": unique_ratio[:10]}  # noqa: B018

# %% [markdown]
# ## Associations
#
# Strongest pairwise column associations. Strong feature↔target links are
# candidate predictors; an implausibly perfect link is a possible leakage flag.

# %%
assoc = skrub.column_associations(RAW)
rows = assoc.to_dicts() if hasattr(assoc, "to_dicts") else assoc.to_dict(orient="records")
target_links = [
    row
    for row in rows
    if row["left_column_name"] == TARGET or row["right_column_name"] == TARGET
]
{"with_target": target_links[:15], "strongest": rows[:10]}  # noqa: B018

# %% [markdown]
# ## Summary
#
# The findings and their modelling implications are written up in
# `data/eda.md`.
