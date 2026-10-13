# 06_dataops

## Question / hypothesis

Does using a skrub DataOps graph (GroupKFold baked in) produce the same RMSE as
`05_tabular_pipeline` while making the split structurally impossible to accidentally
decouple from the data?

## Motivation

- **Sourcing strategy:** user (Day 2 guide, step 6)
- **Source(s):**
  - Day 2 guide §6: "Until now you have juggled separate objects: a DataFrame `visits`,
    a list `cv_splits`, a pipeline `model`. Easy to fit on the wrong columns or forget
    `groups` after a copy-paste. DataOps is skrub's way to write the recipe once, as a
    graph, so features, target, and the grouped split live on the same object."
- **Why this matters:** In `05_tabular_pipeline`, the `cv_splits` list is a separate
  variable that could go stale if the data is filtered or reordered. Baking
  `GroupKFold` and `split_kwargs={"groups": groups}` into `mark_as_X` makes the split
  a property of the graph node — `skore.evaluate(pred)` reads it automatically.

## Method

- **Files touched:** `experiments/06_dataops.py` (self-contained script)
- **Change versus 05_tabular_pipeline:** same model (TableVectorizer + HGBR), different
  plumbing. `skrub.var("visits", visits)` names the input; feature and target nodes are
  marked via `.skb.mark_as_X(...)` and `.skb.mark_as_y()`; steps appended via
  `.skb.apply(...)`. `skore.evaluate(pred)` takes no `splitter=` argument.
  Submission uses `pred.skb.make_learner()` which produces a `SkrubLearner` fitted and
  predicted with an environment dict `{"visits": ...}`.
- **Columns dropped from X:** `Index`, `patient_id`, `target` — same as §5.
- **Cross-validation:** GroupKFold(n_splits=5), groups=patient_id — baked into the graph.
- **Hub key:** `06_dataops`.
- **Out of scope:** hyperparameter tuning, additional feature engineering.

## Risks / things that could invalidate the result

- If skrub drops `Index`/`patient_id` inside the graph differently from a plain
  `.drop(columns=[...])`, the feature set could differ from §5 — worth checking the
  RMSE match.
- `SkrubLearner.predict({"visits": X_test})` requires `X_test` to have all columns
  the graph references (including `patient_id` for the `groups` node), even though
  `patient_id` is not in X. This is expected — the graph resolves `groups` lazily.

## Status

- **State:** done
- **Approved by user on:** 2025-07-11
- **Headline result:** RMSE 7.53 ± 0.15 (GroupKFold 5-fold, DataOps graph) — matches 05_tabular_pipeline exactly
- **Implication for next iteration:** Architecture validated. The DataOps graph is the preferred form for future experiments — groups are structurally bound to the data. This completes Day 2. Next steps: temporal features, drug-timing features, or patient-level aggregations (Day 3 direction).
