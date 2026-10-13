# Day 2 Remaining Steps Plan

## Overview

Day 2 sections §1–§5 are complete. Two sections remain:

- **§6** — skrub DataOps: bake GroupKFold + patient groups directly into the graph so
  the split cannot drift from the data. Same model as §5 (TableVectorizer + HGBR),
  same score expected, different plumbing.
- **§7** — Fit the chosen best model on all training rows and write a final Kaggle CSV.
  The guide says: pick the winner from steps 3–6, `put` a new hub report, refit on all
  training visits, predict on `X_test`, write `submissions/<key>.csv`, upload with the
  hub URL in the description.

Both use hub workspace `barthon`, project `bobathon-esilv`. Login pattern is identical
to all previous scripts.

**Constraint**: `project.put()` only accepts `EstimatorReport` or
`CrossValidationReport`. Evaluate each model individually before pushing.

---

## Sub-Task 1 — skrub DataOps: groups baked into the graph (Day 2 §6)

### Intent
Rewrite the §5 model as a skrub DataOps graph. The graph names the input table with
`skrub.var("visits", visits)`, marks the feature node with `.skb.mark_as_X(cv=GroupKFold(5), split_kwargs={"groups": groups})` so the patient-grouped CV is attached to the data,
marks the target node, appends `TableVectorizer` then HGBR via `.skb.apply(...)`, and
calls `skore.evaluate(pred)` with **no** `splitter=` argument — the graph supplies it.

The score should be identical (or very close) to `05_tabular_pipeline` because the
model and split are the same; the difference is architectural: groups can no longer
drift from `X`.

### Expected Outcomes
- `journal/06_dataops.md` design note filed and status = done.
- `experiments/06_dataops.py` runs to completion.
- Hub key `06_dataops` appears in the project.
- `submissions/06_dataops.csv` written using `SkrubLearner.fit/predict` with env dict.
- JOURNAL.md History row added.

### Todo List
- [ ] Write `journal/06_dataops.md` (design note, state: approved).
- [ ] Write `experiments/06_dataops.py`:
  - Load `X_train`, `y_train`, `X_test`; build `visits`, `y`.
  - `data = skrub.var("visits", visits)`
  - `groups = data["patient_id"]`
  - `X_op = data.drop("target", axis=1).skb.mark_as_X(cv=GroupKFold(n_splits=5), split_kwargs={"groups": groups})`
  - Drop `patient_id` from `X_op` before vectorizing to avoid leakage.
  - `y_op = data["target"].skb.mark_as_y()`
  - `pred = X_op.skb.apply(skrub.TableVectorizer()).skb.apply(HistGradientBoostingRegressor(random_state=0), y=y_op)`
  - `report = skore.evaluate(pred)` — no `splitter=`, graph supplies it.
  - `project.put("06_dataops", report)`
  - Submission: `learner = pred.skb.make_learner()`, `learner.fit({"visits": visits})`,
    `pred_test = learner.predict({"visits": X_test})`, write `submissions/06_dataops.csv`.
- [ ] Run the script; capture RMSE.
- [ ] Update `journal/06_dataops.md` Status block with headline result.
- [ ] Add History row to `journal/JOURNAL.md`.

### Relevant Context
- Day 2 guide §6: `docs/DAY2.md` lines 220–263.
- `05_tabular_pipeline.py` for the column drop pattern and hub login.
- `skrub.var`, `skb.mark_as_X`, `skb.mark_as_y`, `skb.apply`, `skb.make_learner` — DataOps API.
- The guide warns: `X_test` has no `target` column; `learner.predict({"visits": X_test})` handles this.

### Status
[x] done — RMSE 7.53 ± 0.15; hub https://skore.probabl.ai/barthon/bobathon-esilv/cross-validations/67956; submissions/06_dataops.csv (11 013 rows)

---

## Sub-Task 2 — Final submission from best model (Day 2 §7)

### Intent
Pick the best model by grouped-CV RMSE across experiments 03–06, `put` a new hub
report if needed, refit on **all** training rows, predict on `X_test`, write the
Kaggle CSV named after the hub key, and confirm it is ready to upload.

The best model so far is `05_tabular_pipeline` (RMSE 7.53 ± 0.14). After §6 runs
we compare — if `06_dataops` matches, it is the preferred architecture (groups
cannot drift). Either way the submission logic is the same.

**Note**: §5 and §6 already write their own CSVs as part of their scripts. This
sub-task is a **verification step** — confirm the correct CSV exists, is named after
the winning hub key, and that the hub URL is ready to paste into Kaggle.

### Expected Outcomes
- The winning hub key and its URL are identified.
- `submissions/<winning_key>.csv` exists with header `Index,target` and 11 013 rows.
- JOURNAL.md Status updated to reflect the final submission key.

### Todo List
- [ ] After §6 runs, compare RMSE: `05_tabular_pipeline` (7.53) vs `06_dataops`.
- [ ] Confirm the winning submission CSV exists in `submissions/`.
- [ ] Print the winning hub URL (already shown when `project.put` ran).
- [ ] Update JOURNAL.md Status with the final submission key and URL.

### Relevant Context
- Day 2 guide §7: `docs/DAY2.md` lines 266–293.
- Kaggle submission rules: CSV header `Index,target`; URL in submission description.
- Existing CSVs: `submissions/04_hgbr.csv`, `submissions/05_tabular_pipeline.csv`.

### Status
[x] done — winner: 06_dataops (RMSE 7.53); submissions/06_dataops.csv verified (11 013 rows, Index+target); JOURNAL.md updated

---

## Ordering

```
06_dataops  →  07_final_submission_check
```

Sub-Task 2 is a verification step that depends on §6's RMSE being known.
