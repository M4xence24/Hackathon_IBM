# 11_residual_patient

## Question / hypothesis

Does the richer patient severity profile (std of on/off, min/max off, fraction missing,
visit-level deviations from patient mean, disease duration, visit rank) improve on the
`10_residual_patient` residual model?

## Motivation

- **Sourcing strategy:** user (written outside the Bob iteration loop; direct revision
  of `10_test.py`)
- **Source(s):** `experiments/11_test.py`
- **Why this matters:** `10_test.py` used per-patient means but left out within-patient
  variability (std of off/on) and disease progression proxies (disease duration,
  `on_dev_patient`, `age_to_last`). A patient whose `off` score is highly variable is
  clinically different from one with a stable score at the same mean level. Adding these
  features gives the model a richer picture of disease severity trajectory.

## Method

- **Files touched:** `experiments/11_test.py` (flat script; same architecture as
  `10_test.py` with expanded `build_features`)
- **Change versus 10_test.py:** identical architecture (`ResidualRegressor` + same HGBR
  params), but `build_features` adds:
  - `p_std_on`, `p_std_off` — within-patient variability of ON/OFF scores
  - `p_min_off`, `p_max_off` — severity range per patient
  - `p_frac_off_missing` — fraction of visits where `off` was not measured
  - `p_mean_offdiff` — mean of `on − off` per patient
  - `p_mean_time_on` — mean `time_since_intake_on` per patient
  - `on_dev_patient` — deviation of visit `on` from patient mean (`on − p_mean_on`)
  - `age_to_last` — time remaining until patient's last observed visit
  - `disease_duration` — `age − age_at_diagnosis`
  - `p_mean_duration` — mean disease duration per patient
- **HGBR params:** identical to `10_test.py`: `learning_rate=0.05`, `max_iter=500`,
  `max_leaf_nodes=31`, `min_samples_leaf=40`, `l2_regularization=1.0`, `random_state=0`.
- **Cross-validation:** `GroupKFold(n_splits=5)`, `groups=patient_id`.
- **Hub key:** `07_residual_patient` (overwrites `10_test.py`'s push; this is the most
  recent version on hub).
- **Submission:** `submissions/11_residual_patient.csv`.

## Risks / things that could invalidate the result

- Same train+test feature leakage risk as `10_test.py` (patient aggregates built over
  the full concat of train + test).
- `age_to_last` looks forward in time (uses `max(age)` per patient). This is not
  temporal leakage for a cross-sectional model (we do not predict future targets), but
  it encodes information about visits that come after the current one — mildly optimistic
  in a true deployment setting where future visits are unknown.
- Hub key `07_residual_patient` is overwritten by the last run; the earlier
  `improve_rmse_skore.py` and `10_test.py` results are no longer accessible on hub.

## Status

- **State:** done
- **Approved by user on:** 2025-07-11 (run by user outside Bob iteration loop)
- **Headline result:** RMSE 4.58 ± 0.11 (GroupKFold 5-fold, ResidualRegressor + full patient severity profile) — **new best**; 7 % below 10_residual_patient (4.92)
- **Implication for next iteration:** Full patient severity profile (std/min/max off, disease duration, visit rank, on_dev_patient) gives a consistent additional gain. `step6_seeds.py` was run as a follow-up diagnostic to measure seed noise across feature subsets. Next direction: submit `11_residual_patient.csv` to the leaderboard.
