# ADR 0004: Data drift monitoring

- **Status:** Accepted
- **Date:** 2026-11-20
- **Builds on:** [ADR 0001](0001-rul-cap-and-sensor-selection.md) (kept sensors) and the serving
  stack (MLflow registry, FastAPI, PostgreSQL)

## Context

The model was trained and evaluated on FD001. In service, it will silently give worse
predictions if the incoming data stops resembling that data, for example through new operating
conditions, a firmware change or a batch of miscalibrated sensors. We need a signal that the
inputs have shifted, computed from what the service actually stores. The dashboard must be able
to show it through the API, without access to the database or to labels.

## Decision

**What is measured:**

- the **Population Stability Index (PSI)** per column, for the 3 operating settings and the 14
  kept sensors (17 columns);
- **current data:** each unit's **last 30 stored readings**, pooled across the fleet;
- **reference:** decile bins (10 bins at the training deciles) and their shares, computed from
  **all 20,631 training readings**;
- PSI = Σ (current − reference) · ln(current / reference), with empty bins floored at
  ε = 10⁻⁴.

**Thresholds:** stable < **0.1** ≤ moderate < **0.25** ≤ significant.

**Where the reference lives:** it is stored as a JSON artifact **inside the registered model
version**. It belongs to exactly that model, is readable in the MLflow UI, and models without
one still load (the endpoint then answers 503).

**Where it is exposed:** `GET /monitoring/drift?last_n_cycles=30`, and the dashboard's
"Data drift" section.

**Implementation:** our own, in `predmaint.monitoring.drift` (about 150 lines), rather than a
monitoring library (see Alternatives).

### Validation with injected faults

Faults were injected with the replay simulator
(`--inject-drift COLUMN:OFFSET --drift-units RANGE`). Setup:

- the 100 FD001 test engines;
- each unit's last 30 readings (3,000 rows);
- champion model v2.

| Scenario | `s_11` PSI | Other columns |
|---|---|---|
| No injection | 0.072, stable | all stable; max **0.077** (`s_4`) |
| `s_11` +0.5 on **all 100** units | **4.582, significant** | unchanged |
| `s_11` +0.5 on units **91–100** only | **0.050, stable** (0.050018) | unchanged; max 0.077 (`s_4`) |

- The last row was read from the running API (`GET /monitoring/drift?last_n_cycles=30`) while
  the database held that scenario.
- The no-injection `s_11` value, 0.072, was computed with the same code and reference outside
  the API.
- The 10-unit fault does not just go unnoticed: it **lowers** `s_11`'s PSI (0.072 → 0.050).
  The shifted readings, 10% of the pool, land in the reference's open-ended top bin, partly
  filling it rather than creating a new peak. Even a +2.0 offset on those 10 units stays near
  0.065 in our measurements. The blind spot is structural, not a matter of the threshold.

## Alternatives considered

- **Evidently** (open-source Python library for data and model monitoring). It offers drift
  presets with several statistical tests (PSI among them), per-column reports and HTML
  dashboards. Not chosen for now:
  - **Small need:** we need one statistic for 17 columns. Our implementation is short, fully
    unit-tested (including a hand-calculated PSI), and its binning is easy to explain.
  - **Version-locked reference:** we store the reference *bins*, not the training data, as a
    small JSON file inside each model version. That keeps the API image free of training data
    and makes every result reproducible for exactly that model.
  - **Fits the existing design:** the dashboard talks only to the API, and the API returns
    plain JSON that the dashboard plots itself.
  - **Fewer dependencies:** no additional dependency in the serving image.

  Evidently would be worth adopting if we needed many tests, data-quality checks or
  ready-made reports.
- **Per-unit PSI** (each unit's window against the reference). Rejected as the main signal:
  healthy units already score a median PSI of about 3.4 on the sensors, because a single unit
  sits at one life stage (see Consequences). In the 91–100 test, the faulty units ranked
  anywhere from 2nd to 50th of 100.
- **Other statistics** (Kolmogorov–Smirnov, Wasserstein distance). Not evaluated. KS p-values
  become "significant" for tiny shifts at 3,000 rows. PSI's fixed bins give a scale-free number
  with widely used thresholds.
- **A reference from training engines' last 30 cycles only.** It would match the window, but
  live units are at mixed life stages, so it would flag healthy young engines instead.

## Consequences

- **Catches fleet-wide shifts, not single-unit faults.** A firmware update, a new sensor batch
  or a misconfigured feed affecting much of the fleet is detected (+0.5 on all units gives PSI
  4.58). A fault on a few units is not: 10 of 100 units at +0.5 gives 0.050, "stable". A
  stable status does **not** mean every unit is healthy.
- **Degradation also shifts sensors.** The reference spans whole engine lives, so a fleet
  concentrated at one life stage shows sensor drift with no fault. On training engines alone,
  the first 30 cycles score 0.9–1.2 and the last 30 cycles 3.8–4.3. A mixed-age fleet (the
  100 test units) scores 0.02–0.08.
  - Sensor PSI is a prompt to look, not proof of a problem.
  - Drift in the **operating settings** is the clearer signal, because they do not degrade.
  - This caveat is returned by the API and shown in the dashboard.
- **The reference changes with the model.** Rebuilding the model (a new registered version)
  rebuilds the reference. Models registered before this ADR have no reference, and the
  endpoint tells the caller to rebuild.
- **No alerting yet.** Drift is computed on request; nothing notifies anyone.
- **Future work:**
  - **Per-unit monitoring that controls for life stage,** e.g. compare each unit's
    *early-life* readings (first cycles after it enters service) with the training engines'
    early-life distribution. Alternatively, compare each unit with fleet peers at a similar
    predicted RUL, or monitor the model's residual patterns.
  - **Alerting on drift:** a scheduled check that records PSI over time and alerts on a
    sustained "significant" status, especially for the operating settings.
