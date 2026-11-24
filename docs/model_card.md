# Model card: `rul-lgbm-fd001`

A model that estimates how many operating cycles a simulated turbofan engine has left (remaining
useful life, RUL), with a 90% prediction interval. **It is a portfolio project trained on
simulated data. It is not certified for, and must not be used for, real maintenance or
airworthiness decisions.**

## Model details

| | |
|---|---|
| Type | LightGBM regressor (gradient-boosted trees), 500 trees, untuned defaults ([gbdt.py](../src/predmaint/models/gbdt.py)) |
| Input | One engine's history: cycle and 14 of the 21 sensors; any length from 1 cycle |
| Features | 98: the 14 current values, plus rolling mean, std and slope over 10 and 30 cycles. They use only past and current cycles of the same engine |
| Training target | RUL capped at 125 cycles ([ADR 0001](adr/0001-rul-cap-and-sensor-selection.md)) |
| Output | RUL (clipped at 0), lower and upper bound of a 90% interval |
| Intervals | Mondrian split conformal, one width per predicted-RUL bin ([ADR 0003](adr/0003-prediction-intervals.md)) |
| Chosen over | Linear regression, 1D-CNN ([ADR 0002](adr/0002-model-choice.md)) |
| Registry | MLflow `rul-lgbm-fd001`, alias `champion`. Version numbers are local to each MLflow store |
| Evaluated at | commit `30cf99a`, 2026-11-17 |
| Licence | The repository has no licence file yet. The dataset states no licence ([dataset.md](dataset.md)) |

## Intended use

- Demonstrating an end-to-end RUL system on the C-MAPSS FD001 benchmark.
- Comparing RUL methods under the same [evaluation protocol](evaluation.md).
- Ranking FD001-like engines by risk, as decision support for a person.

**Out of scope:**

- Real engines, or any safety-critical or airworthiness decision.
- Automatic maintenance actions with no person in the loop.
- FD002–FD004 or other equipment: FD001 has one operating condition and one fault mode (HPC
  degradation).
- Exact RUL far from failure: above about 125 cycles the model only says "far from failure".
- Detecting sudden failures or faulty sensors.

## Training data

- NASA C-MAPSS FD001, simulated ([dataset.md](dataset.md)): 100 engines run to failure, 20,631
  cycles, lifetimes of 128–362 cycles.
- 7 constant or near-constant sensors and the 3 operating settings are dropped. The cap and the
  drop list were chosen on all 100 training engines (ADR 0001).

## Evaluation data and protocol

- **Development:** 5-fold cross-validation split by engine, on the training engines only.
- **Final test:** the official FD001 test set, 100 engines cut off before failure. One prediction
  per engine at its last cycle, against the uncapped true RUL. Run once, after the model was
  frozen.

**What was seen of the test set before that run.** No model was fitted, tuned or scored on it.
However, summary statistics were inspected while documenting the dataset: row and engine counts,
trajectory lengths, and the range of the true RUL (7–145). That range informed one protocol choice,
scoring validation rows with true RUL ≤ 150. Test sensor data was used later, after the final
run, for the replay and the drift validation.

## Metrics

Final test, 95% bootstrap intervals over engines (10,000 resamples;
[full report](../reports/final_test_fd001.md)):

| Model | RMSE (cycles) | MAE (cycles) | PHM08 score (sum) |
|---|---|---|---|
| **LightGBM** | **14.59** [12.46, 16.63] | **11.07** [9.21, 12.99] | **314.6** [227.3, 412.8] |
| Linear regression | 21.90 [19.10, 24.64] | 17.61 [15.09, 20.19] | 1314.0 [680.1, 2315.3] |
| Mean RUL | 43.07 [38.49, 47.38] | 35.90 [31.18, 40.58] | 33629 [20689, 48145] |

- LightGBM minus linear RMSE, paired bootstrap: **−7.31 [−10.20, −4.57]**.
- **Error direction:** 62 of 100 engines are predicted late (more life than remains), 13 of them by
  more than 20 cycles; 11 are early by more than 20. Mean error +2.05 cycles.
- Cross-validation RMSE was 16.51 ± 1.06 (MAE 12.36 ± 0.90; std across folds). It scores many rows
  per engine, so it is not directly comparable with the test number.

## Prediction intervals and alerting

**Coverage on the test set** (target 90%): **93 of 100** engines (95% Wilson interval
86.3%–96.6%), mean width 48.9 cycles.

| Predicted RUL | Covered | 95% Wilson interval | Mean width |
|---|---|---|---|
| 0–30 | 19 of 23 (83%) | 63%–93% | 12.7 |
| 30–60 | 10 of 12 (83%) | 55%–95% | 31.9 |
| 60–90 | 11 of 12 (92%) | 65%–99% | 57.6 |
| 90+ | 53 of 53 (100%) | 93%–100% | 66.4 |

- Of the 7 misses, 5 had more life left than the interval said (safe). **2 had less** (engines 37
  and 41), each by under one cycle.
- The bins are small: these numbers neither prove nor rule out under-coverage near failure.

**Alerts** use the interval's lower bound: critical below 20 cycles, warning below 50.

| True RUL | Engines | Critical | Warning | No alert |
|---|---|---|---|---|
| ≤ 20 | 16 | 16 | 0 | 0 |
| ≤ 30 | 25 | 21 | 4 | 0 |
| ≤ 50 | 33 | 21 | 12 | 0 |

- All 21 critical alerts had a true RUL of 7–29. The 18 warnings range from 21 to 114, so a warning
  alone does not mean failure is near.
- **This check is descriptive, not a validation.** The thresholds were set after the final test
  run, on round numbers rather than maintenance costs, and 16 of 16 has a 95% Wilson interval of
  81%–100%.

## Known limitations

- **Late errors in mid-life** ([error analysis](error_analysis.md), cross-validation): at true RUL
  50–100 the bias is about +9 cycles, and 26–27% of rows are late by more than 20.
- **Fast-degrading engines are predicted late:** at true RUL 50–100, bias +18 to +20 with 30–100
  cycles of history.
- **The cap biases high RUL low:** no cross-validated prediction exceeds 133.6 cycles (ADR 0001).
- **Some large errors are unexplained**, in cross-validation and on the test set.
- **Intervals are approximate** (ADR 0003): coverage is per predicted bin, not per engine or per
  true RUL. It assumes new engines resemble the training engines, and cycles of one engine are
  correlated.
- **Untuned and small:** LightGBM and the CNN used default settings, with 100 training engines (ADR
  0002). Cap and model were selected on the cross-validation that reports their scores.
- **Drift monitoring** (ADR 0004) sees fleet-wide shifts, not a fault on a few units (`s_11` +0.5 on
  10 of 100 units stayed "stable"). Normal degradation also shifts sensors.
- **Input assumptions:** one reading per cycle. Gaps are accepted, but windows count readings, not
  elapsed cycles. Sensor values are checked for being finite, not for being plausible.

## Ethical and safety considerations

- **The costs are asymmetric.** A late prediction can mean an in-service failure; an early one wastes
  useful life. The model is *not* trained with an asymmetric loss and is late more often than early.
  Use the lower bound, not the point prediction.
- **A person must decide.** Predictions complement inspections and mandated maintenance limits; they
  never extend or replace them.
- **"OK" and "stable" are not guarantees.** About 1 interval in 10 is expected to miss, misses can
  cluster on one engine, and a single faulty sensor can go unnoticed.
- **Automation bias:** ranked tables and colours look authoritative. The dashboard therefore shows
  intervals, marks stale predictions and states the drift caveat.
- **Before any real use:** validate on real engines, set thresholds from the real costs of late and
  early maintenance, monitor each unit, and obtain the approvals the domain requires.
- **Privacy:** the data is simulated and contains no personal data.

## Contact

Questions and issues:
[github.com/manujsingh007/industrial-predictive-maintenance](https://github.com/manujsingh007/industrial-predictive-maintenance/issues)
