# Industrial Predictive Maintenance

[![CI](https://github.com/manujsingh007/industrial-predictive-maintenance/actions/workflows/ci.yml/badge.svg)](https://github.com/manujsingh007/industrial-predictive-maintenance/actions/workflows/ci.yml)

Unplanned failures of industrial equipment are costly and can be unsafe, while replacing parts too
early wastes useful life. This project uses multivariate sensor data from turbofan engines run to
failure to estimate **remaining useful life (RUL)**, predict upcoming failures, and detect anomalous
behaviour, so that maintenance can be planned before a failure occurs. It is built on NASA's
simulated C-MAPSS dataset ([dataset documentation](docs/dataset.md)).

## Status: work in progress

**Done**

- [x] Project setup: uv, ruff, pytest
- [x] CI on GitHub Actions (lint, format check, tests)
- [x] Reproducible data download with SHA-256 checksum verification
- [x] Dataset documentation ([docs/dataset.md](docs/dataset.md))
- [x] **Milestone 1: data understanding (FD001)**
  - Data loader with RUL labels (`predmaint.data.load`)
  - Exploratory analysis ([notebooks/01_eda_fd001.ipynb](notebooks/01_eda_fd001.ipynb))
  - RUL cap and sensor selection
    ([ADR 0001](docs/adr/0001-rul-cap-and-sensor-selection.md))
  - Frozen evaluation protocol ([docs/evaluation.md](docs/evaluation.md))
- [x] **Milestone 2: evaluation framework and baselines (FD001)**
  - Cross-validation split by engine (`predmaint.evaluation.splits`)
  - Metrics: RMSE, MAE, PHM08 score, bootstrap confidence intervals
    (`predmaint.evaluation.metrics`)
  - Baselines: mean RUL and linear regression (`predmaint.models.baselines`)
  - Experiment tracking with MLflow (`predmaint.tracking`)

**Next: Milestone 3**

- [ ] Window features over each engine's recent cycles
- [ ] LightGBM model
- [ ] A deep sequence model
- [ ] RUL-cap tuning (110–150, per [ADR 0001](docs/adr/0001-rul-cap-and-sensor-selection.md))
- [ ] Prediction intervals

Later: failure prediction and anomaly detection.

## Results so far

5-fold engine cross-validation on **FD001 training data**. These are **not** official test-set
results; the test set is used once, at the end ([evaluation protocol](docs/evaluation.md)).

| Model | CV RMSE | CV MAE |
|---|---|---|
| Mean RUL baseline | 45.11 ± 0.09 | 38.59 ± 0.04 |
| Linear regression | 24.34 ± 0.82 | 19.89 ± 0.74 |
| Linear regression + window features | 20.20 ± 0.87 | 16.53 ± 0.70 |
| 1D-CNN on 30-cycle sequences | 18.98 ± 1.04 | 14.49 ± 0.76 |
| **LightGBM + window features** (primary model) | **16.51 ± 1.06** | **12.36 ± 0.90** |

The primary model is LightGBM on window features; see
[ADR 0002](docs/adr/0002-model-choice.md) for why.

### Prediction intervals

These are 90% intervals for LightGBM from the same cross-validation. Each fold's interval is
calibrated only on the other four folds.

| Method | Coverage | Mean width | Worst predicted-RUL bin | Width when predicted RUL < 30 |
|---|---|---|---|---|
| Split conformal | 90.0% | 53.8 | 82.5% (90+) | 42.9 |
| CQR | 90.0% | 61.1 | 80.2% (90+) | 42.7 |
| **Mondrian conformal** (chosen) | **89.9%** | **47.9** | **89.3% (60–90)** | **12.1** |

Widths are in cycles. Mondrian uses a separate correction per predicted-RUL bin: it stays close
to 90% in every bin and is about 3.5× narrower near failure. See
[ADR 0003](docs/adr/0003-prediction-intervals.md) for the full tables and limitations.

To reproduce:

```bash
uv run python -m predmaint.training.run_conformal --subset FD001 --method all
```

Notes:

- Values are in cycles, as mean ± standard deviation across the 5 folds.
- The training target is capped at RUL 125, but scores are against the uncapped true RUL, on
  validation rows with true RUL ≤ 150.

To reproduce the table and browse the logged runs (after the Quickstart below):

```bash
uv run python -m predmaint.training.run_baselines --subset FD001   # prints the table, logs to MLflow
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db           # then open http://127.0.0.1:5000
```

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). It installs Python 3.12 automatically.

macOS: LightGBM needs OpenMP: `brew install libomp`

```bash
git clone https://github.com/manujsingh007/industrial-predictive-maintenance.git
cd industrial-predictive-maintenance
uv sync                                     # create the environment from uv.lock
uv run python -m predmaint.data.download    # download and verify the dataset into data/raw/cmapss/
uv run pytest                               # run the tests
```
