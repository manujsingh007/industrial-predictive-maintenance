# Industrial Predictive Maintenance

[![CI](https://github.com/manujsingh007/industrial-predictive-maintenance/actions/workflows/ci.yml/badge.svg)](https://github.com/manujsingh007/industrial-predictive-maintenance/actions/workflows/ci.yml)

Unplanned failures of industrial equipment are costly and can be unsafe, while replacing parts too
early wastes useful life. This project uses multivariate sensor data from turbofan engines run to
failure to estimate **remaining useful life (RUL)**, predict upcoming failures, and detect anomalous
behaviour, so that maintenance can be planned before a failure occurs. It is built on NASA's
simulated C-MAPSS dataset ([dataset documentation](docs/dataset.md)).

> **Headline result: official FD001 test set, 100 engines, evaluated once**
>
> LightGBM on causal window features:
>
> - **RMSE 14.59** cycles (95% CI 12.46–16.63), **MAE 11.07**, PHM08 score 314.6.
> - The linear baseline scores RMSE 21.90. LightGBM is better by **7.31 cycles RMSE** (paired
>   bootstrap 95% CI −10.20 to −4.57).
> - 90% Mondrian prediction intervals covered **93 of 100 engines**. Per predicted-RUL bin,
>   coverage ranged from 83% to 100% (small bins; see the report).
>
> The test set was sealed until a single final run, after every modelling decision had been
> frozen. Full results: [reports/final_test_fd001.md](reports/final_test_fd001.md).

## Status

**Done**

- [x] Project setup: uv, ruff, pytest, CI on GitHub Actions
- [x] Reproducible data download with SHA-256 checksum verification;
  [dataset documentation](docs/dataset.md)
- [x] **Milestone 1: data understanding (FD001)**
  - Data loader, [exploratory analysis](notebooks/01_eda_fd001.ipynb)
  - RUL cap and sensor selection ([ADR 0001](docs/adr/0001-rul-cap-and-sensor-selection.md))
  - Frozen [evaluation protocol](docs/evaluation.md)
- [x] **Milestone 2: evaluation framework and baselines**
  - Engine-based cross-validation
  - Metrics: RMSE, MAE, PHM08 score, bootstrap intervals
  - Mean and linear baselines
  - MLflow experiment tracking
- [x] **Milestone 3: models, intervals and final test (FD001)**
  - Causal window features
  - LightGBM, chosen as the primary model ([ADR 0002](docs/adr/0002-model-choice.md))
  - 1D-CNN (PyTorch) for comparison
  - RUL-cap tuning: 125 confirmed (ADR 0001)
  - Mondrian conformal prediction intervals
    ([ADR 0003](docs/adr/0003-prediction-intervals.md))
  - One-time final test evaluation ([report](reports/final_test_fd001.md))

**Next: Milestone 4**

- [x] Error analysis: late vs early errors, error vs history length, worst engines and the
  effect of the RUL cap ([summary](docs/error_analysis.md),
  [notebook](notebooks/02_error_analysis_fd001.ipynb)).
- [ ] Serving: FastAPI service with PostgreSQL storage.
- [ ] Dashboard.
- [ ] Docker packaging.

Later: failure prediction and anomaly detection.

## Cross-validation results (development, FD001 training data)

These tables come from **5-fold engine cross-validation on the training data**, used to make
every modelling decision. They are **not** test-set results; see the headline result above.

| Model | CV RMSE | CV MAE |
|---|---|---|
| Mean RUL baseline | 45.11 ± 0.09 | 38.59 ± 0.04 |
| Linear regression | 24.34 ± 0.82 | 19.89 ± 0.74 |
| Linear regression + window features | 20.20 ± 0.87 | 16.53 ± 0.70 |
| 1D-CNN on 30-cycle sequences | 18.98 ± 1.04 | 14.49 ± 0.76 |
| **LightGBM + window features** (primary model) | **16.51 ± 1.06** | **12.36 ± 0.90** |

- Values are in cycles, as mean ± standard deviation across the 5 folds.
- The training target is capped at RUL 125. Scores are against the uncapped true RUL, on
  validation rows with true RUL ≤ 150.
- See [ADR 0002](docs/adr/0002-model-choice.md) for why LightGBM is the primary model.

### Cross-validated prediction intervals (90% target)

Each fold's intervals are calibrated only on the other four folds.

| Method | Coverage | Mean width | Worst predicted-RUL bin | Width when predicted RUL < 30 |
|---|---|---|---|---|
| Split conformal | 90.0% | 53.8 | 82.5% (90+) | 42.9 |
| CQR | 90.0% | 61.1 | 80.2% (90+) | 42.7 |
| **Mondrian conformal** (chosen) | **89.9%** | **47.9** | **89.3% (60–90)** | **12.1** |

Widths are in cycles. See [ADR 0003](docs/adr/0003-prediction-intervals.md) for the full tables
and limitations.

### Reproduce

After the Quickstart below:

```bash
uv run python -m predmaint.training.run_baselines --subset FD001                        # baselines
uv run python -m predmaint.training.run_cv --model lgbm --features window --subset FD001  # LightGBM
uv run python -m predmaint.training.run_conformal --subset FD001 --method all             # intervals
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db    # browse runs at http://127.0.0.1:5000
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
