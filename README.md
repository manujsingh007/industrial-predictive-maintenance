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

**Next**

- [ ] RUL estimation: baselines (mean RUL, linear regression), then stronger models
- [ ] Failure prediction
- [ ] Anomaly detection

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). It installs Python 3.12 automatically.

```bash
git clone https://github.com/manujsingh007/industrial-predictive-maintenance.git
cd industrial-predictive-maintenance
uv sync                                     # create the environment from uv.lock
uv run python -m predmaint.data.download    # download and verify the dataset into data/raw/cmapss/
uv run pytest                               # run the tests
```
