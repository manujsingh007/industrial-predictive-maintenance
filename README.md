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
- [x] **Milestone 4: error analysis**
  - Late vs early errors, error vs history length, worst engines and the effect of the RUL cap
  - [Summary](docs/error_analysis.md) and [notebook](notebooks/02_error_analysis_fd001.ipynb)
- [x] **Milestone 5: serving**
  - Model packaged and registered in MLflow as `rul-lgbm-fd001` with a `champion` alias
  - FastAPI service: `/health`, `/predict` (stateless), `/units/{id}/readings`,
    `/units/{id}/prediction`, `/units/{id}/predictions`
  - PostgreSQL storage with Alembic migrations and a UNIQUE(unit, cycle) constraint
  - Replay simulator that streams test engines into the service
  - Idempotent ingestion: identical re-sends are no-ops, and conflicting values are rejected
    with 409
  - Integration tests against PostgreSQL, also run in CI
  - **No training/serving skew:** for test engine 3, the live service reproduces the final-test
    prediction exactly (RUL 39.33 at cycle 126, interval 23.36–55.30)
- [x] **Milestone 6: dashboard and monitoring**
  - Streamlit dashboard that talks only to the API:
    - a fleet view ranked by the interval's lower bound;
    - a per-engine drill-down with 90% prediction intervals and sensor trends;
    - alerts: critical when the lower bound is below 20 cycles, warning below 50.
  - **Alert check on the 100 FD001 test engines:** every engine with true RUL ≤ 20 (16 of 16)
    was flagged critical, and all 21 critical alerts were within 29 cycles of failure.
  - Drift monitoring with PSI, validated by fault injection
    ([ADR 0004](docs/adr/0004-drift-monitoring.md))

**Next: Milestone 7**

- [ ] Docker images for the API and the dashboard
- [ ] One-command `docker compose up` for the full stack
- [ ] Architecture diagram
- [ ] Screenshots
- [ ] Model card

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

## Quickstart with Docker

The whole stack runs with one command: PostgreSQL, migrations, model build, API and dashboard.
It needs only Docker.

```bash
git clone https://github.com/manujsingh007/industrial-predictive-maintenance.git
cd industrial-predictive-maintenance
cp .env.example .env              # set POSTGRES_PASSWORD (and DATABASE_URL to match)
docker compose up --build         # API: http://localhost:8000/docs  dashboard: http://localhost:8501
```

The first start builds the images, downloads the dataset (checksum-verified) and trains and
registers the model. That takes a few minutes; it took about 3 minutes here, including the
image build. Later starts reuse everything from Docker volumes and are ready in seconds.

Then stream the 100 FD001 test engines into the service and watch the dashboard fill up:

```bash
docker compose --profile demo up replay
```

- **Ports in use?** Set `API_PORT` / `DASHBOARD_PORT` in `.env`.
- **Stop:** `docker compose down`. All data is kept.
- **Reset everything:** `docker compose down -v` deletes the volumes, i.e. the stored
  readings and predictions **and** the trained model. The next `up` starts from scratch.
- **The replay never overwrites stored readings.** If a unit already holds different values (for
  example after a drift-injection experiment), it stops with a 409. Reset first, or replay
  into new ids with `--unit-offset`.

Settings come from environment variables (see `predmaint.config`): `PREDMAINT_DATA_DIR`,
`MLFLOW_TRACKING_URI`, `PREDMAINT_ARTIFACT_ROOT`, `DATABASE_URL`, and `API_BASE_URL` for the
dashboard.

## Quickstart

For local development without Docker.

Requires [uv](https://docs.astral.sh/uv/). It installs Python 3.12 automatically.

macOS: LightGBM needs OpenMP: `brew install libomp`

```bash
git clone https://github.com/manujsingh007/industrial-predictive-maintenance.git
cd industrial-predictive-maintenance
uv sync                                     # create the environment from uv.lock
uv run python -m predmaint.data.download    # download and verify the dataset into data/raw/cmapss/
uv run pytest                               # run the tests
```

## Run the full service locally

Needs Docker (Docker Desktop on macOS) and the Quickstart above. Run each step in its own
terminal where noted.

```bash
# 1. Database: copy the example settings, set a password, then start PostgreSQL
cp .env.example .env                  # edit POSTGRES_PASSWORD and DATABASE_URL to match
docker compose up -d                  # wait for "healthy": docker compose ps

# 2. Schema: create or upgrade the tables
uv run alembic upgrade head

# 3. Model: train on all training engines and register it as the "champion" (once)
uv run python -m predmaint.training.build_model --subset FD001

# 4. API (keep this terminal open); interactive docs at http://127.0.0.1:8000/docs
uv run uvicorn predmaint.serving.api:app

# 5. In a second terminal: replay test engines as if they were live
uv run python -m predmaint.simulation.replay --subset FD001 --units 1-10 --predict-every 10

# 6. In a third terminal: the fleet dashboard (opens http://localhost:8501)
uv run --group dashboard streamlit run src/predmaint/dashboard/app.py
```

- **The replay** sends each engine's readings in cycle order and asks for a prediction every 10
  cycles and at the last one. Add `--delay 0.2` to watch it at a "live" pace.
- **Resuming:** if the replay is interrupted, run the same command again.
  - Already-stored cycles are re-sent once as a check. Identical readings are accepted as a
    no-op, so retries are always safe.
  - If the stored readings differ, for example because another engine's data is under the same
    unit id, the replay stops with a clear error instead of mixing the two histories.
- **No label leakage:** the replay never reads the true-RUL file, so no labels reach the service.
- **Results:** `GET /units/{id}/predictions` returns a unit's prediction history.
- **Dashboard:** it reads only from the API. The default is http://127.0.0.1:8000; set
  `API_BASE_URL` to use another one.
  - The fleet table ranks units by the **lower bound** of their 90% interval (critical below 20
    cycles, warning below 50; both adjustable in the sidebar).
  - Selecting a unit shows its predicted RUL with the interval band and four key sensors.
  - Press **Refresh** after new data arrives.
  - True RUL is never shown: an operator would not know it.
- **Stopping:** `docker compose down` stops PostgreSQL; the data stays in a Docker volume.

On macOS, if `docker` is not found although Docker Desktop is running, add its tools to your
PATH: `export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"`.

## Monitoring

`GET /monitoring/drift?last_n_cycles=30` (also shown in the dashboard) reports the Population
Stability Index for the 3 operating settings and 14 kept sensors. It compares each unit's last 30
readings, pooled across the fleet, with the training data's profile stored alongside the model.
Thresholds: stable < 0.1 ≤ moderate < 0.25 ≤ significant.

What it can and cannot detect:

- **Fleet-wide shifts:** detected. `s_11` +0.5 on all 100 test units gives PSI 4.58.
- **A fault on a few units:** not detected. The same offset on 10 units gives 0.050, "stable".
- **Engine degradation** also shifts sensor distributions, so sensor drift needs care. Drift in
  the operating settings is the clearer signal.

See [ADR 0004](docs/adr/0004-drift-monitoring.md) for the validation and future work.

To try it, replay with an injected fault into a fresh database or new unit ids, e.g.
`--inject-drift s_11:0.5 --unit-offset 1000`. Units already stored with their real readings
reject altered values with 409, by design.
