"""Fixtures for integration tests against a real PostgreSQL (``DATABASE_URL``).

Each test session works in its own temporary schema (``test_<random>``), created by running
the real Alembic migrations and dropped at the end. Tables in the regular schema, such as
local development data, are never touched.
"""

import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from predmaint import config
from predmaint.data.load import SENSOR_COLUMNS
from predmaint.db.settings import get_database_url
from predmaint.features.selection import KEPT_SENSORS
from predmaint.models.conformal import MONDRIAN_EDGES
from predmaint.models.gbdt import make_lgbm
from predmaint.serving.api import create_app
from predmaint.serving.predictor import RULPredictor, make_predictor
from predmaint.serving.registry import LoadedModel
from predmaint.training.run_cv import build_features


def base_url() -> str:
    url = get_database_url()
    if not url:
        pytest.fail("DATABASE_URL is not set: start PostgreSQL (docker compose up -d) and set it")
    return url


def alembic_config() -> Config:
    cfg = Config(str(config.project_root() / "alembic.ini"))
    cfg.attributes["configure_logger"] = False
    return cfg


@contextmanager
def fresh_schema(migrate: bool = True) -> Iterator[str]:
    """Create a temporary schema, optionally migrate it to head, yield its URL, then drop it."""
    base = base_url()
    schema = f"test_{uuid.uuid4().hex[:10]}"
    admin = create_engine(base)
    with admin.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    url = make_url(base).update_query_dict({"options": f"-csearch_path={schema}"})
    schema_url = url.render_as_string(hide_password=False)
    try:
        if migrate:
            upgrade(schema_url, "head")
        yield schema_url
    finally:
        with admin.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def upgrade(url: str, revision: str) -> None:
    engine = create_engine(url)
    with engine.begin() as conn:
        cfg = alembic_config()
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, revision)
    engine.dispose()


def downgrade(url: str, revision: str) -> None:
    engine = create_engine(url)
    with engine.begin() as conn:
        cfg = alembic_config()
        cfg.attributes["connection"] = conn
        command.downgrade(cfg, revision)
    engine.dispose()


@pytest.fixture(scope="session")
def schema_url() -> Iterator[str]:
    with fresh_schema() as url:
        yield url


@pytest.fixture(autouse=True)
def clean_tables(schema_url: str) -> None:
    engine = create_engine(schema_url)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE sensor_readings, predictions RESTART IDENTITY"))
    engine.dispose()


@pytest.fixture(scope="session")
def predictor() -> RULPredictor:
    """A tiny real predictor trained on synthetic engines (no MLflow, no real data)."""
    rng = np.random.default_rng(0)
    frames = []
    for unit in range(1, 7):
        n = 120 + 10 * unit
        f = pd.DataFrame(rng.normal(100, 1, (n, len(SENSOR_COLUMNS))), columns=SENSOR_COLUMNS)
        f[KEPT_SENSORS[0]] += np.linspace(0, 5, n)
        f.insert(0, "unit", unit)
        f.insert(1, "cycle", np.arange(1, n + 1))
        f["rul"] = n - f["cycle"]
        frames.append(f)
    data, features = build_features(pd.concat(frames, ignore_index=True), "window", (10, 30))
    model = make_lgbm(n_estimators=20).fit(data[features], np.minimum(data["rul"], 125))
    return make_predictor(
        model,
        sensors=KEPT_SENSORS,
        windows=(10, 30),
        cap=125,
        bin_edges=MONDRIAN_EDGES,
        q_per_bin=(6.0, 16.0, 29.0, 33.0),
        alpha=0.1,
        subset="FD001",
    )


@pytest.fixture
def client(predictor: RULPredictor, schema_url: str) -> Iterator[TestClient]:
    app = create_app(lambda: LoadedModel(predictor, "rul-lgbm-fd001", "7"), database_url=schema_url)
    with TestClient(app) as c:
        yield c
