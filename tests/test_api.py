"""Tests for the FastAPI service with a fake predictor (no MLflow, no real data)."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from predmaint.data.load import SENSOR_COLUMNS, SETTING_COLUMNS
from predmaint.serving.api import create_app, get_model
from predmaint.serving.predictor import InvalidHistoryError, RULPrediction
from predmaint.serving.registry import LoadedModel
from predmaint.serving.schemas import MAX_READINGS

RESPONSE_KEYS = {"unit_id", "rul", "lower", "upper", "n_cycles_seen", "model_name", "model_version"}


class FakePredictor:
    """Returns a fixed prediction and records the DataFrame it was given."""

    def __init__(self, error: str | None = None) -> None:
        self.error = error
        self.seen: pd.DataFrame | None = None

    def predict_engine(self, history: pd.DataFrame) -> RULPrediction:
        self.seen = history
        if self.error:
            raise InvalidHistoryError(self.error)
        return RULPrediction(
            unit=int(history["unit"].iloc[0]),
            rul=42.0,
            rul_raw=42.0,
            lower=30.0,
            upper=55.5,
            n_cycles_seen=len(history),
            last_cycle=int(history["cycle"].iloc[-1]),
            pred_bin="30-60",
        )


def reading(cycle: int, **overrides: float) -> dict:
    row = {"cycle": cycle, **dict.fromkeys(SETTING_COLUMNS, 0.0)}
    row.update({s: 100.0 + i for i, s in enumerate(SENSOR_COLUMNS)})
    row.update(overrides)
    return row


def request_body(n: int = 3, unit_id: int = 7) -> dict:
    return {"unit_id": unit_id, "readings": [reading(c) for c in range(1, n + 1)]}


@pytest.fixture
def fake() -> FakePredictor:
    return FakePredictor()


@pytest.fixture
def client(fake: FakePredictor):
    app = create_app(lambda: LoadedModel(fake, "rul-lgbm-fd001", "3"))
    with TestClient(app) as c:  # "with" runs the lifespan (startup loading)
        yield c


@pytest.fixture
def client_without_model():
    def failing_loader() -> LoadedModel:
        raise RuntimeError("no registered model 'rul-lgbm-fd001'")

    with TestClient(create_app(failing_loader)) as c:
        yield c


# --- health ----------------------------------------------------------------------------------


def test_health_reports_model(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "model_name": "rul-lgbm-fd001",
        "model_version": "3",
        "database": "not configured",
    }


def test_health_is_503_without_model(client_without_model: TestClient) -> None:
    r = client_without_model.get("/health")
    assert r.status_code == 503
    assert r.json()["status"] == "unavailable"


# --- predict: happy path ---------------------------------------------------------------------


def test_predict_returns_schema_and_values(client: TestClient, fake: FakePredictor) -> None:
    r = client.post("/predict", json=request_body(n=3, unit_id=7))
    assert r.status_code == 200
    body = r.json()
    assert set(body) == RESPONSE_KEYS
    assert body == {
        "unit_id": 7,
        "rul": 42.0,
        "lower": 30.0,
        "upper": 55.5,
        "n_cycles_seen": 3,
        "model_name": "rul-lgbm-fd001",
        "model_version": "3",
    }


def test_request_is_converted_to_one_engine_history(
    client: TestClient, fake: FakePredictor
) -> None:
    client.post("/predict", json=request_body(n=4, unit_id=12))
    seen = fake.seen
    assert seen is not None
    assert list(seen.columns) == ["unit", "cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS]
    assert seen["unit"].tolist() == [12] * 4
    assert seen["cycle"].tolist() == [1, 2, 3, 4]
    assert seen["s_2"].iloc[0] == 101.0


def test_single_reading_is_enough(client: TestClient) -> None:
    r = client.post("/predict", json=request_body(n=1))
    assert r.status_code == 200 and r.json()["n_cycles_seen"] == 1


# --- predict: invalid input -> 422 -----------------------------------------------------------


def mutate(body: dict, fn) -> dict:
    fn(body)
    return body


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (mutate(request_body(3), lambda b: b["readings"].append(reading(3))), "duplicate cycle"),
        (mutate(request_body(3), lambda b: b["readings"].reverse()), "strictly increasing"),
        ({"unit_id": 1, "readings": []}, "at least 1 item"),
        (
            {"unit_id": 1, "readings": [reading(c) for c in range(1, MAX_READINGS + 2)]},
            f"at most {MAX_READINGS} items",
        ),
        (mutate(request_body(2), lambda b: b["readings"][0].pop("s_11")), "s_11"),
        (mutate(request_body(2), lambda b: b["readings"][0].update(s_99=1.0)), "s_99"),
        (
            mutate(request_body(2), lambda b: b["readings"][0].update(cycle=0)),
            "greater than or equal to 1",
        ),
        (mutate(request_body(2), lambda b: b["readings"][0].update(s_4="hot")), "s_4"),
        ({"readings": [reading(1)]}, "unit_id"),
    ],
    ids=[
        "duplicate",
        "unsorted",
        "empty",
        "too many",
        "missing sensor",
        "unknown field",
        "cycle 0",
        "text sensor",
        "no unit_id",
    ],
)
def test_invalid_requests_give_422(client: TestClient, fake: FakePredictor, body, message) -> None:
    r = client.post("/predict", json=body)
    assert r.status_code == 422
    assert message in str(r.json()["detail"])
    assert fake.seen is None  # rejected before reaching the model


def test_predictor_validation_error_becomes_422() -> None:
    fake = FakePredictor(error="sensor values must be finite numbers; problems in: ['s_4']")
    with TestClient(create_app(lambda: LoadedModel(fake, "m", "1"))) as c:
        r = c.post("/predict", json=request_body(2))
    assert r.status_code == 422
    assert r.json()["detail"] == "sensor values must be finite numbers; problems in: ['s_4']"


# --- no model -> 503 ----------------------------------------------------------------------------


def test_predict_is_503_without_model(client_without_model: TestClient) -> None:
    r = client_without_model.post("/predict", json=request_body(2))
    assert r.status_code == 503
    assert "no registered model" in r.json()["detail"]


def test_dependency_can_be_overridden(client_without_model: TestClient) -> None:
    # The predictor is a FastAPI dependency, so tests (or other callers) can swap it.
    fake = FakePredictor()
    client_without_model.app.dependency_overrides[get_model] = lambda: LoadedModel(fake, "x", "9")
    r = client_without_model.post("/predict", json=request_body(2))
    assert r.status_code == 200 and r.json()["model_version"] == "9"


# --- with the real predictor class (tiny synthetic model, still no MLflow) ---------------------


def test_real_predictor_behind_the_api() -> None:
    from predmaint.features.selection import KEPT_SENSORS
    from predmaint.models.conformal import MONDRIAN_EDGES
    from predmaint.models.gbdt import make_lgbm
    from predmaint.serving.predictor import make_predictor
    from predmaint.training.run_cv import build_features

    rng = np.random.default_rng(0)
    frames = []
    for u in range(1, 7):
        n = 120 + 10 * u
        f = pd.DataFrame(rng.normal(100, 1, (n, len(SENSOR_COLUMNS))), columns=SENSOR_COLUMNS)
        f.insert(0, "unit", u)
        f.insert(1, "cycle", np.arange(1, n + 1))
        f["rul"] = n - f["cycle"]
        frames.append(f)
    data, features = build_features(pd.concat(frames, ignore_index=True), "window", (10, 30))
    model = make_lgbm(n_estimators=10).fit(data[features], np.minimum(data["rul"], 125))
    predictor = make_predictor(
        model,
        sensors=KEPT_SENSORS,
        windows=(10, 30),
        cap=125,
        bin_edges=MONDRIAN_EDGES,
        q_per_bin=(6, 16, 29, 33),
        alpha=0.1,
        subset="FD001",
    )

    with TestClient(create_app(lambda: LoadedModel(predictor, "rul-lgbm-fd001", "1"))) as c:
        r = c.post("/predict", json=request_body(n=40))
    assert r.status_code == 200
    body = r.json()
    assert body["n_cycles_seen"] == 40
    assert 0 <= body["lower"] <= body["rul"] <= body["upper"]


# --- database endpoints without a database (DB-free) ----------------------------------------


def test_unit_endpoints_are_503_without_database(client: TestClient) -> None:
    body = {"readings": [reading(1)]}
    for method, url, kwargs in (
        ("post", "/units/5/readings", {"json": body}),
        ("get", "/units/5/prediction", {}),
        ("get", "/units/5/predictions", {}),
        ("get", "/units/5", {}),
    ):
        r = getattr(client, method)(url, **kwargs)
        assert r.status_code == 503, url
        assert "DATABASE_URL" in r.json()["detail"]


def test_unreachable_database_gives_503_not_500(fake: FakePredictor) -> None:
    # Nothing listens on port 1: the connection is refused immediately.
    url = "postgresql+psycopg://user:pw@127.0.0.1:1/none"
    app = create_app(lambda: LoadedModel(fake, "m", "1"), database_url=url)
    with TestClient(app) as c:
        health = c.get("/health")
        assert health.status_code == 200  # the model is fine; /predict still works
        assert health.json()["database"] == "unavailable"
        r = c.post("/units/5/readings", json={"readings": [reading(1)]})
    assert r.status_code == 503
    assert "Database unavailable" in r.json()["detail"]


def test_stateless_predict_works_without_database(client: TestClient) -> None:
    assert client.post("/predict", json=request_body(2)).status_code == 200
