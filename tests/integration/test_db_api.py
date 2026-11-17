"""Integration tests: migrations, storage and the /units endpoints against real PostgreSQL."""

import pandas as pd
import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from predmaint.data.load import SENSOR_COLUMNS, SETTING_COLUMNS
from predmaint.db.models import Base, SensorReadingRow

from .conftest import downgrade, fresh_schema, upgrade

pytestmark = pytest.mark.integration


def reading(cycle: int, drift: float = 0.0) -> dict:
    row = {"cycle": cycle, **dict.fromkeys(SETTING_COLUMNS, 0.001 * cycle)}
    row.update({s: 100.0 + i + drift * cycle for i, s in enumerate(SENSOR_COLUMNS)})
    return row


def readings(cycles: range, drift: float = 0.05) -> dict:
    return {"readings": [reading(c, drift) for c in cycles]}


# --- migrations -------------------------------------------------------------------------------


def test_migration_matches_the_models(schema_url: str) -> None:
    engine = create_engine(schema_url)
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
        tables = set(inspect(conn).get_table_names())
        unique = inspect(conn).get_unique_constraints("sensor_readings")
    engine.dispose()
    assert {"sensor_readings", "predictions", "alembic_version"} <= tables
    assert diff == []  # models and migration describe the same schema
    assert {
        "name": "uq_sensor_readings_unit_cycle",
        "column_names": ["unit_id", "cycle"],
    }.items() <= unique[0].items()


def test_downgrade_and_upgrade_round_trip() -> None:
    with fresh_schema(migrate=True) as url:
        downgrade(url, "base")
        engine = create_engine(url)
        with engine.connect() as conn:
            assert set(inspect(conn).get_table_names()) == {"alembic_version"}
        engine.dispose()
        upgrade(url, "head")
        engine = create_engine(url)
        with engine.connect() as conn:
            assert {"sensor_readings", "predictions"} <= set(inspect(conn).get_table_names())
        engine.dispose()


def test_database_rejects_duplicate_unit_cycle(schema_url: str) -> None:
    engine = create_engine(schema_url)
    with Session(engine) as session:
        values = {k: v for k, v in reading(1).items() if k != "cycle"}
        session.add_all(
            [
                SensorReadingRow(unit_id=1, cycle=1, **values),
                SensorReadingRow(unit_id=1, cycle=1, **values),
            ]
        )
        with pytest.raises(IntegrityError):
            session.commit()
    engine.dispose()


# --- POST /units/{id}/readings -------------------------------------------------------------


def test_store_readings_and_accumulate(client: TestClient) -> None:
    r = client.post("/units/3/readings", json=readings(range(1, 11)))
    assert r.status_code == 201
    assert r.json() == {"unit_id": 3, "n_stored": 10, "n_total": 10, "last_cycle": 10}
    r = client.post("/units/3/readings", json=readings(range(11, 16)))
    assert r.json() == {"unit_id": 3, "n_stored": 5, "n_total": 15, "last_cycle": 15}


def test_duplicate_cycles_give_409_and_store_nothing(client: TestClient) -> None:
    client.post("/units/4/readings", json=readings(range(1, 6)))
    r = client.post("/units/4/readings", json=readings(range(4, 9)))  # 4, 5 already stored
    assert r.status_code == 409
    assert "[4, 5]" in r.json()["detail"]
    again = client.post("/units/4/readings", json=readings(range(6, 7)))
    assert again.json()["n_total"] == 6  # cycles 6-8 of the rejected batch were not stored


def test_same_cycle_for_different_units_is_fine(client: TestClient) -> None:
    assert client.post("/units/1/readings", json=readings(range(1, 3))).status_code == 201
    assert client.post("/units/2/readings", json=readings(range(1, 3))).status_code == 201


@pytest.mark.parametrize(
    ("url", "body"),
    [
        ("/units/5/readings", {"readings": [reading(2), reading(1)]}),
        ("/units/5/readings", {"readings": []}),
        ("/units/5/readings", {"readings": [{**reading(1), "s_4": "hot"}]}),
        ("/units/0/readings", readings(range(1, 3))),
    ],
    ids=["unsorted", "empty", "text sensor", "unit 0"],
)
def test_invalid_readings_give_422(client: TestClient, url: str, body: dict) -> None:
    assert client.post(url, json=body).status_code == 422


def test_stored_values_round_trip_exactly(client: TestClient, schema_url: str) -> None:
    sent = readings(range(1, 4))["readings"]
    client.post("/units/8/readings", json={"readings": sent})
    from predmaint.db import repository

    engine = create_engine(schema_url)
    with Session(engine) as session:
        history = repository.get_history(session, 8)
    engine.dispose()
    expected = pd.DataFrame(sent)
    pd.testing.assert_frame_equal(
        history.drop(columns="unit").reset_index(drop=True), expected, check_dtype=False
    )


# --- GET /units/{id}/prediction and /predictions ----------------------------------------------


def test_unknown_unit_gives_404(client: TestClient) -> None:
    assert client.get("/units/99/prediction").status_code == 404
    assert client.get("/units/99/predictions").status_code == 404


def test_prediction_uses_stored_history_and_is_stored(client: TestClient, predictor) -> None:
    body = readings(range(1, 41))
    client.post("/units/6/readings", json=body)
    r = client.get("/units/6/prediction")
    assert r.status_code == 200
    got = r.json()

    history = pd.DataFrame(body["readings"])
    history.insert(0, "unit", 6)
    direct = predictor.predict_engine(history)
    assert got["rul"] == pytest.approx(direct.rul)
    assert (got["lower"], got["upper"]) == pytest.approx((direct.lower, direct.upper))
    assert got["n_cycles_seen"] == 40 and got["cycle"] == 40
    assert got["model_name"] == "rul-lgbm-fd001" and got["model_version"] == "7"

    stored = client.get("/units/6/predictions").json()
    assert len(stored) == 1 and stored[0]["id"] == got["id"]
    assert stored[0]["rul"] == pytest.approx(got["rul"])


def test_same_prediction_from_stateless_and_stored_paths(client: TestClient) -> None:
    body = readings(range(1, 31))
    client.post("/units/9/readings", json=body)
    stored = client.get("/units/9/prediction").json()
    stateless = client.post("/predict", json={"unit_id": 9, **body}).json()
    assert stored["rul"] == pytest.approx(stateless["rul"])
    assert stored["lower"] == pytest.approx(stateless["lower"])


def test_prediction_history_newest_first_with_limit(client: TestClient) -> None:
    client.post("/units/7/readings", json=readings(range(1, 11)))
    first = client.get("/units/7/prediction").json()
    client.post("/units/7/readings", json=readings(range(11, 21)))
    second = client.get("/units/7/prediction").json()
    client.post("/units/7/readings", json=readings(range(21, 31)))
    third = client.get("/units/7/prediction").json()

    history = client.get("/units/7/predictions").json()
    assert [p["id"] for p in history] == [third["id"], second["id"], first["id"]]
    assert [p["cycle"] for p in history] == [30, 20, 10]
    assert [p["id"] for p in client.get("/units/7/predictions?limit=2").json()] == [
        third["id"],
        second["id"],
    ]


def test_unit_with_readings_but_no_predictions_has_empty_history(client: TestClient) -> None:
    client.post("/units/10/readings", json=readings(range(1, 3)))
    r = client.get("/units/10/predictions")
    assert r.status_code == 200 and r.json() == []


def test_health_reports_database_ok(client: TestClient) -> None:
    assert client.get("/health").json()["database"] == "ok"


def test_unit_summary_for_resuming(client: TestClient) -> None:
    assert client.get("/units/11").status_code == 404
    client.post("/units/11/readings", json=readings(range(1, 26)))
    assert client.get("/units/11").json() == {
        "unit_id": 11,
        "n_readings": 25,
        "last_cycle": 25,
        "n_predictions": 0,
        "last_prediction_cycle": None,
    }
    client.get("/units/11/prediction")
    summary = client.get("/units/11").json()
    assert summary["n_predictions"] == 1 and summary["last_prediction_cycle"] == 25
