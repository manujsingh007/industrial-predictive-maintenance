"""Dashboard client and Streamlit app tests against a mock API (httpx.MockTransport).

The client tests need only httpx. The app tests need the "dashboard" dependency group and are
skipped without it.
"""

import json
import sys
from pathlib import Path

import httpx
import pandas as pd
import pytest

from predmaint.dashboard import client as client_module
from predmaint.dashboard.client import ApiClient, ApiError

APP = Path(__file__).parents[1] / "src" / "predmaint" / "dashboard" / "app.py"

UNITS = [
    {
        "unit_id": 1,
        "n_readings": 31,
        "last_cycle": 31,
        "latest_prediction": {
            "rul": 112.8,
            "lower": 79.7,
            "upper": 146.0,
            "cycle": 31,
            "created_at": "2026-11-18T10:00:00Z",
        },
    },
    {
        "unit_id": 3,
        "n_readings": 126,
        "last_cycle": 126,
        "latest_prediction": {
            "rul": 39.3,
            "lower": 23.4,
            "upper": 55.3,
            "cycle": 126,
            "created_at": "2026-11-18T10:00:00Z",
        },
    },
    {
        "unit_id": 7,
        "n_readings": 140,
        "last_cycle": 140,
        "latest_prediction": {
            "rul": 12.0,
            "lower": 5.6,
            "upper": 18.4,
            "cycle": 140,
            "created_at": "2026-11-18T10:00:00Z",
        },
    },
    {"unit_id": 9, "n_readings": 4, "last_cycle": 4, "latest_prediction": None},
]


def predictions(unit_id: int) -> list[dict]:
    return [
        {
            "id": 10 * unit_id + k,
            "unit_id": unit_id,
            "cycle": 25 * k,
            "rul": 120 - 20 * k,
            "lower": 100 - 20 * k,
            "upper": 140 - 20 * k,
            "model_name": "rul-lgbm-fd001",
            "model_version": "1",
            "created_at": f"2026-11-18T10:0{k}:00Z",
        }
        for k in (3, 2, 1)
    ]


def readings(unit_id: int) -> list[dict]:
    return [
        {
            "unit_id": unit_id,
            "cycle": c,
            "setting_1": 0.0,
            "setting_2": 0.0,
            "setting_3": 100.0,
            **{f"s_{i}": 100.0 + i + 0.1 * c for i in range(1, 22)},
            "ingested_at": "2026-11-18T10:00:00Z",
        }
        for c in range(1, 6)
    ]


def fake_api(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/health":
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "model_name": "rul-lgbm-fd001",
                "model_version": "1",
                "database": "ok",
            },
        )
    if path == "/units":
        return httpx.Response(200, json=UNITS)
    parts = path.strip("/").split("/")
    unit_id = int(parts[1])
    if parts[2] == "predictions":
        return httpx.Response(200, json=predictions(unit_id))
    if parts[2] == "readings":
        return httpx.Response(200, json=readings(unit_id))
    return httpx.Response(404, json={"detail": "not found"})


# --- client ----------------------------------------------------------------------------------


def test_client_reads_units_predictions_and_readings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module, "TRANSPORT", httpx.MockTransport(fake_api))
    c = ApiClient("http://api")
    assert [u["unit_id"] for u in c.units()] == [1, 3, 7, 9]
    assert len(c.predictions(3)) == 3 and len(c.readings(3)) == 5
    assert c.health()["status"] == "ok"


def test_client_sends_limit_only_when_given(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=[])

    monkeypatch.setattr(client_module, "TRANSPORT", httpx.MockTransport(record))
    c = ApiClient("http://api")
    c.readings(3)
    c.readings(3, limit=50)
    assert seen == ["http://api/units/3/readings", "http://api/units/3/readings?limit=50"]


def test_unreachable_api_gives_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr(client_module, "TRANSPORT", httpx.MockTransport(refuse))
    with pytest.raises(ApiError, match="Cannot reach the API at http://api .*Is it running"):
        ApiClient("http://api").units()


def test_error_responses_carry_the_api_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_db(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(503, json={"status": "unavailable", "database": "ok"})
        return httpx.Response(503, json={"detail": "Database not configured"})

    monkeypatch.setattr(client_module, "TRANSPORT", httpx.MockTransport(no_db))
    c = ApiClient("http://api")
    assert c.health()["status"] == "unavailable"  # API up but no model: not an exception
    with pytest.raises(ApiError, match="GET /units failed with 503: Database not configured"):
        c.units()


def test_api_url_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PREDMAINT_API_URL", "http://example:9000/")
    assert client_module.api_url() == "http://example:9000"
    monkeypatch.delenv("PREDMAINT_API_URL")
    assert client_module.api_url() == "http://127.0.0.1:8000"


# --- Streamlit app (needs the "dashboard" group) ----------------------------------------------


@pytest.fixture
def app_test(monkeypatch: pytest.MonkeyPatch):
    testing = pytest.importorskip("streamlit.testing.v1", reason="needs: uv sync --group dashboard")
    monkeypatch.setattr(client_module, "TRANSPORT", httpx.MockTransport(fake_api))
    return testing.AppTest.from_file(str(APP), default_timeout=30)


def test_app_fleet_view_kpis_and_ranking(app_test) -> None:
    at = app_test.run()
    assert not at.exception, at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics == {"Units monitored": "4", "● Critical": "1", "▲ Warning": "1"}
    table = at.dataframe[0].value
    assert table["unit"].tolist() == [7, 3, 1, 9]  # by lower bound; unpredicted unit last
    assert table["status"].tolist() == ["critical", "warning", "ok", "no prediction"]


def test_app_drill_down_and_no_true_rul(app_test) -> None:
    at = app_test.run()
    assert at.selectbox[0].options[0].startswith("Unit 7")
    at.selectbox[0].select(3).run()
    assert not at.exception
    page = json.dumps([str(e.value) for e in at.markdown] + [c.value for c in at.caption])
    assert "true rul" in page.lower()  # only in the explanation that it is not shown
    assert "true_rul" not in page and "RUL_FD001" not in page


def test_app_thresholds_change_statuses(app_test) -> None:
    at = app_test.run()
    at.number_input[0].set_value(30.0).run()  # critical < 30: unit 3 (lower 23.4) too
    assert {m.label: m.value for m in at.metric}["● Critical"] == "2"


def test_app_reports_unreachable_api(monkeypatch: pytest.MonkeyPatch) -> None:
    testing = pytest.importorskip("streamlit.testing.v1", reason="needs: uv sync --group dashboard")

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr(client_module, "TRANSPORT", httpx.MockTransport(refuse))
    at = testing.AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert "Cannot reach the API" in at.error[0].value
    assert len(at.metric) == 0  # nothing else rendered


def chart_layers_and_data(chart) -> tuple[list[str], list]:
    """Mark types of a rendered Altair chart and its datasets as DataFrames."""
    import pyarrow as pa

    spec = json.loads(chart.proto.spec)
    marks = [
        layer["mark"]["type"] if isinstance(layer["mark"], dict) else layer["mark"]
        for layer in spec.get("layer", [])
    ]
    data = [pa.ipc.open_stream(d.data.data).read_all().to_pandas() for d in chart.proto.datasets]
    return marks, data


def test_app_renders_rul_chart_with_band_and_thresholds_plus_four_sensors(app_test) -> None:
    at = app_test.run()
    charts = at.get("vega_lite_chart")
    assert len(charts) == 5  # 1 RUL chart + 4 key sensors
    marks, data = chart_layers_and_data(charts[0])
    assert marks == ["area", "line", "point", "rule", "text"]  # band, RUL, points, thresholds
    history = next(d for d in data if "rul" in d.columns)
    assert history["cycle"].tolist() == [25, 50, 75]  # predictions in cycle order
    limits = next(d for d in data if "label" in d.columns)
    assert limits["value"].tolist() == [20.0, 50.0]

    at.number_input[1].set_value(60.0).run()  # thresholds in the chart follow the sidebar
    _, data = chart_layers_and_data(at.get("vega_lite_chart")[0])
    assert next(d for d in data if "label" in d.columns)["value"].tolist() == [20.0, 60.0]


def displayed_table(at) -> pd.DataFrame:
    """The table as the browser shows it (the Styler's display values, not the raw data)."""
    import pyarrow as pa

    styler = at.dataframe[0].proto.arrow_data.styler
    return pa.ipc.open_stream(styler.display_values).read_all().to_pandas()


def test_table_shows_rul_lower_upper_with_one_decimal(app_test, monkeypatch) -> None:
    full_precision = {"rul": 39.330592, "lower": 23.363054, "upper": 55.29813}
    units = [dict(u) for u in UNITS]
    units[1] = {
        **units[1],
        "latest_prediction": {**units[1]["latest_prediction"], **full_precision},
    }
    monkeypatch.setattr(sys.modules[__name__], "UNITS", units)
    at = app_test.run()
    shown = displayed_table(at)
    row = shown.loc[at.dataframe[0].value["unit"] == 3].iloc[0]
    assert (row["RUL"], row["lower"], row["upper"]) == ("39.3", "23.4", "55.3")
    assert row["status"] == "▲ Warning" and row["predicted at cycle"] == "126"
    unpredicted = shown.loc[at.dataframe[0].value["unit"] == 9].iloc[0]
    assert (unpredicted["RUL"], unpredicted["lower"], unpredicted["upper"]) == ("–", "–", "–")


def test_rul_tooltips_use_one_decimal(app_test) -> None:
    spec = json.loads(app_test.run().get("vega_lite_chart")[0].proto.spec)
    tooltips = [
        layer["encoding"]["tooltip"]
        for layer in spec["layer"]
        if "tooltip" in layer.get("encoding", {})
    ]
    assert len(tooltips) == 2  # the shaded band and the points
    for tooltip in tooltips:
        formats = {t["field"]: t.get("format") for t in tooltip}
        assert formats == {"cycle": None, "rul": ".1f", "lower": ".1f", "upper": ".1f"}
