"""Demo mode: export, snapshot client and the public entry point (mock API, no real network)."""

import json
import re
import socket
import subprocess
import sys
import tomllib
from pathlib import Path

import httpx
import pytest

from predmaint import config
from predmaint.dashboard import client as client_module
from predmaint.dashboard.client import ApiClient, ApiError
from predmaint.dashboard.snapshot import (
    READING_COLUMNS,
    SnapshotClient,
    from_columns,
    to_columns,
)
from predmaint.dashboard.source import DataSourceError, make_source
from predmaint.demo.export_snapshot import ExportError, collect, main, write_snapshot

ROOT = Path(__file__).parents[1]
ENTRY = ROOT / "demo" / "streamlit_app.py"
MODEL = {"model_name": "rul-lgbm-fd001", "model_version": "1"}
SENSORS = [f"s_{i}" for i in range(1, 22)]


def unit(unit_id: int, lower: float) -> dict:
    latest = {"rul": lower + 10, "lower": lower, "upper": lower + 20, "cycle": 30}
    latest["created_at"] = "2026-11-24T10:00:00Z"
    return {"unit_id": unit_id, "n_readings": 30, "last_cycle": 30, "latest_prediction": latest}


class FakeApi:
    """The API routes the dashboard uses, with adjustable state."""

    def __init__(self) -> None:
        self.units = [unit(1, 80.0), unit(2, 5.5), unit(7, 30.25)]
        self.prediction_model = dict(MODEL)
        self.drift_version = "1"
        self.methods: set[str] = set()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.methods.add(request.method)
        path, query = request.url.path, request.url.params
        if path == "/health":
            return httpx.Response(200, json={"status": "ok", **MODEL, "database": "ok"})
        if path == "/units":
            return httpx.Response(200, json=self.units)
        if path == "/monitoring/drift":
            return httpx.Response(200, json=self.drift(int(query["last_n_cycles"])))
        _, unit_id, what = path.strip("/").split("/")
        if int(unit_id) not in {u["unit_id"] for u in self.units}:
            return httpx.Response(404, json={"detail": f"unit {unit_id} is unknown"})
        rows = getattr(self, what)(int(unit_id))
        limit = int(query["limit"]) if "limit" in query else None
        if limit:
            rows = rows[:limit] if what == "predictions" else rows[-limit:]
        return httpx.Response(200, json=rows)

    def predictions(self, unit_id: int) -> list[dict]:  # newest first, like the API
        return [
            {
                "id": 100 * unit_id + c,
                "unit_id": unit_id,
                "cycle": c,
                "rul": 130.123456789 - c,
                "lower": 100.5 - c,
                "upper": 150.25 - c,
                **self.prediction_model,
                "created_at": f"2026-11-24T10:00:{c}Z",
            }
            for c in (30, 20, 10)
        ]

    def readings(self, unit_id: int) -> list[dict]:
        return [
            {
                "unit_id": unit_id,
                "cycle": c,
                "setting_1": 0.0,
                "setting_2": 0.0,
                "setting_3": 100.0,
                **{s: 100.0 + 0.1 * c for s in SENSORS},
                "ingested_at": "2026-11-24T10:00:00Z",
            }
            for c in range(1, 31)
        ]

    def drift(self, n: int) -> dict:
        column = {
            "column": "s_11",
            "psi": 0.07,
            "status": "stable",
            "n_current": 90,
            "bin_edges": [47.2],
            "reference_shares": [0.5, 0.5],
            "current_shares": [0.4, 0.6],
        }
        return {
            "last_n_cycles": n,
            "n_units": 3,
            "n_readings": 90,
            "reference_rows": 20631,
            "model_version": self.drift_version,
            "caveat": "degradation",
            "columns": [column],
        }


@pytest.fixture
def fake_api() -> FakeApi:
    return FakeApi()


@pytest.fixture
def api(fake_api: FakeApi, monkeypatch: pytest.MonkeyPatch) -> ApiClient:
    monkeypatch.setattr(client_module, "TRANSPORT", httpx.MockTransport(fake_api))
    return ApiClient("http://api")


@pytest.fixture
def snapshot_dir(api: ApiClient, tmp_path: Path) -> Path:
    write_snapshot(collect(api), tmp_path / "snapshot")
    return tmp_path / "snapshot"


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record and refuse every outgoing connection and name lookup."""
    attempts: list[str] = []

    def refuse(*args, **kwargs):
        attempts.append(repr(args[1:] or args))
        raise AssertionError(f"network access in demo mode: {args!r}")

    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, refuse)
    for name in ("create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, name, refuse)
    monkeypatch.setattr(httpx.Client, "send", refuse)
    return attempts


# --- same shapes as the API client -----------------------------------------------------------


def test_snapshot_client_returns_the_same_data_as_the_api_client(api, snapshot_dir) -> None:
    snap = SnapshotClient(snapshot_dir)
    assert snap.units() == api.units()
    assert snap.drift(30) == api.drift(30)
    assert snap.health().keys() == api.health().keys()
    assert {k: snap.health()[k] for k in MODEL} == MODEL
    for unit_id in (1, 2, 7):
        assert snap.predictions(unit_id) == api.predictions(unit_id)
        assert snap.predictions(unit_id, limit=2) == api.predictions(unit_id, limit=2)
        for limit in (None, 5):
            expected = [{c: r[c] for c in READING_COLUMNS} for r in api.readings(unit_id, limit)]
            assert snap.readings(unit_id, limit) == expected


def test_both_clients_offer_the_same_methods_and_attributes(api, snapshot_dir) -> None:
    snap = SnapshotClient(snapshot_dir)
    for name in (
        "label",
        "snapshot",
        "drift_window",
        "health",
        "units",
        "predictions",
        "readings",
        "drift",
    ):
        assert hasattr(api, name) and hasattr(snap, name), name
    assert api.snapshot is None and api.drift_window is None
    assert snap.snapshot["n_units"] == 3 and snap.drift_window == 30


def test_both_clients_report_errors_as_data_source_errors(api, snapshot_dir) -> None:
    for source in (api, SnapshotClient(snapshot_dir)):
        with pytest.raises(DataSourceError):
            source.predictions(999)
    assert issubclass(ApiError, DataSourceError)
    with pytest.raises(DataSourceError, match="last 30 readings only"):
        SnapshotClient(snapshot_dir).drift(10)


def test_column_format_round_trip() -> None:
    rows = [{"a": 1, "b": None}, {"a": 2, "b": "x"}]
    assert from_columns(to_columns(rows)) == rows
    assert from_columns(to_columns([])) == []


# --- choosing the source ---------------------------------------------------------------------


def test_source_is_the_api_by_default_and_the_snapshot_when_configured(snapshot_dir) -> None:
    assert isinstance(make_source(None), ApiClient)
    assert isinstance(make_source(snapshot_dir), SnapshotClient)


def test_environment_wins_over_secret_and_relative_paths_are_in_the_project(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PREDMAINT_HOME", str(tmp_path))
    monkeypatch.delenv("PREDMAINT_DEMO_SNAPSHOT", raising=False)
    assert config.demo_snapshot_dir() is None
    assert config.demo_snapshot_dir("demo/snapshot") == tmp_path.resolve() / "demo" / "snapshot"
    monkeypatch.setenv("PREDMAINT_DEMO_SNAPSHOT", "/elsewhere")
    assert config.demo_snapshot_dir("demo/snapshot") == Path("/elsewhere")


@pytest.mark.parametrize("damage", ["missing", "corrupt", "old format"])
def test_unusable_snapshot_fails_and_never_falls_back_to_the_api(
    snapshot_dir: Path, no_network: list[str], damage: str
) -> None:
    meta = snapshot_dir / "meta.json"
    if damage == "missing":
        meta.unlink()
    else:
        meta.write_text("{oops" if damage == "corrupt" else '{"format_version": 0}')
    with pytest.raises(DataSourceError):
        make_source(snapshot_dir)
    assert no_network == []


# --- export ----------------------------------------------------------------------------------


def test_export_only_reads_and_contains_no_labels(fake_api, snapshot_dir) -> None:
    assert fake_api.methods == {"GET"}
    keys: set[str] = set()

    def walk(value) -> None:
        if isinstance(value, dict):
            keys.update(value)
            value = list(value.values())
        if isinstance(value, list):
            for item in value:
                walk(item)

    for path in snapshot_dir.glob("*.json"):
        walk(json.loads(path.read_text()))
    assert not [k for k in keys if re.search(r"true|label|actual", k)]
    readings = json.loads((snapshot_dir / "readings.json").read_text())
    assert set(readings["1"]) == set(READING_COLUMNS)  # key sensors only


def test_export_refuses_predictions_or_drift_of_another_model(api, fake_api) -> None:
    fake_api.prediction_model["model_version"] = "2"
    with pytest.raises(ExportError, match="rul-lgbm-fd001 v2.*has rul-lgbm-fd001 v1 loaded"):
        collect(api)
    fake_api.prediction_model["model_version"] = "1"
    fake_api.drift_version = "9"
    with pytest.raises(ExportError, match="drift by v9"):
        collect(api)


def test_export_refuses_an_empty_fleet_and_oversized_output(api, fake_api, tmp_path) -> None:
    with pytest.raises(ExportError, match="limit 10"):
        write_snapshot(collect(api), tmp_path / "big", max_bytes=10)
    assert not (tmp_path / "big").exists()  # nothing written
    fake_api.units = []
    with pytest.raises(ExportError, match="no units"):
        collect(api)


def test_export_command_reports_size_or_explains_the_failure(
    api, fake_api, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = ["--output", str(tmp_path / "s"), "--base-url", "http://api"]
    assert main(out) == 0
    assert "Total 0.0" in capsys.readouterr().out
    fake_api.prediction_model["model_version"] = "2"
    assert main(["--output", str(tmp_path / "t"), "--base-url", "http://api"]) == 1
    assert "Snapshot not written" in capsys.readouterr().err
    assert not (tmp_path / "t").exists()


# --- the committed snapshot and requirements ----------------------------------------------------


def test_committed_snapshot_is_small_complete_and_consistent() -> None:
    directory = ROOT / "demo" / "snapshot"
    assert sum(p.stat().st_size for p in directory.glob("*.json")) < 5_000_000
    snap = SnapshotClient(directory)
    meta, units = snap.snapshot, snap.units()
    assert len(units) == meta["n_units"] == 100
    made_by = set()
    for u in units:
        rows, predictions = snap.readings(u["unit_id"]), snap.predictions(u["unit_id"])
        assert len(rows) == u["n_readings"] and rows[-1]["cycle"] == u["last_cycle"]
        assert predictions[0]["rul"] == u["latest_prediction"]["rul"]
        made_by |= {(p["model_name"], p["model_version"]) for p in predictions}
    assert made_by == {(meta["model_name"], meta["model_version"])}


def test_demo_requirements_are_minimal_and_match_the_lockfile() -> None:
    lines = (ROOT / "demo" / "requirements.txt").read_text().splitlines()
    pins = dict(ln.split("==") for ln in lines if ln and not ln.startswith("#"))
    assert set(pins) == {"streamlit", "pandas", "altair"}
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    locked = {p["name"]: p["version"] for p in lock["package"]}
    assert pins == {name: locked[name] for name in pins}


# --- the app in demo mode (needs the "dashboard" group) ------------------------------------------


@pytest.fixture
def run_demo(monkeypatch: pytest.MonkeyPatch):
    testing = pytest.importorskip("streamlit.testing.v1", reason="needs: uv sync --group dashboard")
    # Register the variable so monkeypatch removes what the entry point sets.
    monkeypatch.setenv("PREDMAINT_DEMO_SNAPSHOT", "")
    return lambda: testing.AppTest.from_file(str(ENTRY), default_timeout=60).run()


def test_demo_entry_point_renders_without_any_network_access(run_demo, no_network) -> None:
    at = run_demo()
    assert not at.exception and not at.error, (at.exception, at.error)
    assert len(at.get("vega_lite_chart")) == 6
    assert {m.label: m.value for m in at.metric}["Units monitored"] == "100"
    at.selectbox[0].select(3).run()  # a rerun must render the page again, not a blank one
    assert not at.exception and len(at.get("vega_lite_chart")) == 6
    assert no_network == []


def test_demo_banner_states_origin_date_model_and_links(run_demo) -> None:
    meta = json.loads((ROOT / "demo" / "snapshot" / "meta.json").read_text())
    at = run_demo()
    banner = at.info[0].value
    for text in (
        "static snapshot",
        "NASA C-MAPSS FD001",
        meta["generated_at"][:10],
        f"`{meta['model_name']}` v{meta['model_version']}",
        "#quickstart-with-docker",
        "https://github.com/manujsingh007/industrial-predictive-maintenance",
    ):
        assert text in banner
    assert len(at.button) == 0  # nothing to refresh
    assert at.number_input(key="drift_n").disabled


def test_live_mode_has_no_banner(api, monkeypatch: pytest.MonkeyPatch) -> None:
    testing = pytest.importorskip("streamlit.testing.v1", reason="needs: uv sync --group dashboard")
    monkeypatch.delenv("PREDMAINT_DEMO_SNAPSHOT", raising=False)
    app = ROOT / "src" / "predmaint" / "dashboard" / "app.py"
    at = testing.AppTest.from_file(str(app), default_timeout=60).run()
    assert not at.exception and not any("snapshot" in i.value for i in at.info)
    assert len(at.button) == 1 and not at.number_input(key="drift_n").disabled


def chart_spec(chart) -> dict:
    """The chart without its dataset names, which are only hashes of the data."""

    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k not in ("data", "datasets")}
        return [strip(v) for v in value] if isinstance(value, list) else value

    return strip(json.loads(chart.proto.spec))


def test_demo_and_live_app_show_the_same_fleet(
    api, snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    testing = pytest.importorskip("streamlit.testing.v1", reason="needs: uv sync --group dashboard")
    app = str(ROOT / "src" / "predmaint" / "dashboard" / "app.py")
    monkeypatch.delenv("PREDMAINT_DEMO_SNAPSHOT", raising=False)
    live = testing.AppTest.from_file(app, default_timeout=60).run()
    monkeypatch.setenv("PREDMAINT_DEMO_SNAPSHOT", str(snapshot_dir))
    demo = testing.AppTest.from_file(app, default_timeout=60).run()
    assert not live.exception and not demo.exception
    for i in (0, 1):  # fleet table, drift table
        assert live.dataframe[i].value.equals(demo.dataframe[i].value)
    assert [m.value for m in live.metric] == [m.value for m in demo.metric]
    specs = [[chart_spec(c) for c in at.get("vega_lite_chart")] for at in (live, demo)]
    assert len(specs[0]) == 6 and specs[0] == specs[1]


def test_streamlit_secret_selects_the_snapshot(
    snapshot_dir: Path, no_network, monkeypatch: pytest.MonkeyPatch
) -> None:
    testing = pytest.importorskip("streamlit.testing.v1", reason="needs: uv sync --group dashboard")
    monkeypatch.delenv("PREDMAINT_DEMO_SNAPSHOT", raising=False)
    app = ROOT / "src" / "predmaint" / "dashboard" / "app.py"
    at = testing.AppTest.from_file(str(app), default_timeout=60)
    at.secrets["PREDMAINT_DEMO_SNAPSHOT"] = str(snapshot_dir)
    at.run()
    assert not at.exception and "static snapshot" in at.info[0].value
    assert no_network == []


def test_demo_mode_works_without_httpx_and_the_heavy_packages_installed() -> None:
    """The demo environment lacks these; importing any of them there would crash the app."""
    code = f"""
import sys
for name in ("httpx", "mlflow", "lightgbm", "sklearn", "sqlalchemy", "psycopg", "fastapi", "torch"):
    sys.modules[name] = None  # makes "import name" fail
sys.path.insert(0, {str(ROOT / "src")!r})
from pathlib import Path
from predmaint.dashboard import logic
from predmaint.dashboard.source import make_source
source = make_source(Path({str(ROOT / "demo" / "snapshot")!r}))
table = logic.fleet_table(source.units())
logic.drift_table(source.drift(source.drift_window))
print(len(table))
"""
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "100"
