"""Tests for the replay simulator against a fake in-memory service (httpx.MockTransport)."""

import json
from collections.abc import Callable

import httpx
import numpy as np
import pandas as pd
import pytest

from predmaint.data.load import SENSOR_COLUMNS, SETTING_COLUMNS
from predmaint.simulation import replay
from predmaint.simulation.replay import ReplayConfig, ReplayError, parse_units, run_replay

VALUE_COLUMNS = [*SETTING_COLUMNS, *SENSOR_COLUMNS]


class FakeService:
    """Mimics the /units API rules.

    POST is idempotent but strict: a reading identical to a stored one is a no-op, a different
    one gives 409 naming the cycles, and batches are all or nothing. GET /units/{id} is 404 if
    the unit is unknown.
    """

    def __init__(self) -> None:
        self.readings: dict[int, dict[int, dict]] = {}  # unit -> cycle -> reading
        self.predicted_at: dict[int, list[int]] = {}
        self.log: list[str] = []  # "GET /units/1", "POST 1 [3, 4]", "PREDICT 1@10"
        self.fail: Callable[[httpx.Request], httpx.Response | None] | None = None

    @property
    def cycles(self) -> dict[int, list[int]]:
        return {u: sorted(r) for u, r in self.readings.items()}

    def preload(self, data: pd.DataFrame, unit: int, cycles: range, as_unit: int | None = None):
        """Store rows of ``data`` (as if an earlier run had sent them)."""
        rows = data[(data["unit"] == unit) & data["cycle"].isin(cycles)]
        store = self.readings.setdefault(unit if as_unit is None else as_unit, {})
        for r in rows[["cycle", *VALUE_COLUMNS]].to_dict(orient="records"):
            store[int(r["cycle"])] = {**r, "cycle": int(r["cycle"])}

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.fail is not None:
            injected = self.fail(request)
            if injected is not None:
                return injected
        parts = request.url.path.strip("/").split("/")  # ["units", "1", ...]
        unit = int(parts[1])
        stored = self.readings.setdefault(unit, {})
        if request.method == "GET" and len(parts) == 2:
            self.log.append(f"GET /units/{unit}")
            if not stored and not self.predicted_at.get(unit):
                return httpx.Response(404, json={"detail": f"unit {unit} is unknown"})
            preds = self.predicted_at.get(unit, [])
            return httpx.Response(
                200,
                json={
                    "unit_id": unit,
                    "n_readings": len(stored),
                    "last_cycle": max(stored, default=None),
                    "n_predictions": len(preds),
                    "last_prediction_cycle": preds[-1] if preds else None,
                },
            )
        if request.method == "POST":
            batch = json.loads(request.content)["readings"]
            self.log.append(f"POST {unit} {[r['cycle'] for r in batch]}")
            conflicts = [
                r["cycle"] for r in batch if r["cycle"] in stored and stored[r["cycle"]] != r
            ]
            if conflicts:
                return httpx.Response(
                    409,
                    json={"detail": f"unit {unit} already has different readings for {conflicts}"},
                )
            new = [r for r in batch if r["cycle"] not in stored]
            for r in new:
                stored[r["cycle"]] = r
            return httpx.Response(
                201, json={"n_stored": len(new), "n_already_stored": len(batch) - len(new)}
            )
        if not stored:
            return httpx.Response(404, json={"detail": "no stored readings"})
        last = max(stored)
        self.predicted_at.setdefault(unit, []).append(last)
        self.log.append(f"PREDICT {unit}@{last}")
        return httpx.Response(200, json={"cycle": last, "rul": 50.0, "lower": 40.0, "upper": 60.0})

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler), base_url="http://test")


def trajectories(lengths: dict[int, int], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for unit, n in lengths.items():
        f = pd.DataFrame(rng.normal(size=(n, 24)), columns=VALUE_COLUMNS)
        f.insert(0, "unit", unit)
        f.insert(1, "cycle", np.arange(1, n + 1))
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def run(service: FakeService, lengths: dict[int, int], data: pd.DataFrame | None = None, **config):
    sleeps: list[float] = []
    data = trajectories(lengths) if data is None else data
    summary = run_replay(
        data, sorted(lengths), ReplayConfig(**config), http=service.client(), sleep=sleeps.append
    )
    summary.sleeps = sleeps  # type: ignore[attr-defined]
    return summary


# --- order and cadence -------------------------------------------------------------------------


def test_one_reading_per_request_in_cycle_order_with_prediction_cadence() -> None:
    service = FakeService()
    run(service, {1: 25}, predict_every=10)
    expected = ["GET /units/1"]
    for c in range(1, 26):
        expected.append(f"POST 1 [{c}]")
        if c in (10, 20, 25):  # every 10 cycles and at the last cycle
            expected.append(f"PREDICT 1@{c}")
    assert service.log == expected


def test_batches_predict_after_the_batch_containing_a_checkpoint() -> None:
    service = FakeService()
    run(service, {1: 22}, predict_every=10, batch_size=4)
    assert service.log == [
        "GET /units/1",
        "POST 1 [1, 2, 3, 4]",
        "POST 1 [5, 6, 7, 8]",
        "POST 1 [9, 10, 11, 12]",
        "PREDICT 1@12",  # batch contains cycle 10
        "POST 1 [13, 14, 15, 16]",
        "POST 1 [17, 18, 19, 20]",
        "PREDICT 1@20",
        "POST 1 [21, 22]",
        "PREDICT 1@22",  # last cycle
    ]


def test_final_prediction_not_duplicated_when_last_cycle_is_a_checkpoint() -> None:
    service = FakeService()
    run(service, {1: 20}, predict_every=10)
    assert service.predicted_at[1] == [10, 20]


def test_units_are_replayed_one_after_another_with_offset() -> None:
    service = FakeService()
    summary = run(service, {3: 5, 1: 4}, predict_every=100, unit_offset=1000)
    units_in_order = [line.split()[1] for line in service.log if line.startswith("POST")]
    assert units_in_order == ["1001"] * 4 + ["1003"] * 5
    assert summary.readings_sent == 9 and summary.predictions_made == 2


def test_delay_is_applied_between_requests() -> None:
    service = FakeService()
    summary = run(service, {1: 3}, predict_every=10, delay=0.25)
    assert summary.sleeps == [0.25] * 4  # 3 POSTs + 1 final prediction


# --- resume: stored cycles are verified by re-sending them, not blindly skipped ----------------


def test_resume_verifies_stored_cycles_then_continues() -> None:
    data, service = trajectories({1: 25}), FakeService()
    service.preload(data, 1, range(1, 14))  # an earlier run stored cycles 1-13
    service.predicted_at[1] = [10]
    summary = run(service, {1: 25}, data, predict_every=10)
    posted = [line for line in service.log if line.startswith("POST")]
    assert posted[0] == f"POST 1 {list(range(1, 14))}"  # one verification request
    assert posted[1] == "POST 1 [14]" and len(posted) == 1 + 12
    assert service.predicted_at[1] == [10, 20, 25]
    assert summary.units[0].verified == 13 and summary.units[0].sent == 12


def test_resume_catches_up_a_missed_checkpoint_prediction_after_verifying() -> None:
    data, service = trajectories({1: 25}), FakeService()
    service.preload(data, 1, range(1, 21))  # stopped after storing 20, before predicting
    service.predicted_at[1] = [10]
    run(service, {1: 25}, data, predict_every=10)
    assert service.log[:4] == [
        "GET /units/1",
        f"POST 1 {list(range(1, 21))}",
        "PREDICT 1@20",
        "POST 1 [21]",
    ]


def test_completed_unit_is_only_verified() -> None:
    data, service = trajectories({1: 25}), FakeService()
    service.preload(data, 1, range(1, 26))
    service.predicted_at[1] = [10, 20, 25]
    summary = run(service, {1: 25}, data, predict_every=10)
    assert service.log == ["GET /units/1", f"POST 1 {list(range(1, 26))}"]
    assert summary.units[0].sent == 0 and summary.units[0].verified == 25


def test_interrupted_run_resumes_to_the_same_end_state() -> None:
    data, service = trajectories({1: 25}), FakeService()
    calls = {"n": 0}

    def crash_midway(request: httpx.Request) -> httpx.Response | None:
        calls["n"] += 1
        if calls["n"] == 9:
            return httpx.Response(400, json={"detail": "simulated crash"})
        return None

    service.fail = crash_midway
    with pytest.raises(ReplayError):
        run(service, {1: 25}, data, predict_every=10)
    service.fail = None
    run(service, {1: 25}, data, predict_every=10)  # second run verifies, then continues
    assert service.cycles[1] == list(range(1, 26))
    assert service.predicted_at[1][-1] == 25


# --- mixed history is detected -------------------------------------------------------------------


def test_another_engines_history_under_the_same_id_stops_the_replay() -> None:
    # Unit 1's id already holds the first 13 cycles of a DIFFERENT engine (e.g. a training
    # engine). Skipping by last_cycle would silently append test cycles 14-25 to it.
    test_data, other_engine = trajectories({1: 25}, seed=0), trajectories({1: 25}, seed=99)
    service = FakeService()
    service.preload(other_engine, 1, range(1, 14))
    with pytest.raises(ReplayError, match="stored readings do not match test engine 1") as err:
        run(service, {1: 25}, test_data, predict_every=10)
    assert err.value.status_code == 409
    assert "--unit-offset" in str(err.value)
    assert service.cycles[1] == list(range(1, 14))  # nothing appended
    assert not any(line.startswith("POST 1 [14") for line in service.log)


def test_more_stored_cycles_than_the_engine_has_stops_the_replay() -> None:
    data, service = trajectories({1: 10}), FakeService()
    service.preload(trajectories({1: 40}, seed=5), 1, range(1, 41))
    with pytest.raises(ReplayError, match="readings up to cycle 40, but test engine 1 has only 10"):
        run(service, {1: 10}, data)
    assert service.log == ["GET /units/1"]


# --- errors and retries --------------------------------------------------------------------------


def test_client_error_stops_with_clear_message() -> None:
    service = FakeService()
    service.fail = lambda r: (
        httpx.Response(422, json={"detail": "sensor values must be finite"})
        if r.method == "POST"
        else None
    )
    with pytest.raises(ReplayError, match="POST /units/1/readings failed with 422: sensor values"):
        run(service, {1: 5})
    assert service.log == ["GET /units/1"]  # nothing after the failed request


def test_connection_errors_are_retried() -> None:
    service = FakeService()
    failures = {"left": 2}

    def flaky(request: httpx.Request) -> httpx.Response | None:
        if request.method == "POST" and failures["left"]:
            failures["left"] -= 1
            raise httpx.ConnectError("connection refused", request=request)
        return None

    service.fail = flaky
    summary = run(service, {1: 3}, predict_every=10, backoff=0.1)
    assert service.cycles[1] == [1, 2, 3]
    assert summary.retries == 2
    assert summary.sleeps[:2] == [0.1, 0.2]  # exponential backoff


def test_retry_after_lost_response_is_a_harmless_no_op() -> None:
    # The service stores cycle 2 but the response never arrives. The client simply re-sends;
    # because POSTs are idempotent, the re-send is accepted as "already stored".
    data, service = trajectories({1: 3}), FakeService()
    lost = {"done": False}

    def lose_response(request: httpx.Request) -> httpx.Response | None:
        body = json.loads(request.content) if request.method == "POST" else None
        if body and body["readings"][0]["cycle"] == 2 and not lost["done"]:
            lost["done"] = True
            service.handler(request)  # committed on the server ...
            raise httpx.ReadError("connection reset", request=request)  # ... response lost
        return None

    service.fail = lose_response
    summary = run(service, {1: 3}, data, predict_every=10, backoff=0)
    assert service.cycles[1] == [1, 2, 3]
    assert service.log.count("POST 1 [2]") == 2  # original + identical re-send
    assert summary.units[0].sent == 2 and summary.units[0].verified == 1
    assert summary.retries == 1


def test_server_errors_are_retried_then_give_up() -> None:
    service = FakeService()
    service.fail = lambda r: httpx.Response(503, json={"detail": "Database unavailable"})
    with pytest.raises(ReplayError, match="after 3 retries: HTTP 503: Database unavailable"):
        run(service, {1: 2}, max_retries=3, backoff=0)


# --- CLI pieces ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "units"),
    [("1-10", list(range(1, 11))), ("1,4,7-9", [1, 4, 7, 8, 9]), (" 3 , 3,2", [2, 3])],
)
def test_parse_units(spec: str, units: list[int]) -> None:
    assert parse_units(spec) == units


@pytest.mark.parametrize("spec", ["", "5-2", "0-3", "a"])
def test_parse_units_rejects_bad_specs(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_units(spec)


def test_unknown_test_unit_is_rejected() -> None:
    with pytest.raises(ReplayError, match="not in the test data: \\[9\\]"):
        run_replay(trajectories({1: 3}), [1, 9], ReplayConfig(), http=FakeService().client())


def test_main_reads_trajectories_only_never_true_rul(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import predmaint.data.load as load_module

    def forbidden(*args, **kwargs):
        raise AssertionError("the replay must never read true RUL labels")

    monkeypatch.setattr(load_module, "load_test_rul", forbidden)
    monkeypatch.setattr(replay, "load_test", lambda subset: trajectories({1: 12, 2: 3}))
    service = FakeService()
    real_client = httpx.Client
    monkeypatch.setattr(
        replay.httpx,
        "Client",
        lambda **kw: real_client(
            transport=httpx.MockTransport(service.handler), base_url="http://t"
        ),
    )
    assert replay.main(["--subset", "FD001", "--units", "1-2", "--predict-every", "5"]) == 0
    out = capsys.readouterr().out
    assert "unit 1: sent 12 new of 12 cycles" in out and "Done: 2 units, 15 readings sent" in out
    assert not hasattr(replay, "load_test_rul")  # not even imported


# --- drift injection -----------------------------------------------------------------------------


def test_inject_drift_only_changes_selected_units_and_column() -> None:
    from predmaint.simulation.replay import inject_drift

    data = trajectories({1: 5, 2: 5, 3: 5})
    out = inject_drift(data, "s_11", 0.5, [2, 3])
    changed = out["unit"].isin([2, 3])
    np.testing.assert_allclose(out.loc[changed, "s_11"], data.loc[changed, "s_11"] + 0.5)
    pd.testing.assert_series_equal(out.loc[~changed, "s_11"], data.loc[~changed, "s_11"])
    other = [c for c in data.columns if c != "s_11"]
    pd.testing.assert_frame_equal(out[other], data[other])  # every other column untouched
    assert data.loc[changed, "s_11"].equals(trajectories({1: 5, 2: 5, 3: 5}).loc[changed, "s_11"])


@pytest.mark.parametrize("spec", ["s_11", "s_99:1", "cycle:1", "s_11:abc", ":1"])
def test_parse_drift_rejects_bad_specs(spec: str) -> None:
    from predmaint.simulation.replay import parse_drift

    with pytest.raises(ValueError):
        parse_drift(spec)


def test_parse_drift() -> None:
    from predmaint.simulation.replay import parse_drift

    assert parse_drift("s_11:0.5") == ("s_11", 0.5)
    assert parse_drift("setting_1:-0.002") == ("setting_1", -0.002)


def test_main_injects_drift_into_sent_readings_and_says_so(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = trajectories({1: 3, 2: 3, 3: 3})
    monkeypatch.setattr(replay, "load_test", lambda subset: data.copy())
    service = FakeService()
    real_client = httpx.Client
    monkeypatch.setattr(
        replay.httpx,
        "Client",
        lambda **kw: real_client(
            transport=httpx.MockTransport(service.handler), base_url="http://t"
        ),
    )
    args = [
        "--subset",
        "FD001",
        "--units",
        "1-3",
        "--inject-drift",
        "s_4:2.5",
        "--drift-units",
        "2-3",
    ]
    assert replay.main(args) == 0
    assert "DRIFT INJECTED: s_4 +2.5 on 2 unit(s) 2-3" in capsys.readouterr().out
    for unit in (1, 2, 3):
        original = data[data["unit"] == unit].set_index("cycle")
        for cycle, sent in service.readings[unit].items():
            expected = original.at[cycle, "s_4"] + (2.5 if unit in (2, 3) else 0.0)
            assert sent["s_4"] == pytest.approx(expected)
            assert sent["s_11"] == pytest.approx(original.at[cycle, "s_11"])


def test_drift_units_without_inject_drift_is_an_error() -> None:
    with pytest.raises(SystemExit):
        replay.main(["--subset", "FD001", "--drift-units", "1-3"])
