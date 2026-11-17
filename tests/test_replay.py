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


class FakeService:
    """Mimics the /units API rules: all-or-nothing batches, 409 on duplicates, 404 if unknown."""

    def __init__(self) -> None:
        self.cycles: dict[int, list[int]] = {}
        self.predicted_at: dict[int, list[int]] = {}
        self.log: list[str] = []  # "GET /units/1", "POST 1 [3, 4]", "PREDICT 1@10"
        self.fail: Callable[[httpx.Request], httpx.Response | None] | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.fail is not None:
            injected = self.fail(request)
            if injected is not None:
                return injected
        parts = request.url.path.strip("/").split("/")  # ["units", "1", ...]
        unit = int(parts[1])
        stored = self.cycles.setdefault(unit, [])
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
            cycles = [r["cycle"] for r in json.loads(request.content)["readings"]]
            self.log.append(f"POST {unit} {cycles}")
            clash = sorted(set(cycles) & set(stored))
            if clash:
                return httpx.Response(409, json={"detail": f"already has cycles {clash}"})
            stored.extend(cycles)
            return httpx.Response(201, json={"n_total": len(stored)})
        if not stored:
            return httpx.Response(404, json={"detail": "no stored readings"})
        last = max(stored)
        self.predicted_at.setdefault(unit, []).append(last)
        self.log.append(f"PREDICT {unit}@{last}")
        return httpx.Response(200, json={"cycle": last, "rul": 50.0, "lower": 40.0, "upper": 60.0})

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler), base_url="http://test")


def trajectories(lengths: dict[int, int]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frames = []
    for unit, n in lengths.items():
        f = pd.DataFrame(rng.normal(size=(n, 24)), columns=[*SETTING_COLUMNS, *SENSOR_COLUMNS])
        f.insert(0, "unit", unit)
        f.insert(1, "cycle", np.arange(1, n + 1))
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def run(service: FakeService, lengths: dict[int, int], **config) -> replay.ReplaySummary:
    sleeps: list[float] = []
    summary = run_replay(
        trajectories(lengths),
        sorted(lengths),
        ReplayConfig(**config),
        http=service.client(),
        sleep=sleeps.append,
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


# --- resume ------------------------------------------------------------------------------------


def test_resume_skips_already_stored_cycles() -> None:
    service = FakeService()
    service.cycles[1] = list(range(1, 14))  # an earlier run stored cycles 1-13
    service.predicted_at[1] = [10]
    summary = run(service, {1: 25}, predict_every=10)
    posted = [line for line in service.log if line.startswith("POST")]
    assert posted[0] == "POST 1 [14]" and len(posted) == 12  # cycles 14-25 only
    assert service.predicted_at[1] == [10, 20, 25]
    assert summary.units[0].skipped == 13 and summary.units[0].sent == 12


def test_resume_catches_up_a_missed_checkpoint_prediction() -> None:
    service = FakeService()
    service.cycles[1] = list(range(1, 21))  # stopped after storing 20, before predicting
    service.predicted_at[1] = [10]
    run(service, {1: 25}, predict_every=10)
    assert service.log[:3] == ["GET /units/1", "PREDICT 1@20", "POST 1 [21]"]


def test_completed_unit_sends_nothing() -> None:
    service = FakeService()
    service.cycles[1] = list(range(1, 26))
    service.predicted_at[1] = [10, 20, 25]
    summary = run(service, {1: 25}, predict_every=10)
    assert service.log == ["GET /units/1"]
    assert summary.units[0].sent == 0 and summary.units[0].skipped == 25


def test_interrupted_run_resumes_to_the_same_end_state() -> None:
    service = FakeService()
    calls = {"n": 0}

    def crash_midway(request: httpx.Request) -> httpx.Response | None:
        calls["n"] += 1
        if calls["n"] == 9:
            return httpx.Response(400, json={"detail": "simulated crash"})
        return None

    service.fail = crash_midway
    with pytest.raises(ReplayError):
        run(service, {1: 25}, predict_every=10)
    service.fail = None
    run(service, {1: 25}, predict_every=10)  # second run picks up where the first stopped
    assert service.cycles[1] == list(range(1, 26))
    assert service.predicted_at[1][-1] == 25


# --- errors and retries ------------------------------------------------------------------------


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


def test_lost_response_after_commit_is_not_resent() -> None:
    # The service stores cycle 2 but the response never arrives. The retry must notice the
    # batch is stored (via GET /units/1) instead of re-sending it and failing with 409.
    service = FakeService()
    lost = {"done": False}

    def lose_response(request: httpx.Request) -> httpx.Response | None:
        if (
            request.method == "POST"
            and b'"cycle":2' in request.content.replace(b" ", b"")
            and not lost["done"]
        ):
            lost["done"] = True
            service.cycles.setdefault(1, []).append(2)  # committed on the server
            raise httpx.ReadError("connection reset", request=request)
        return None

    service.fail = lose_response
    run(service, {1: 3}, predict_every=10, backoff=0)
    assert service.cycles[1] == [1, 2, 3]
    assert service.log.count("POST 1 [2]") == 0  # never re-sent


def test_server_errors_are_retried_then_give_up() -> None:
    service = FakeService()
    service.fail = lambda r: httpx.Response(503, json={"detail": "Database unavailable"})
    with pytest.raises(ReplayError, match="after 3 retries: HTTP 503: Database unavailable"):
        run(service, {1: 2}, max_retries=3, backoff=0)


# --- CLI pieces --------------------------------------------------------------------------------


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
    assert "unit 1: sent 12 of 12 cycles" in out and "Done: 2 units, 15 readings sent" in out
    assert not hasattr(replay, "load_test_rul")  # not even imported
