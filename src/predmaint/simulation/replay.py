"""Replay test-set engines into the running service, as if they were live engines.

Run with, for example:
``uv run python -m predmaint.simulation.replay --subset FD001 --units 1-10 --predict-every 10``

For each selected unit, readings are POSTed to ``/units/{id}/readings`` in cycle order
(``--batch-size`` readings per request, default 1). ``GET /units/{id}/prediction`` is called
after every ``--predict-every`` cycles and at the unit's last cycle.

**No label leakage.** The simulator reads only the test *trajectories* (``load_test``). It never
reads ``RUL_FD00x.txt``, so the service cannot see a single true label through it.

**Resumable and verified.** The service accepts a re-sent reading that is identical to a stored
one as a no-op, and rejects a different one with 409. So on (re)start the simulator does not
blindly skip stored cycles: it re-sends them in large batches, which *verifies* that what is
stored under this unit id really is this engine. It then continues with the new cycles. A mixed
history, e.g. another engine's readings under the same id, stops the run with a clear error.
If the last run stopped between storing a checkpoint cycle and predicting, that prediction is
requested after verification.

**Retries.** Connection errors and 5xx responses are retried with exponential backoff. Because
POSTs are idempotent, a retry simply re-sends the same batch: if the first attempt was stored
and only the response was lost, the re-send is a harmless no-op. Any other 4xx stops the run.
"""

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx
import pandas as pd

from predmaint.data.load import SENSOR_COLUMNS, SETTING_COLUMNS, load_test
from predmaint.features.selection import SUBSETS_WITH_ADR
from predmaint.serving.schemas import MAX_READINGS

READING_COLUMNS = ["cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS]


class ReplayError(RuntimeError):
    """The replay cannot continue (client error from the service, or retries exhausted)."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


VERIFY_BATCH = MAX_READINGS  # re-send already-stored cycles in as few requests as possible


@dataclass
class ReplayConfig:
    """Settings of one replay run."""

    base_url: str = "http://127.0.0.1:8000"
    predict_every: int = 10
    batch_size: int = 1
    delay: float = 0.0  # seconds to wait between requests ("live" pace)
    max_retries: int = 5
    backoff: float = 0.5  # first retry wait; doubles each attempt
    unit_offset: int = 0  # service unit id = test unit + offset

    def __post_init__(self) -> None:
        if self.predict_every < 1 or self.batch_size < 1:
            raise ValueError("predict_every and batch_size must be at least 1")
        if self.delay < 0 or self.max_retries < 0:
            raise ValueError("delay and max_retries must not be negative")


@dataclass
class UnitResult:
    unit_id: int
    n_cycles: int
    sent: int = 0  # new readings stored by this run
    verified: int = 0  # readings re-sent and confirmed identical to what was already stored
    requests: int = 0
    predictions: list[dict] = field(default_factory=list)


@dataclass
class ReplaySummary:
    units: list[UnitResult]
    retries: int
    seconds: float

    @property
    def readings_sent(self) -> int:
        return sum(u.sent for u in self.units)

    @property
    def predictions_made(self) -> int:
        return sum(len(u.predictions) for u in self.units)


class ServiceClient:
    """Thin HTTP client for the RUL service, with retries."""

    def __init__(self, http: httpx.Client, config: ReplayConfig, sleep: Callable = time.sleep):
        self.http, self.config, self.sleep = http, config, sleep
        self.retries = 0

    def _request(self, method: str, url: str, *, ok_404: bool = False, **kwargs) -> httpx.Response:
        """Send with retries on connection errors and 5xx; raise ReplayError on other 4xx."""
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.http.request(method, url, **kwargs)
            except httpx.TransportError as exc:
                problem = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code < 400 or (ok_404 and response.status_code == 404):
                    return response
                if response.status_code < 500:
                    raise ReplayError(
                        f"{method} {url} failed with {response.status_code}: {_detail(response)}",
                        status_code=response.status_code,
                    )
                problem = f"HTTP {response.status_code}: {_detail(response)}"
            if attempt == self.config.max_retries:
                break
            self.retries += 1
            wait = self.config.backoff * 2**attempt
            print(f"  retry {attempt + 1}/{self.config.max_retries} in {wait:.1f}s ({problem})")
            self.sleep(wait)  # then simply re-send: POSTs are idempotent
        raise ReplayError(
            f"{method} {url} failed after {self.config.max_retries} retries: {problem}"
        )

    def summary(self, unit_id: int) -> dict:
        r = self._request("GET", f"/units/{unit_id}", ok_404=True)
        if r.status_code == 404:
            return {"last_cycle": None, "last_prediction_cycle": None}
        return r.json()

    def post_readings(self, unit_id: int, readings: list[dict]) -> dict:
        return self._request(
            "POST", f"/units/{unit_id}/readings", json={"readings": readings}
        ).json()

    def predict(self, unit_id: int) -> dict:
        return self._request("GET", f"/units/{unit_id}/prediction").json()


def _detail(response: httpx.Response) -> str:
    try:
        return str(response.json().get("detail", response.text))[:300]
    except ValueError:
        return response.text[:300]


def _readings(df: pd.DataFrame) -> list[dict]:
    out = df[READING_COLUMNS].to_dict(orient="records")
    for r in out:
        r["cycle"] = int(r["cycle"])
    return out


def replay_unit(
    client: ServiceClient, test_unit: int, trajectory: pd.DataFrame, config: ReplayConfig
) -> UnitResult:
    """Send one unit's trajectory and request predictions; verify and resume after stored cycles."""
    unit_id = test_unit + config.unit_offset
    trajectory = trajectory.sort_values("cycle")
    last_cycle = int(trajectory["cycle"].iloc[-1])
    result = UnitResult(unit_id=unit_id, n_cycles=len(trajectory))

    def wait() -> None:
        if config.delay:
            client.sleep(config.delay)

    def is_checkpoint(cycle: int) -> bool:
        return cycle % config.predict_every == 0 or cycle == last_cycle

    status = client.summary(unit_id)
    result.requests += 1
    stored = status["last_cycle"] or 0
    if stored > last_cycle:
        raise ReplayError(
            f"unit {unit_id} already has readings up to cycle {stored}, but test engine "
            f"{test_unit} has only {last_cycle} cycles: the stored history belongs to another "
            "engine. Use --unit-offset or a fresh database."
        )

    def post(batch: list[dict]) -> dict:
        try:
            return client.post_readings(unit_id, batch)
        except ReplayError as exc:
            if exc.status_code == 409:
                raise ReplayError(
                    f"unit {unit_id}: the stored readings do not match test engine {test_unit} "
                    f"({exc}). The unit id may hold another engine's data; use --unit-offset "
                    "or a fresh database.",
                    status_code=409,
                ) from exc
            raise

    rows = _readings(trajectory)
    # 1. Verify what is already stored by re-sending it (identical readings are no-ops).
    already = [r for r in rows if r["cycle"] <= stored]
    for start in range(0, len(already), VERIFY_BATCH):
        answer = post(already[start : start + VERIFY_BATCH])
        result.verified += answer.get("n_already_stored", 0)
        result.sent += answer.get("n_stored", 0)  # gaps in the stored prefix get filled
        result.requests += 1

    # 2. A run that stopped after storing a checkpoint but before predicting: catch up.
    if stored and is_checkpoint(stored) and (status["last_prediction_cycle"] or 0) < stored:
        result.predictions.append(client.predict(unit_id))
        result.requests += 1

    # 3. New cycles, at the configured batch size and pace.
    pending = [r for r in rows if r["cycle"] > stored]
    for start in range(0, len(pending), config.batch_size):
        batch = pending[start : start + config.batch_size]
        wait()
        answer = post(batch)
        result.sent += answer.get("n_stored", len(batch))
        result.verified += answer.get("n_already_stored", 0)
        result.requests += 1
        if any(is_checkpoint(r["cycle"]) for r in batch):
            wait()
            result.predictions.append(client.predict(unit_id))
            result.requests += 1
    return result


def parse_units(spec: str) -> list[int]:
    """'1-10' -> 1..10; '1,4,7-9' -> 1, 4, 7, 8, 9 (sorted, unique)."""
    units: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            if lo > hi:
                raise ValueError(f"invalid range {part!r}")
            units.update(range(lo, hi + 1))
        else:
            units.add(int(part))
    if not units or min(units) < 1:
        raise ValueError(f"invalid unit selection {spec!r}")
    return sorted(units)


def run_replay(
    trajectories: pd.DataFrame,
    units: list[int],
    config: ReplayConfig,
    *,
    http: httpx.Client | None = None,
    sleep: Callable = time.sleep,
) -> ReplaySummary:
    """Replay the selected units of ``trajectories`` (unit, cycle, settings, sensors)."""
    missing = sorted(set(units) - set(trajectories["unit"].unique()))
    if missing:
        raise ReplayError(f"units not in the test data: {missing[:10]}")
    own_http = http is None
    http = http or httpx.Client(base_url=config.base_url, timeout=30.0)
    client = ServiceClient(http, config, sleep)
    started, results = time.perf_counter(), []
    try:
        for unit in units:
            r = replay_unit(client, unit, trajectories[trajectories["unit"] == unit], config)
            results.append(r)
            last = r.predictions[-1] if r.predictions else None
            latest = (
                f"latest RUL {last['rul']:.1f} [{last['lower']:.1f}, {last['upper']:.1f}] "
                f"at cycle {last['cycle']}"
                if last
                else "no new prediction"
            )
            print(
                f"unit {r.unit_id}: sent {r.sent} new of {r.n_cycles} cycles, verified "
                f"{r.verified} already stored, {len(r.predictions)} predictions, {latest}"
            )
    finally:
        if own_http:
            http.close()
    return ReplaySummary(results, client.retries, time.perf_counter() - started)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Replay test engines into the RUL service.")
    parser.add_argument("--subset", required=True, choices=SUBSETS_WITH_ADR)
    parser.add_argument("--units", default="1-10", help="e.g. 1-10 or 1,4,7-9 (default 1-10)")
    parser.add_argument("--predict-every", type=int, default=10, help="cycles between predictions")
    parser.add_argument("--batch-size", type=int, default=1, help="readings per POST (default 1)")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between requests")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--unit-offset", type=int, default=0, help="service id = test unit + offset"
    )
    args = parser.parse_args(argv)

    try:
        config = ReplayConfig(
            base_url=args.base_url,
            predict_every=args.predict_every,
            batch_size=args.batch_size,
            delay=args.delay,
            max_retries=args.max_retries,
            unit_offset=args.unit_offset,
        )
        units = parse_units(args.units)
    except ValueError as exc:
        parser.error(str(exc))

    trajectories = load_test(args.subset)  # trajectories only: true RUL is never read
    print(f"Replaying {len(units)} {args.subset} test units to {config.base_url} ...")
    try:
        summary = run_replay(trajectories, units, config)
    except ReplayError as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        return 1
    print(
        f"Done: {len(summary.units)} units, {summary.readings_sent} readings sent, "
        f"{summary.predictions_made} predictions, {summary.retries} retries, "
        f"{summary.seconds:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
