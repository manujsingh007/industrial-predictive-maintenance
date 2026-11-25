"""Export a static snapshot of the running system for the public demo.

Run with: ``python -m predmaint.demo.export_snapshot`` (docker compose stack running, test
engines replayed). It only sends GET requests to the API, which holds no true RUL.

The demo states publicly which model produced its numbers, so the export refuses to write a
snapshot that would make this untrue: every stored prediction and the drift report must come from
the model the API has loaded.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from predmaint import config
from predmaint.dashboard.client import ApiClient
from predmaint.dashboard.snapshot import FORMAT_VERSION, READING_COLUMNS, to_columns
from predmaint.dashboard.source import DataSourceError

DRIFT_WINDOW = 30
MAX_BYTES = 5_000_000
PREDICTION_LIMIT = 1000  # the API's maximum


class ExportError(RuntimeError):
    """The running system's state would make a misleading or oversized snapshot."""


def collect(client: ApiClient, drift_window: int = DRIFT_WINDOW) -> dict[str, object]:
    """Read everything the dashboard shows; return the content of each snapshot file."""
    health = client.health()
    if health.get("status") != "ok":
        raise ExportError("the API has no model loaded")
    model = (health["model_name"], health["model_version"])
    units = client.units()
    if not units:
        raise ExportError("the API has no units; replay some engines first")

    predictions, readings, made_by = {}, {}, set()
    for unit in units:
        key = str(unit["unit_id"])
        rows = client.predictions(unit["unit_id"], limit=PREDICTION_LIMIT)
        if len(rows) >= PREDICTION_LIMIT:
            raise ExportError(f"unit {key} may have more than {PREDICTION_LIMIT} predictions")
        made_by |= {(r["model_name"], r["model_version"]) for r in rows}
        predictions[key] = to_columns(rows)
        readings[key] = to_columns(client.readings(unit["unit_id"]), READING_COLUMNS)
    drift = client.drift(drift_window)

    if made_by != {model} or drift["model_version"] != model[1]:
        found = ", ".join(f"{n} v{v}" for n, v in sorted(made_by)) or "none"
        raise ExportError(
            f"stored predictions were made by [{found}] and drift by v{drift['model_version']}, "
            f"but the API has {model[0]} v{model[1]} loaded. Replay into an empty database with "
            "this model, then export again."
        )
    meta = {
        "format_version": FORMAT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_name": model[0],
        "model_version": model[1],
        "drift_window": drift_window,
        "n_units": len(units),
        "n_readings": sum(len(t["cycle"]) for t in readings.values()),
        "n_predictions": sum(len(t["cycle"]) for t in predictions.values()),
    }
    return {
        "meta": meta,
        "units": units,
        "predictions": predictions,
        "readings": readings,
        "drift": drift,
    }


def write_snapshot(content: dict[str, object], directory: Path, max_bytes: int = MAX_BYTES) -> int:
    """Write one compact JSON file per entry; return the total size. Too large writes nothing."""
    files = {name: json.dumps(v, separators=(",", ":")) + "\n" for name, v in content.items()}
    size = sum(len(text.encode()) for text in files.values())
    if size > max_bytes:
        raise ExportError(f"snapshot would be {size:,} bytes (limit {max_bytes:,})")
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (directory / f"{name}.json").write_text(text, encoding="utf-8")
    return size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a demo snapshot from the running API.")
    parser.add_argument("--base-url", default=None, help="API URL (default: API_BASE_URL)")
    parser.add_argument("--output", type=Path, default=config.project_root() / "demo" / "snapshot")
    args = parser.parse_args(argv)
    client = ApiClient(args.base_url, timeout=30.0)
    try:
        content = collect(client)
        size = write_snapshot(content, args.output)
    except (ExportError, DataSourceError) as exc:
        print(f"Snapshot not written: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()
    meta = content["meta"]
    print(
        f"{meta['n_units']} units, {meta['n_readings']} readings, {meta['n_predictions']} "
        f"predictions by {meta['model_name']} v{meta['model_version']}"
    )
    for path in sorted(args.output.glob("*.json")):
        print(f"  {path.name:18} {path.stat().st_size / 1e6:6.3f} MB")
    print(f"Total {size / 1e6:.3f} MB in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
