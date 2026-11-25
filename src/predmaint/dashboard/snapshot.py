"""Snapshot format and the read-only client for it (standard library only, no network).

Files: ``meta.json`` (what, when, which model), ``units.json``, ``predictions.json``,
``readings.json`` (cycle and the 4 key sensors only) and ``drift.json``. Lists of rows are stored
by column, which avoids repeating every key on every row; the client turns them back into the
rows the API returns.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from predmaint.dashboard.source import KEY_SENSORS, DataSourceError

FORMAT_VERSION = 1
READING_COLUMNS = ["unit_id", "cycle", *KEY_SENSORS]


def to_columns(rows: Sequence[Mapping], columns: Sequence[str] | None = None) -> dict[str, list]:
    names = list(columns) if columns else (list(rows[0]) if rows else [])
    return {name: [row[name] for row in rows] for name in names}


def from_columns(table: Mapping[str, list]) -> list[dict]:
    return [dict(zip(table, values, strict=True)) for values in zip(*table.values(), strict=True)]


class SnapshotClient:
    """Same methods and shapes as ``ApiClient``, answered from files.

    Two limits of a snapshot: readings carry only ``READING_COLUMNS``, and drift exists for one
    window only (``drift_window``).
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.snapshot: dict[str, Any] = self._load("meta")
        if self.snapshot.get("format_version") != FORMAT_VERSION:
            raise DataSourceError(f"Unsupported snapshot format in {directory}; export it again.")
        self.drift_window: int = self.snapshot["drift_window"]
        self.label = "static snapshot"

    def _load(self, name: str) -> Any:
        path = self.directory / f"{name}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DataSourceError(
                f"Cannot read the demo snapshot file {path} ({type(exc).__name__})."
            ) from exc

    def _unit(self, name: str, unit_id: int) -> list[dict]:
        table = self._load(name).get(str(unit_id))
        if table is None:
            raise DataSourceError(f"Unit {unit_id} is not in the snapshot.")
        return from_columns(table)

    def health(self) -> dict:
        return {
            "status": "ok",
            "model_name": self.snapshot["model_name"],
            "model_version": self.snapshot["model_version"],
            "database": "not used (static snapshot)",
        }

    def units(self) -> list[dict]:
        return self._load("units")

    def predictions(self, unit_id: int, limit: int = 1000) -> list[dict]:
        return self._unit("predictions", unit_id)[:limit]  # newest first, like the API

    def readings(self, unit_id: int, limit: int | None = None) -> list[dict]:
        rows = self._unit("readings", unit_id)
        return rows[-limit:] if limit else rows  # most recent, in cycle order, like the API

    def drift(self, last_n_cycles: int = 30) -> dict:
        if last_n_cycles != self.drift_window:
            raise DataSourceError(
                f"The snapshot holds drift for the last {self.drift_window} readings only."
            )
        return self._load("drift")
