"""Where the dashboard gets its data: the live API (default) or a static snapshot (demo).

Both sources offer the same methods and return the same shapes, so the app and ``logic.py`` do
not know which one they are using. The sources are imported only when chosen: demo mode never
imports ``httpx``, so it has no HTTP client to make a network call with.
"""

from pathlib import Path
from typing import Any, Protocol

KEY_SENSORS = ["s_11", "s_4", "s_12", "s_7"]  # strongest relation to RUL (notebook 01)


class DataSourceError(RuntimeError):
    """The data source failed; the message is shown to the user."""


class DashboardSource(Protocol):
    label: str  # shown in the sidebar
    snapshot: dict[str, Any] | None  # snapshot metadata in demo mode, None for the live API
    drift_window: int | None  # the only available drift window, or None if it can be chosen

    def health(self) -> dict: ...
    def units(self) -> list[dict]: ...
    def predictions(self, unit_id: int, limit: int = 1000) -> list[dict]: ...
    def readings(self, unit_id: int, limit: int | None = None) -> list[dict]: ...
    def drift(self, last_n_cycles: int = 30) -> dict: ...


def make_source(snapshot_dir: Path | None) -> DashboardSource:
    """Snapshot source if a directory is configured, otherwise the API.

    A configured but unusable snapshot raises; it never falls back to the network.
    """
    if snapshot_dir is not None:
        from predmaint.dashboard.snapshot import SnapshotClient

        return SnapshotClient(snapshot_dir)
    from predmaint.dashboard.client import ApiClient

    return ApiClient()
