"""Data drift monitoring with the Population Stability Index (PSI).

**PSI.** Cut the reference distribution (all training readings) into 10 bins at its deciles, so
each bin holds about 10% of the reference. Then measure which share of the current readings
falls into each bin:

    PSI = sum over bins of (current share - reference share) * ln(current share / reference share)

Identical distributions give 0; the more probability mass moves between bins, the larger PSI
gets. Common rule of thumb: < 0.1 stable, 0.1-0.25 moderate, > 0.25 significant. An empty bin
would make ln(0) undefined, so shares are floored at a small epsilon.

**Caveat: degradation itself shifts sensor distributions.** The reference covers whole engine
lives, healthy to failure. The current window is each unit's *most recent* cycles. A fleet that
is late in life shows sensors shifted in the degradation direction, which raises sensor PSI with
no fault at all. So sensor PSI is a prompt to look, not proof of a problem. The operating
settings do not degrade, so drift there is a clearer sign of a real change, such as new flight
conditions or a misconfigured feed.

Monitored columns: the 3 operating settings plus the 14 kept sensors (ADR 0001).
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from predmaint.data.load import SETTING_COLUMNS
from predmaint.features.selection import KEPT_SENSORS

MONITORED_COLUMNS = [*SETTING_COLUMNS, *KEPT_SENSORS]
N_BINS = 10
EPSILON = 1e-4
MODERATE, SIGNIFICANT = 0.1, 0.25
DRIFT_STATUSES = ("stable", "moderate", "significant")


@dataclass(frozen=True)
class ColumnProfile:
    """Bins of one column: ``edges`` are the inner cut points (bins are [e_i, e_i+1), with
    open ends below the first and above the last edge); ``shares`` sum to 1."""

    edges: list[float]
    shares: list[float]
    n: int


@dataclass(frozen=True)
class ReferenceProfile:
    columns: dict[str, ColumnProfile]
    n_rows: int
    n_bins: int = N_BINS

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "ReferenceProfile":
        raw = json.loads(text)
        return cls(
            columns={c: ColumnProfile(**p) for c, p in raw["columns"].items()},
            n_rows=raw["n_rows"],
            n_bins=raw["n_bins"],
        )


def _bin_index(edges: Sequence[float], values: ArrayLike) -> np.ndarray:
    # side="right": a value equal to an edge goes to the bin above it, for reference and
    # current values alike, so discrete columns (e.g. integer-valued s_17) are binned consistently.
    return np.searchsorted(np.asarray(edges, dtype=float), np.asarray(values, dtype=float), "right")


def _shares(edges: Sequence[float], values: ArrayLike) -> np.ndarray:
    idx = _bin_index(edges, values)
    counts = np.bincount(idx, minlength=len(edges) + 1)
    return counts / counts.sum()


def build_reference_profile(
    train_df: pd.DataFrame, columns: Sequence[str] = MONITORED_COLUMNS, n_bins: int = N_BINS
) -> ReferenceProfile:
    """Decile bin edges and reference shares per column, from the training readings.

    Columns with few distinct values (e.g. ``setting_3``, always 100 in FD001) get fewer bins,
    because repeated quantiles collapse into one edge.
    """
    missing = [c for c in columns if c not in train_df.columns]
    if missing:
        raise ValueError(f"train_df is missing columns: {missing}")
    profiles = {}
    for column in columns:
        values = train_df[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or len(values) == 0:
            raise ValueError(f"column {column} must contain finite values")
        quantiles = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
        edges = np.unique(quantiles).tolist()
        profiles[column] = ColumnProfile(edges, _shares(edges, values).tolist(), len(values))
    return ReferenceProfile(profiles, n_rows=len(train_df), n_bins=n_bins)


def current_shares(profile: ColumnProfile, values: ArrayLike) -> list[float]:
    """Share of ``values`` in each of the reference bins."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("no current values")
    return _shares(profile.edges, values).tolist()


def psi(profile: ColumnProfile, current_values: ArrayLike, epsilon: float = EPSILON) -> float:
    """Population Stability Index of ``current_values`` against a column's reference bins."""
    ref = np.maximum(np.asarray(profile.shares, dtype=float), epsilon)
    cur = np.maximum(np.asarray(current_shares(profile, current_values)), epsilon)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def drift_status(value: float, moderate: float = MODERATE, significant: float = SIGNIFICANT) -> str:
    """ "stable" if PSI < 0.1, "moderate" if < 0.25, else "significant"."""
    if value < moderate:
        return "stable"
    if value < significant:
        return "moderate"
    return "significant"


def drift_report(profile: ReferenceProfile, current: pd.DataFrame) -> list[dict]:
    """PSI, status and bins for every profiled column present in ``current``."""
    rows = []
    for column, col_profile in profile.columns.items():
        values = current[column].dropna().to_numpy(dtype=float) if column in current else []
        if len(values) == 0:
            continue
        value = psi(col_profile, values)
        rows.append(
            {
                "column": column,
                "psi": value,
                "status": drift_status(value),
                "n_current": len(values),
                "bin_edges": col_profile.edges,
                "reference_shares": col_profile.shares,
                "current_shares": current_shares(col_profile, values),
            }
        )
    return rows
