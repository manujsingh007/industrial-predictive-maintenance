"""Pure dashboard logic: risk status, ranking and tables. No Streamlit, no HTTP; fully unit-tested.

**Why alerts use the interval's lower bound, not the point prediction.** The 90% interval says
the true RUL is very likely above its lower bound: about 95% if misses split evenly between the
two sides. So "lower < 20" means "it is *plausible* that fewer than 20 cycles remain". A point
prediction of 60 with an interval of ±33 looks safe, but the engine could have 27 cycles left.
Alerting on the lower bound builds the model's own uncertainty into the alert. The error analysis
(docs/error_analysis.md) also found that late, dangerous errors cluster in mid-life, exactly where
a point-based alert would fire too late.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import pandas as pd

STATUSES = ("critical", "warning", "ok", "no prediction")
STATUS_LABELS = {
    "critical": "● Critical",
    "warning": "▲ Warning",
    "ok": "✓ OK",
    "no prediction": "– No prediction",
}
STATUS_COLOURS = {  # fixed status palette; always shown with a text label, never colour alone
    "critical": "#d03b3b",
    "warning": "#fab219",
    "ok": "#0ca30c",
    "no prediction": "#8a8984",
}


@dataclass(frozen=True)
class RiskThresholds:
    """Alert thresholds on the interval's lower bound, in cycles."""

    critical: float = 20.0
    warning: float = 50.0

    def __post_init__(self) -> None:
        if not 0 <= self.critical < self.warning:
            raise ValueError(
                f"need 0 <= critical < warning, got critical={self.critical}, "
                f"warning={self.warning}"
            )


DEFAULT_THRESHOLDS = RiskThresholds()


def risk_status(lower: float | None, thresholds: RiskThresholds = DEFAULT_THRESHOLDS) -> str:
    """ "critical" if lower < critical, "warning" if lower < warning, else "ok".

    ``None`` (no prediction yet) gives "no prediction".
    """
    if lower is None:
        return "no prediction"
    if lower < thresholds.critical:
        return "critical"
    if lower < thresholds.warning:
        return "warning"
    return "ok"


def rank_units(units: Iterable[Mapping]) -> list[Mapping]:
    """Most at-risk first: by lower bound ascending; units without a prediction last.

    Ties are broken by unit id, so the order is stable. ``units`` are rows of ``GET /units``.
    """

    def key(u: Mapping) -> tuple[int, float, int]:
        p = u.get("latest_prediction")
        return (0, float(p["lower"]), u["unit_id"]) if p else (1, 0.0, u["unit_id"])

    return sorted(units, key=key)


FLEET_COLUMNS = [
    "unit",
    "status",
    "last cycle",
    "RUL",
    "lower",
    "upper",
    "predicted at cycle",
    "stale",
]


def fleet_table(
    units: Iterable[Mapping], thresholds: RiskThresholds = DEFAULT_THRESHOLDS
) -> pd.DataFrame:
    """Ranked fleet table. ``stale`` is True when readings arrived after the latest prediction."""
    rows = []
    for u in rank_units(units):
        p = u.get("latest_prediction")
        rows.append(
            {
                "unit": u["unit_id"],
                "status": risk_status(p["lower"] if p else None, thresholds),
                "last cycle": u["last_cycle"],
                "RUL": p["rul"] if p else None,
                "lower": p["lower"] if p else None,
                "upper": p["upper"] if p else None,
                "predicted at cycle": p["cycle"] if p else None,
                "stale": bool(p) and p["cycle"] < u["last_cycle"],
            }
        )
    return pd.DataFrame(rows, columns=FLEET_COLUMNS)


def fleet_kpis(table: pd.DataFrame) -> dict[str, int]:
    """Counts for the KPI tiles."""
    counts = table["status"].value_counts()
    return {
        "units": len(table),
        "critical": int(counts.get("critical", 0)),
        "warning": int(counts.get("warning", 0)),
        "no prediction": int(counts.get("no prediction", 0)),
    }


def prediction_history(predictions: Iterable[Mapping]) -> pd.DataFrame:
    """A unit's predictions in cycle order, for the RUL chart.

    If the same cycle was predicted more than once (e.g. after a model update), the newest wins.
    """
    df = pd.DataFrame(
        list(predictions), columns=["cycle", "rul", "lower", "upper", "created_at", "id"]
    )
    if df.empty:
        return df[["cycle", "rul", "lower", "upper"]]
    df = df.sort_values(["cycle", "created_at", "id"]).drop_duplicates("cycle", keep="last")
    return df[["cycle", "rul", "lower", "upper"]].reset_index(drop=True)


# --- data drift ------------------------------------------------------------------------------

DRIFT_LABELS = {"stable": "✓ Stable", "moderate": "▲ Moderate", "significant": "● Significant"}
DRIFT_COLOURS = {
    "stable": STATUS_COLOURS["ok"],
    "moderate": STATUS_COLOURS["warning"],
    "significant": STATUS_COLOURS["critical"],
}
DRIFT_COLUMNS = ["column", "kind", "PSI", "status", "readings"]


def drift_table(report: Mapping) -> pd.DataFrame:
    """One row per monitored column, highest PSI first. ``kind`` separates operating settings
    (a clearer drift signal) from sensors (also shifted by normal degradation)."""
    rows = [
        {
            "column": c["column"],
            "kind": "setting" if c["column"].startswith("setting") else "sensor",
            "PSI": c["psi"],
            "status": c["status"],
            "readings": c["n_current"],
        }
        for c in report.get("columns", [])
    ]
    table = pd.DataFrame(rows, columns=DRIFT_COLUMNS)
    return table.sort_values(["PSI", "column"], ascending=[False, True]).reset_index(drop=True)


def bin_labels(edges: Iterable[float]) -> list[str]:
    """Readable labels for the bins defined by inner ``edges``: "< a", "a – b", ..., "≥ z"."""
    e = [f"{x:.4g}" for x in edges]
    if not e:
        return ["all values"]
    return [f"< {e[0]}", *(f"{a} – {b}" for a, b in zip(e, e[1:], strict=False)), f"≥ {e[-1]}"]


def histogram_frame(column_drift: Mapping) -> pd.DataFrame:
    """Long table for a grouped bar chart: bin, order, source ("reference"/"current"), share."""
    labels = bin_labels(column_drift["bin_edges"])
    rows = []
    for source, key in (("reference", "reference_shares"), ("current", "current_shares")):
        rows += [
            {"bin": label, "order": i, "source": source, "share": share}
            for i, (label, share) in enumerate(zip(labels, column_drift[key], strict=True))
        ]
    return pd.DataFrame(rows, columns=["bin", "order", "source", "share"])
