"""Tests for the dashboard's pure logic (no Streamlit, no HTTP)."""

import pandas as pd
import pytest

from predmaint.dashboard.logic import (
    DEFAULT_THRESHOLDS,
    DRIFT_COLOURS,
    DRIFT_LABELS,
    FLEET_COLUMNS,
    STATUS_COLOURS,
    STATUS_LABELS,
    STATUSES,
    RiskThresholds,
    bin_labels,
    drift_table,
    fleet_kpis,
    fleet_table,
    histogram_frame,
    prediction_history,
    rank_units,
    risk_status,
)


def unit(
    unit_id: int, lower: float | None, last_cycle: int = 100, cycle: int | None = None
) -> dict:
    prediction = None
    if lower is not None:
        prediction = {
            "rul": lower + 15,
            "lower": lower,
            "upper": lower + 30,
            "cycle": cycle if cycle is not None else last_cycle,
            "created_at": "2026-11-18T10:00:00Z",
        }
    return {
        "unit_id": unit_id,
        "n_readings": last_cycle,
        "last_cycle": last_cycle,
        "latest_prediction": prediction,
    }


# --- risk_status -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lower", "status"),
    [
        (0.0, "critical"),
        (19.99, "critical"),
        (20.0, "warning"),
        (49.99, "warning"),
        (50.0, "ok"),
        (300.0, "ok"),
        (None, "no prediction"),
    ],
)
def test_risk_status_default_boundaries(lower: float | None, status: str) -> None:
    assert risk_status(lower) == status


def test_risk_status_custom_thresholds() -> None:
    t = RiskThresholds(critical=10, warning=30)
    assert [risk_status(x, t) for x in (9, 10, 29, 30)] == ["critical", "warning", "warning", "ok"]


@pytest.mark.parametrize(("critical", "warning"), [(50, 20), (30, 30), (-1, 10)])
def test_invalid_thresholds_are_rejected(critical: float, warning: float) -> None:
    with pytest.raises(ValueError, match="critical < warning"):
        RiskThresholds(critical=critical, warning=warning)


def test_defaults_and_every_status_has_label_and_colour() -> None:
    assert (DEFAULT_THRESHOLDS.critical, DEFAULT_THRESHOLDS.warning) == (20, 50)
    assert set(STATUS_LABELS) == set(STATUS_COLOURS) == set(STATUSES)


# --- ranking -------------------------------------------------------------------------------------


def test_rank_by_lower_bound_with_unpredicted_units_last() -> None:
    units = [unit(1, 80), unit(2, None), unit(3, 5), unit(4, 35), unit(5, None)]
    assert [u["unit_id"] for u in rank_units(units)] == [3, 4, 1, 2, 5]


def test_rank_breaks_ties_by_unit_id() -> None:
    assert [u["unit_id"] for u in rank_units([unit(9, 10), unit(2, 10), unit(5, 10)])] == [2, 5, 9]


def test_rank_uses_lower_bound_not_point_prediction() -> None:
    # Unit 1 has the higher point prediction but a wider interval reaching lower down.
    a = {
        "unit_id": 1,
        "last_cycle": 50,
        "latest_prediction": {"rul": 70, "lower": 25, "upper": 115, "cycle": 50},
    }
    b = {
        "unit_id": 2,
        "last_cycle": 50,
        "latest_prediction": {"rul": 60, "lower": 45, "upper": 75, "cycle": 50},
    }
    assert [u["unit_id"] for u in rank_units([b, a])] == [1, 2]


def test_rank_empty_fleet() -> None:
    assert rank_units([]) == []


# --- fleet table and KPIs ------------------------------------------------------------------------


def test_fleet_table_columns_order_status_and_staleness() -> None:
    units = [unit(1, 70), unit(2, 10, last_cycle=120, cycle=110), unit(3, None), unit(4, 40)]
    table = fleet_table(units)
    assert list(table.columns) == FLEET_COLUMNS
    assert table["unit"].tolist() == [2, 4, 1, 3]
    assert table["status"].tolist() == ["critical", "warning", "ok", "no prediction"]
    assert table["stale"].tolist() == [True, False, False, False]
    row = table.iloc[0]
    assert (row["RUL"], row["lower"], row["upper"], row["predicted at cycle"]) == (25, 10, 40, 110)
    assert pd.isna(table.iloc[3]["RUL"])


def test_fleet_table_respects_thresholds() -> None:
    table = fleet_table([unit(1, 40)], RiskThresholds(critical=45, warning=60))
    assert table["status"].tolist() == ["critical"]


def test_fleet_table_of_empty_fleet_has_columns() -> None:
    assert list(fleet_table([]).columns) == FLEET_COLUMNS


def test_kpis_count_statuses() -> None:
    table = fleet_table([unit(1, 5), unit(2, 15), unit(3, 30), unit(4, 99), unit(5, None)])
    assert fleet_kpis(table) == {"units": 5, "critical": 2, "warning": 1, "no prediction": 1}
    assert fleet_kpis(fleet_table([])) == {
        "units": 0,
        "critical": 0,
        "warning": 0,
        "no prediction": 0,
    }


# --- prediction history -------------------------------------------------------------------------


def test_prediction_history_sorted_by_cycle_newest_wins_per_cycle() -> None:
    predictions = [  # API order: newest first
        {
            "id": 4,
            "cycle": 30,
            "rul": 20,
            "lower": 10,
            "upper": 30,
            "created_at": "2026-11-18T12:00",
        },
        {
            "id": 3,
            "cycle": 20,
            "rul": 41,
            "lower": 31,
            "upper": 51,
            "created_at": "2026-11-18T11:00",
        },
        {
            "id": 2,
            "cycle": 20,
            "rul": 40,
            "lower": 30,
            "upper": 50,
            "created_at": "2026-11-18T10:30",
        },
        {
            "id": 1,
            "cycle": 10,
            "rul": 60,
            "lower": 50,
            "upper": 70,
            "created_at": "2026-11-18T10:00",
        },
    ]
    history = prediction_history(predictions)
    assert history["cycle"].tolist() == [10, 20, 30]
    assert history.loc[history["cycle"] == 20, "rul"].item() == 41  # the newer prediction
    assert list(history.columns) == ["cycle", "rul", "lower", "upper"]


def test_prediction_history_empty() -> None:
    assert prediction_history([]).empty


# --- data drift --------------------------------------------------------------------------------


def column(name: str, value: float, status: str) -> dict:
    return {
        "column": name,
        "psi": value,
        "status": status,
        "n_current": 300,
        "bin_edges": [1.0, 2.0],
        "reference_shares": [0.3, 0.4, 0.3],
        "current_shares": [0.1, 0.4, 0.5],
    }


def test_drift_table_sorted_by_psi_with_kind() -> None:
    report = {
        "columns": [
            column("s_4", 0.12, "moderate"),
            column("setting_1", 0.01, "stable"),
            column("s_11", 0.9, "significant"),
        ]
    }
    table = drift_table(report)
    assert table["column"].tolist() == ["s_11", "s_4", "setting_1"]
    assert table["kind"].tolist() == ["sensor", "sensor", "setting"]
    assert table["status"].tolist() == ["significant", "moderate", "stable"]
    assert drift_table({"columns": []}).empty


def test_drift_statuses_have_labels_and_colours() -> None:
    assert set(DRIFT_LABELS) == set(DRIFT_COLOURS) == {"stable", "moderate", "significant"}


def test_bin_labels() -> None:
    assert bin_labels([47.25, 47.5]) == ["< 47.25", "47.25 – 47.5", "≥ 47.5"]
    assert bin_labels([100.0]) == ["< 100", "≥ 100"]
    assert bin_labels([]) == ["all values"]


def test_histogram_frame_pairs_reference_and_current_per_bin() -> None:
    frame = histogram_frame(column("s_11", 0.9, "significant"))
    assert len(frame) == 6
    ref = frame[frame["source"] == "reference"]["share"].tolist()
    cur = frame[frame["source"] == "current"]["share"].tolist()
    assert ref == [0.3, 0.4, 0.3] and cur == [0.1, 0.4, 0.5]
    assert frame["bin"].tolist()[:3] == ["< 1", "1 – 2", "≥ 2"]
