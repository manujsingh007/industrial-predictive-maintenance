"""Tests for PSI-based drift monitoring (synthetic data)."""

import math

import numpy as np
import pandas as pd
import pytest

from predmaint.monitoring.drift import (
    EPSILON,
    MONITORED_COLUMNS,
    ReferenceProfile,
    build_reference_profile,
    current_shares,
    drift_report,
    drift_status,
    psi,
)


@pytest.fixture
def reference() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.normal(100, 2, 20_000), "b": rng.integers(390, 400, 20_000)})


def test_monitors_three_settings_and_fourteen_sensors() -> None:
    assert len(MONITORED_COLUMNS) == 17
    assert MONITORED_COLUMNS[:3] == ["setting_1", "setting_2", "setting_3"]


def test_reference_bins_are_deciles(reference: pd.DataFrame) -> None:
    profile = build_reference_profile(reference, ["a"]).columns["a"]
    assert len(profile.edges) == 9 and len(profile.shares) == 10
    np.testing.assert_allclose(profile.shares, 0.1, atol=0.002)
    assert sum(profile.shares) == pytest.approx(1.0)


def test_identical_distribution_gives_psi_near_zero(reference: pd.DataFrame) -> None:
    profile = build_reference_profile(reference, ["a"]).columns["a"]
    same = np.random.default_rng(1).normal(100, 2, 20_000)  # fresh sample, same distribution
    assert psi(profile, same) < 0.01
    assert psi(profile, reference["a"]) == pytest.approx(0.0, abs=1e-9)


def test_shifted_distribution_gives_psi_above_025(reference: pd.DataFrame) -> None:
    profile = build_reference_profile(reference, ["a"]).columns["a"]
    shifted = np.random.default_rng(2).normal(101.5, 2, 5_000)  # shift of 0.75 std
    value = psi(profile, shifted)
    assert value > 0.25 and drift_status(value) == "significant"


def test_psi_grows_with_the_shift(reference: pd.DataFrame) -> None:
    profile = build_reference_profile(reference, ["a"]).columns["a"]
    rng = np.random.default_rng(3)
    values = [psi(profile, rng.normal(100 + s, 2, 5_000)) for s in (0.0, 0.5, 1.0, 2.0)]
    assert values == sorted(values)


def test_hand_calculated_psi_with_epsilon_for_empty_bins() -> None:
    # Two bins (edge 0): reference 50/50; all current values in the upper bin.
    profile = build_reference_profile(pd.DataFrame({"x": [-1.0, 1.0] * 50}), ["x"], n_bins=2)
    profile = profile.columns["x"]
    assert profile.shares == [0.5, 0.5]
    expected = (EPSILON - 0.5) * math.log(EPSILON / 0.5) + (1 - 0.5) * math.log(1 / 0.5)
    assert psi(profile, [2.0, 3.0, 4.0]) == pytest.approx(expected)
    assert math.isfinite(expected)


def test_constant_column_and_discrete_values(reference: pd.DataFrame) -> None:
    profile = build_reference_profile(pd.DataFrame({"c": [100.0] * 500}), ["c"]).columns["c"]
    assert profile.edges == [100.0] and profile.shares == [0.0, 1.0]  # collapsed bins
    assert psi(profile, [100.0] * 40) == pytest.approx(0.0, abs=1e-9)
    assert psi(profile, [101.0] * 40) == pytest.approx(0.0, abs=1e-9)  # still in the top bin
    assert drift_status(psi(profile, [99.0] * 40)) == "significant"
    discrete = build_reference_profile(reference, ["b"]).columns["b"]
    assert psi(discrete, reference["b"]) == pytest.approx(0.0, abs=1e-9)  # ties binned consistently


@pytest.mark.parametrize(
    ("value", "status"),
    [
        (0.0, "stable"),
        (0.0999, "stable"),
        (0.1, "moderate"),
        (0.2499, "moderate"),
        (0.25, "significant"),
        (3.0, "significant"),
    ],
)
def test_status_thresholds(value: float, status: str) -> None:
    assert drift_status(value) == status


def test_profile_json_round_trip(reference: pd.DataFrame) -> None:
    profile = build_reference_profile(reference, ["a", "b"])
    restored = ReferenceProfile.from_json(profile.to_json())
    assert restored == profile
    assert restored.n_rows == 20_000


def test_drift_report_rows_and_bins(reference: pd.DataFrame) -> None:
    profile = build_reference_profile(reference, ["a", "b"])
    current = pd.DataFrame({"a": np.random.default_rng(4).normal(103, 2, 300)})  # no "b"
    rows = drift_report(profile, current)
    assert [r["column"] for r in rows] == ["a"]  # columns missing from current are skipped
    row = rows[0]
    assert row["n_current"] == 300 and row["status"] == "significant"
    assert len(row["current_shares"]) == len(row["reference_shares"]) == len(row["bin_edges"]) + 1
    assert sum(row["current_shares"]) == pytest.approx(1.0)


def test_invalid_inputs(reference: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        build_reference_profile(reference, ["zzz"])
    with pytest.raises(ValueError, match="finite"):
        build_reference_profile(pd.DataFrame({"x": [1.0, np.nan]}), ["x"])
    profile = build_reference_profile(reference, ["a"]).columns["a"]
    with pytest.raises(ValueError, match="no current values"):
        current_shares(profile, [])
