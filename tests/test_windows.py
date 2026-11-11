"""Tests for predmaint.features.windows and the run_cv feature sets (synthetic data only)."""

import numpy as np
import pandas as pd
import pytest

from predmaint.features.selection import KEPT_SENSORS
from predmaint.features.windows import add_window_features, window_feature_names
from predmaint.training.run_cv import build_features

SENSORS = ["a", "b"]
WINDOWS = (3, 5)
FEATURES = window_feature_names(SENSORS, WINDOWS)


def make_engine(unit: int, n_cycles: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "unit": unit,
            "cycle": np.arange(1, n_cycles + 1),
            "a": rng.normal(100, 5, n_cycles),
            "b": rng.normal(0, 1, n_cycles),
        }
    )


@pytest.fixture
def fleet() -> pd.DataFrame:
    df = pd.concat(
        [make_engine(1, 20, 0), make_engine(2, 12, 1), make_engine(3, 8, 2)], ignore_index=True
    )
    return df.sample(frac=1, random_state=0)  # shuffled rows: engines interleaved


# --- (a) causality ------------------------------------------------------------------


@pytest.mark.parametrize("changed_cycle", [1, 7, 20])
def test_changing_a_future_row_never_changes_earlier_features(
    fleet: pd.DataFrame, changed_cycle: int
) -> None:
    before = add_window_features(fleet, SENSORS, WINDOWS)
    modified = fleet.copy()
    row = (modified["unit"] == 1) & (modified["cycle"] == changed_cycle)
    modified.loc[row, SENSORS] += 1_000.0
    after = add_window_features(modified, SENSORS, WINDOWS)

    earlier = (fleet["unit"] == 1) & (fleet["cycle"] < changed_cycle)
    pd.testing.assert_frame_equal(after.loc[earlier, FEATURES], before.loc[earlier, FEATURES])
    # sanity check: the change does show up at the changed cycle itself
    assert not after.loc[row, FEATURES].equals(before.loc[row, FEATURES])


# --- (b) no mixing of engines -------------------------------------------------------


def test_engine_features_do_not_depend_on_other_engines(fleet: pd.DataFrame) -> None:
    alone = add_window_features(make_engine(2, 12, 1), SENSORS, WINDOWS)
    in_fleet = add_window_features(fleet, SENSORS, WINDOWS)
    in_fleet = in_fleet[in_fleet["unit"] == 2].sort_values("cycle")
    np.testing.assert_allclose(in_fleet[FEATURES].to_numpy(), alone[FEATURES].to_numpy())


def test_changing_another_engine_leaves_features_unchanged(fleet: pd.DataFrame) -> None:
    before = add_window_features(fleet, SENSORS, WINDOWS)
    modified = fleet.copy()
    modified.loc[modified["unit"] == 1, SENSORS] *= -50.0
    after = add_window_features(modified, SENSORS, WINDOWS)
    others = fleet["unit"] != 1
    pd.testing.assert_frame_equal(after.loc[others, FEATURES], before.loc[others, FEATURES])


# --- (c) slope ----------------------------------------------------------------------


@pytest.mark.parametrize("gradient", [3.0, -0.25, 0.0])
def test_slope_of_linear_series_equals_gradient(gradient: float) -> None:
    df = pd.DataFrame({"unit": 1, "cycle": np.arange(1, 41)})
    df["a"] = 9000.0 + gradient * df["cycle"]  # large offset, like the real sensor s_9
    out = add_window_features(df, ["a"], (10, 30))
    for w in (10, 30):
        slope = out[f"a_slope_{w}"]
        assert slope.iloc[0] == 0.0  # first cycle: one point, no trend (documented)
        np.testing.assert_allclose(slope.iloc[1:], gradient, atol=1e-8)


def test_slope_uses_cycle_as_time_axis() -> None:
    # cycles 2, 4, 6 with values 0, 10, 20: slope is 5 per cycle, not 10 per row
    df = pd.DataFrame({"unit": 1, "cycle": [2, 4, 6], "a": [0.0, 10.0, 20.0]})
    assert add_window_features(df, ["a"], (3,))["a_slope_3"].iloc[-1] == pytest.approx(5.0)


def test_mean_and_std_hand_calculated() -> None:
    df = pd.DataFrame({"unit": 1, "cycle": [1, 2, 3, 4, 5], "a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = add_window_features(df, ["a"], (3,))
    # cycle 5, window 3 -> values 3, 4, 5: mean 4, population std sqrt(2/3)
    assert out["a_mean_3"].iloc[-1] == pytest.approx(4.0)
    assert out["a_std_3"].iloc[-1] == pytest.approx(np.sqrt(2 / 3))
    # cycle 2, window 3 -> shrinks to the 2 available values 1, 2: mean 1.5, std 0.5
    assert out["a_mean_3"].iloc[1] == pytest.approx(1.5)
    assert out["a_std_3"].iloc[1] == pytest.approx(0.5)


# --- (d) columns and NaN ------------------------------------------------------------


def test_output_columns_index_and_order(fleet: pd.DataFrame) -> None:
    out = add_window_features(fleet, SENSORS, WINDOWS)
    assert list(out.columns) == list(fleet.columns) + FEATURES
    assert len(FEATURES) == len(SENSORS) * len(WINDOWS) * 3
    assert out.index.equals(fleet.index)  # same rows, same order
    pd.testing.assert_frame_equal(out[fleet.columns], fleet)  # inputs untouched


def test_no_nan_and_first_cycle_values_as_documented(fleet: pd.DataFrame) -> None:
    out = add_window_features(fleet, SENSORS, WINDOWS)
    assert not out[FEATURES].isna().any().any()
    first = out[out["cycle"] == 1]
    for s in SENSORS:
        for w in WINDOWS:
            np.testing.assert_allclose(first[f"{s}_mean_{w}"], first[s])  # mean = the value
            assert (first[f"{s}_std_{w}"] == 0).all()
            assert (first[f"{s}_slope_{w}"] == 0).all()


def test_window_of_one_has_zero_std_and_slope(fleet: pd.DataFrame) -> None:
    out = add_window_features(fleet, ["a"], (1,))
    np.testing.assert_allclose(out["a_mean_1"], out["a"])
    assert (out["a_std_1"] == 0).all()
    assert (out["a_slope_1"] == 0).all()


# --- validation ----------------------------------------------------------------------


@pytest.mark.parametrize("windows", [(), (0,), (-3,), (2.5,)])
def test_invalid_windows_raise(fleet: pd.DataFrame, windows: tuple) -> None:
    with pytest.raises(ValueError, match="windows"):
        add_window_features(fleet, SENSORS, windows)


def test_duplicate_unit_cycle_raises(fleet: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        add_window_features(pd.concat([fleet, fleet.iloc[:1]], ignore_index=True), SENSORS)


def test_missing_sensor_raises(fleet: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing"):
        add_window_features(fleet, ["zzz"])


# --- run_cv feature sets ---------------------------------------------------------------


def test_build_features_sets() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(30, len(KEPT_SENSORS))), columns=KEPT_SENSORS)
    df.insert(0, "unit", np.repeat([1, 2, 3], 10))
    df.insert(1, "cycle", np.tile(np.arange(1, 11), 3))

    data, names = build_features(df, "current")
    assert names == KEPT_SENSORS and data is df

    data, names = build_features(df, "window", (10, 30))
    assert len(names) == 14 + 14 * 3 * 2
    assert set(names) <= set(data.columns)
    assert not data[names].isna().any().any()

    with pytest.raises(ValueError, match="unknown feature set"):
        build_features(df, "bogus")
