"""Tests for predmaint.features.selection."""

import numpy as np
import pandas as pd
import pytest

from predmaint.data.load import SENSOR_COLUMNS, SETTING_COLUMNS
from predmaint.features.selection import (
    DEFAULT_RUL_CAP,
    DROPPED_COLUMNS,
    DROPPED_SENSORS,
    KEPT_SENSORS,
    cap_rul,
)


def test_cap_rul_clips_only_values_above_cap() -> None:
    rul = pd.Series([0, 100, 125, 126, 300], index=[10, 11, 12, 13, 14])
    capped = cap_rul(rul)
    assert capped.tolist() == [0, 100, 125, 125, 125]
    assert capped.index.tolist() == [10, 11, 12, 13, 14]  # index preserved for alignment


def test_cap_rul_default_is_125() -> None:
    assert DEFAULT_RUL_CAP == 125
    assert cap_rul(np.array([200]))[0] == 125


def test_cap_rul_custom_cap_and_numpy_input() -> None:
    out = cap_rul(np.array([50, 110, 150]), cap=110)
    assert isinstance(out, np.ndarray)
    assert out.tolist() == [50, 110, 110]


def test_cap_rul_does_not_modify_input() -> None:
    rul = pd.Series([10, 200])
    cap_rul(rul)
    assert rul.tolist() == [10, 200]


@pytest.mark.parametrize("cap", [0, -5])
def test_cap_rul_rejects_non_positive_cap(cap: int) -> None:
    with pytest.raises(ValueError):
        cap_rul(pd.Series([1, 2]), cap=cap)


def test_sensor_lists_match_adr_0001() -> None:
    assert len(KEPT_SENSORS) == 14
    assert len(DROPPED_SENSORS) == 7
    assert set(KEPT_SENSORS).isdisjoint(DROPPED_SENSORS)
    assert set(KEPT_SENSORS) | set(DROPPED_SENSORS) == set(SENSOR_COLUMNS)
    assert set(SETTING_COLUMNS) <= set(DROPPED_COLUMNS)
