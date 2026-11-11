"""Sensor selection and RUL cap for FD001.

Decided in docs/adr/0001-rul-cap-and-sensor-selection.md. These choices apply to FD001 only;
other subsets need their own analysis and ADR.
"""

import numpy as np
import pandas as pd

from predmaint.data.load import SETTING_COLUMNS

SUBSETS_WITH_ADR = ("FD001",)

DEFAULT_RUL_CAP = 125

KEPT_SENSORS = [
    "s_2",
    "s_3",
    "s_4",
    "s_7",
    "s_8",
    "s_9",  # flagged in ADR 0001: only 71% of engines follow the fleet direction
    "s_11",
    "s_12",
    "s_13",
    "s_14",  # flagged in ADR 0001: only 58% of engines follow the fleet direction
    "s_15",
    "s_17",
    "s_20",
    "s_21",
]

CONSTANT_SENSORS = ["s_1", "s_5", "s_10", "s_16", "s_18", "s_19"]
NEAR_ZERO_VARIANCE_SENSORS = ["s_6"]
DROPPED_SENSORS = CONSTANT_SENSORS + NEAR_ZERO_VARIANCE_SENSORS
DROPPED_COLUMNS = DROPPED_SENSORS + SETTING_COLUMNS


def cap_rul[T: (pd.Series, np.ndarray)](rul: T, cap: float = DEFAULT_RUL_CAP) -> T:
    """Clip RUL values at ``cap``: ``min(rul, cap)``. Used for the training target only.

    Evaluation always compares predictions against the uncapped true RUL (docs/evaluation.md).

    Args:
        rul: RUL values (Series or array); returned as the same type.
        cap: Upper limit; must be positive.

    Raises:
        ValueError: If ``cap`` is not positive.
    """
    if cap <= 0:
        raise ValueError(f"cap must be positive, got {cap}")
    return np.minimum(rul, cap)
