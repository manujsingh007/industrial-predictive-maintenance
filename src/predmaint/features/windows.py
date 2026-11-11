"""Causal rolling-window features per engine.

Two leakage rules are enforced here:

1. **Causal windows.** The features at cycle *t* use only cycles <= *t* of the same engine:
   trailing windows, never centred or forward-looking ones. The test set stops engines before
   failure, so at prediction time only the past exists. A window that looked ahead would see
   the degradation still to come, giving CV scores that cannot be reproduced on the test set.
2. **No mixing of engines.** Windows are computed separately for each ``unit``. A window
   running across the boundary between two engines would blend one engine's final, near-failure
   cycles into the next engine's first cycles. Across CV folds, it would also carry data from
   training engines into validation engines.

Early cycles: windows shrink to the history available (``min_periods=1``). At cycle 5 a
30-cycle window uses cycles 1-5. At an engine's first cycle the window holds one point: the mean
is that value, and the std and slope are defined as 0 (no spread and no trend observed yet). So
the output has **no NaN**, and every row stays usable and scored.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

STATS = ("mean", "std", "slope")


def window_feature_names(sensors: Sequence[str], windows: Sequence[int]) -> list[str]:
    """Names of the columns added by :func:`add_window_features`, in order."""
    return [f"{s}_{stat}_{w}" for s in sensors for w in windows for stat in STATS]


def add_window_features(
    df: pd.DataFrame, sensors: Sequence[str], windows: Sequence[int] = (10, 30)
) -> pd.DataFrame:
    """Return a copy of ``df`` with causal rolling mean, std and slope per sensor and window.

    For each sensor ``s`` and window ``w``, three columns are added:

    - ``{s}_mean_{w}``: mean of the last ``w`` cycles (including the current one).
    - ``{s}_std_{w}``: population standard deviation (ddof=0) over the same cycles.
    - ``{s}_slope_{w}``: least-squares slope of ``s`` against ``cycle`` over the same
      cycles, in sensor units per cycle.

    Windows are computed per ``unit`` after sorting by ``cycle``. The result keeps the input's
    index and row order. See the module docstring for how early cycles are handled.

    Args:
        df: Rows with ``unit``, ``cycle`` and the sensor columns; index must be unique.
        sensors: Sensor columns to summarise.
        windows: Window lengths in cycles (positive integers).

    Raises:
        ValueError: On missing columns, non-positive windows, a non-unique index or
            duplicate (unit, cycle) rows.
    """
    missing = {"unit", "cycle", *sensors} - set(df.columns)
    if missing:
        raise ValueError(f"df is missing columns: {sorted(missing)}")
    if not windows or any(int(w) != w or w < 1 for w in windows):
        raise ValueError(f"windows must be positive integers, got {windows}")
    if not df.index.is_unique:
        raise ValueError("df must have a unique index")
    if df.duplicated(["unit", "cycle"]).any():
        raise ValueError("df has duplicate (unit, cycle) rows")

    ordered = df.sort_values(["unit", "cycle"])
    x = ordered["cycle"].astype(float)
    y = ordered[list(sensors)].astype(float)
    # Columns needed for the least-squares slope: slope = cov(x, y) / var(x), from window means.
    parts = pd.concat(
        [x.rename("x"), (x * x).rename("xx"), y, y.mul(x, axis=0).add_suffix("__xy")], axis=1
    )
    groups = parts.groupby(ordered["unit"], sort=False)
    history = ordered.groupby("unit").cumcount() + 1  # cycles seen so far, per engine

    new_columns: dict[str, pd.Series] = {}
    for w in windows:
        means = groups.rolling(int(w), min_periods=1).mean().droplevel(0)
        stds = groups[list(sensors)].rolling(int(w), min_periods=1).std(ddof=0).droplevel(0)
        # Decide "single point" by counting, not by testing var_x > 0: floating-point
        # rounding can leave a tiny non-zero variance in a one-point window.
        multi_point = (np.minimum(history, w) >= 2).reindex(means.index)
        var_x = means["xx"] - means["x"] ** 2
        safe_var_x = var_x.where(multi_point, 1.0)
        for s in sensors:
            cov_xy = means[f"{s}__xy"] - means["x"] * means[s]
            new_columns[f"{s}_mean_{w}"] = means[s]
            new_columns[f"{s}_std_{w}"] = stds[s].where(multi_point, 0.0)
            new_columns[f"{s}_slope_{w}"] = (cov_xy / safe_var_x).where(multi_point, 0.0)

    features = pd.DataFrame(new_columns).reindex(df.index)
    return pd.concat([df, features[window_feature_names(sensors, windows)]], axis=1)
