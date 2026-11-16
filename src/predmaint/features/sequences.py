"""Fixed-length, causal input sequences per engine for sequence models (numpy only).

For each row (engine *u*, cycle *t*) the sequence holds the ``window`` most recent rows of the
same engine, ending at *t*: cycles t - window + 1 .. t, oldest first. The same two leakage
rules as the window features apply: only past and current cycles, and never another engine.

**Left padding.** When an engine has fewer than ``window`` cycles up to *t*, the missing
leading positions are filled with copies of the engine's **first observed cycle**. For example,
with ``window=5`` the sequence at cycle 2 is ``[c1, c1, c1, c1, c2]``. Repeating the first cycle
means "the engine looked like this before we started observing it", which suits engines that
start healthy. It also keeps padded values in the same range as real readings, unlike zeros, which
after standardisation would mean "an average engine of the fleet".
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def build_sequences(
    units: ArrayLike, cycles: ArrayLike, values: ArrayLike, window: int
) -> NDArray[np.float32]:
    """Build one causal, left-padded sequence per row.

    Args:
        units: Engine id of each row, shape (n,).
        cycles: Cycle of each row, shape (n,); orders the rows within an engine.
        values: Feature values, shape (n, n_features).
        window: Sequence length in cycles.

    Returns:
        Array of shape (n, window, n_features) in the **same row order as the input**;
        position ``window - 1`` is the row's own cycle.

    Raises:
        ValueError: On mismatched lengths, a window below 1 or duplicate (unit, cycle) rows.
    """
    units = np.asarray(units)
    cycles = np.asarray(cycles)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not (len(units) == len(cycles) == len(values)):
        raise ValueError("units, cycles and values must describe the same rows; values is 2-D")
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")

    order = np.lexsort((cycles, units))  # sort by unit, then cycle
    sorted_units, sorted_cycles = units[order], cycles[order]
    if np.any((sorted_units[1:] == sorted_units[:-1]) & (sorted_cycles[1:] == sorted_cycles[:-1])):
        raise ValueError("duplicate (unit, cycle) rows")

    out = np.empty((len(values), window, values.shape[1]), dtype=np.float32)
    starts = np.flatnonzero(np.r_[True, sorted_units[1:] != sorted_units[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for start, end in zip(starts, ends, strict=True):
        rows = order[start:end]
        engine = values[rows]  # this engine only, oldest cycle first
        padded = np.concatenate([np.repeat(engine[:1], window - 1, axis=0), engine])
        # sliding_window_view -> (n_rows, n_features, window); move time to axis 1
        windows = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)
        out[rows] = windows.transpose(0, 2, 1)
    return out
