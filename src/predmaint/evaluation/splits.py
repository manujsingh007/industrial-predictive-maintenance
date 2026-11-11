"""Cross-validation splits by engine, as defined in docs/evaluation.md."""

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def engine_folds(
    df: pd.DataFrame, n_splits: int = 5, seed: int = 42
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Return an iterator of ``(train_idx, val_idx)`` for K-fold CV grouped by engine.

    All rows of an engine (column ``unit``) land in the same fold, so no engine is in both
    training and validation. Each engine is validated exactly once across the folds.

    Args:
        df: Rows to split; must contain a ``unit`` column.
        n_splits: Number of folds.
        seed: Shuffling seed; the same seed gives the same folds.

    Returns:
        Iterator of pairs of **positional** row-index arrays (use with ``df.iloc``).

    Raises:
        ValueError: If ``unit`` is missing or there are fewer engines than folds. Raised
            immediately when called, not lazily on first iteration.
    """
    if "unit" not in df.columns:
        raise ValueError("df must contain a 'unit' column")
    n_units = df["unit"].nunique()
    if n_units < n_splits:
        raise ValueError(f"need at least {n_splits} engines for {n_splits} folds, got {n_units}")
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return splitter.split(df, groups=df["unit"])
