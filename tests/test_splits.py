"""Tests for predmaint.evaluation.splits."""

import numpy as np
import pandas as pd
import pytest

from predmaint.evaluation.splits import engine_folds


@pytest.fixture
def fleet() -> pd.DataFrame:
    """23 engines with different lengths, rows shuffled, non-default index."""
    rng = np.random.default_rng(0)
    rows = [(unit, cycle) for unit in range(1, 24) for cycle in range(1, rng.integers(5, 30))]
    df = pd.DataFrame(rows, columns=["unit", "cycle"])
    df = df.sample(frac=1, random_state=1)  # shuffle rows: units are not contiguous
    df.index = df.index + 1000  # positional != label index
    return df


def test_yields_n_splits_folds(fleet: pd.DataFrame) -> None:
    assert len(list(engine_folds(fleet, n_splits=5))) == 5


def test_no_unit_in_both_train_and_val(fleet: pd.DataFrame) -> None:
    for train_idx, val_idx in engine_folds(fleet):
        train_units = set(fleet["unit"].iloc[train_idx])
        val_units = set(fleet["unit"].iloc[val_idx])
        assert train_units.isdisjoint(val_units)


def test_every_unit_validated_exactly_once(fleet: pd.DataFrame) -> None:
    val_units = [u for _, val_idx in engine_folds(fleet) for u in set(fleet["unit"].iloc[val_idx])]
    assert sorted(val_units) == sorted(fleet["unit"].unique())


def test_every_row_is_in_train_or_val(fleet: pd.DataFrame) -> None:
    for train_idx, val_idx in engine_folds(fleet):
        assert len(train_idx) + len(val_idx) == len(fleet)
        assert set(train_idx).isdisjoint(val_idx)


def test_same_seed_gives_same_folds(fleet: pd.DataFrame) -> None:
    a = list(engine_folds(fleet, seed=42))
    b = list(engine_folds(fleet, seed=42))
    for (ta, va), (tb, vb) in zip(a, b, strict=True):
        np.testing.assert_array_equal(ta, tb)
        np.testing.assert_array_equal(va, vb)


def test_different_seed_gives_different_folds(fleet: pd.DataFrame) -> None:
    def fold_units(seed: int) -> list[frozenset]:
        return [frozenset(fleet["unit"].iloc[v]) for _, v in engine_folds(fleet, seed=seed)]

    assert set(fold_units(1)) != set(fold_units(2))


def test_missing_unit_column_raises() -> None:
    with pytest.raises(ValueError, match="unit"):
        engine_folds(pd.DataFrame({"cycle": [1, 2, 3]}))


def test_too_few_engines_raises_immediately() -> None:
    df = pd.DataFrame({"unit": [1, 1, 2, 3], "cycle": [1, 2, 1, 1]})
    with pytest.raises(ValueError, match="at least 5 engines"):
        engine_folds(df, n_splits=5)
