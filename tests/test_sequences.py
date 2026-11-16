"""Tests for predmaint.features.sequences (numpy only; no PyTorch needed)."""

import numpy as np
import pytest

from predmaint.features.sequences import build_sequences


def engine_rows(unit: int, n: int, offset: float = 0.0) -> tuple[list, list, list]:
    """Rows whose single feature equals cycle number + offset, so positions are readable."""
    cycles = list(range(1, n + 1))
    return [unit] * n, cycles, [[c + offset] for c in cycles]


def test_left_padding_repeats_first_cycle() -> None:
    units, cycles, values = engine_rows(1, 3)
    seq = build_sequences(units, cycles, values, window=5)[:, :, 0]
    assert seq.tolist() == [
        [1, 1, 1, 1, 1],  # cycle 1: four padded copies of cycle 1, then cycle 1 itself
        [1, 1, 1, 1, 2],  # cycle 2
        [1, 1, 1, 2, 3],  # cycle 3: padding only on the left, current cycle is last
    ]


def test_full_window_uses_most_recent_cycles_oldest_first() -> None:
    units, cycles, values = engine_rows(1, 7)
    seq = build_sequences(units, cycles, values, window=3)[:, :, 0]
    assert seq[-1].tolist() == [5, 6, 7]
    assert seq[3].tolist() == [2, 3, 4]


def test_changing_a_future_row_never_changes_earlier_sequences() -> None:
    units, cycles, values = engine_rows(1, 10)
    before = build_sequences(units, cycles, values, window=4)
    values[6][0] = 999.0  # change cycle 7
    after = build_sequences(units, cycles, values, window=4)
    np.testing.assert_array_equal(after[:6], before[:6])  # cycles 1-6 untouched
    assert not np.array_equal(after[6], before[6])


def test_engines_never_mix_and_row_order_is_preserved() -> None:
    u1, c1, v1 = engine_rows(1, 4, offset=100)
    u2, c2, v2 = engine_rows(2, 3, offset=200)
    # interleave rows of both engines in a shuffled order
    rows = list(zip(u1 + u2, c1 + c2, v1 + v2, strict=True))
    shuffled = [rows[i] for i in [4, 0, 6, 2, 5, 1, 3]]
    units, cycles, values = (list(col) for col in zip(*shuffled, strict=True))
    seq = build_sequences(units, cycles, values, window=3)[:, :, 0]
    for (unit, cycle, _), s in zip(shuffled, seq, strict=True):
        base = 100 if unit == 1 else 200
        expected = [base + max(1, cycle - k) for k in (2, 1, 0)]
        assert s.tolist() == expected


def test_output_shape_and_dtype() -> None:
    seq = build_sequences([1, 1, 2], [1, 2, 1], np.ones((3, 4)), window=6)
    assert seq.shape == (3, 6, 4)
    assert seq.dtype == np.float32


def test_duplicate_rows_and_bad_window_raise() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build_sequences([1, 1], [1, 1], [[0.0], [1.0]], window=2)
    with pytest.raises(ValueError, match="window"):
        build_sequences([1], [1], [[0.0]], window=0)
