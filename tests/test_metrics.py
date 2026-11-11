"""Tests for predmaint.evaluation.metrics with hand-calculated values."""

import math

import numpy as np
import pytest

from predmaint.evaluation.metrics import bootstrap_ci, mae, phm08_score, rmse

# true RUL and predictions with errors d = pred - true = [+2, -2, +4, -4]
Y_TRUE = [10, 20, 30, 40]
Y_PRED = [12, 18, 34, 36]


# --- RMSE / MAE --------------------------------------------------------------------


def test_rmse_hand_calculated() -> None:
    # squared errors 4, 4, 16, 16 -> mean 10 -> sqrt(10)
    assert rmse(Y_TRUE, Y_PRED) == pytest.approx(math.sqrt(10))


def test_mae_hand_calculated() -> None:
    # absolute errors 2, 2, 4, 4 -> mean 3
    assert mae(Y_TRUE, Y_PRED) == pytest.approx(3.0)


def test_perfect_predictions_score_zero() -> None:
    for metric in (rmse, mae, phm08_score):
        assert metric(Y_TRUE, Y_TRUE) == 0.0


# --- PHM08 score -----------------------------------------------------------------


def test_phm08_late_uses_scale_10() -> None:
    # d = +10 -> exp(10/10) - 1 = e - 1
    assert phm08_score([50], [60]) == pytest.approx(math.e - 1)


def test_phm08_early_uses_scale_13() -> None:
    # d = -13 -> exp(13/13) - 1 = e - 1
    assert phm08_score([50], [37]) == pytest.approx(math.e - 1)


def test_phm08_is_a_sum_over_engines() -> None:
    # d = +10, -13, 0 -> (e - 1) + (e - 1) + 0
    assert phm08_score([50, 50, 50], [60, 37, 50]) == pytest.approx(2 * (math.e - 1))


def test_phm08_penalises_late_more_than_equally_early() -> None:
    late = phm08_score([50], [60])  # d = +10 -> e^(10/10) - 1 ≈ 1.718
    early = phm08_score([50], [40])  # d = -10 -> e^(10/13) - 1 ≈ 1.159
    assert late == pytest.approx(math.exp(1) - 1)
    assert early == pytest.approx(math.exp(10 / 13) - 1)
    assert late > early


def test_rmse_and_mae_are_symmetric_unlike_phm08() -> None:
    assert rmse([50], [60]) == rmse([50], [40])
    assert mae([50], [60]) == mae([50], [40])


# --- input validation -------------------------------------------------------------


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [([1, 2], [1]), ([], []), ([[1, 2]], [[1, 2]])],
    ids=["length mismatch", "empty", "2-D"],
)
def test_invalid_inputs_raise(y_true: list, y_pred: list) -> None:
    for metric in (rmse, mae, phm08_score):
        with pytest.raises(ValueError):
            metric(y_true, y_pred)


# --- bootstrap ---------------------------------------------------------------------


@pytest.fixture
def predictions() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y_true = rng.integers(5, 150, size=100).astype(float)
    y_pred = y_true + rng.normal(0, 15, size=100)
    return y_true, y_pred


def test_bootstrap_same_seed_is_reproducible(predictions: tuple[np.ndarray, np.ndarray]) -> None:
    a = bootstrap_ci(rmse, *predictions, n_resamples=2_000, seed=7)
    b = bootstrap_ci(rmse, *predictions, n_resamples=2_000, seed=7)
    assert a == b


def test_bootstrap_different_seed_changes_interval(
    predictions: tuple[np.ndarray, np.ndarray],
) -> None:
    a = bootstrap_ci(rmse, *predictions, n_resamples=2_000, seed=1)
    b = bootstrap_ci(rmse, *predictions, n_resamples=2_000, seed=2)
    assert a[0] == b[0]  # point estimate does not depend on the seed
    assert a[1:] != b[1:]


@pytest.mark.parametrize("metric", [rmse, mae, phm08_score])
def test_bootstrap_interval_contains_point(
    metric, predictions: tuple[np.ndarray, np.ndarray]
) -> None:
    point, low, high = bootstrap_ci(metric, *predictions)
    assert point == metric(*predictions)
    assert low < point < high


def test_bootstrap_constant_errors_give_zero_width_interval() -> None:
    # every engine has |d| = 5, so every resample has MAE exactly 5
    point, low, high = bootstrap_ci(mae, [10, 20, 30], [15, 15, 35], n_resamples=500)
    assert point == low == high == pytest.approx(5.0)


def test_bootstrap_rejects_zero_resamples() -> None:
    with pytest.raises(ValueError):
        bootstrap_ci(rmse, Y_TRUE, Y_PRED, n_resamples=0)
