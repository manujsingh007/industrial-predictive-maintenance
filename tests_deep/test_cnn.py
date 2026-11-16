"""Tests for the 1D-CNN (tiny synthetic data, CPU).

Run separately from tests/ (see tests_deep/README.md): uv run --group deep pytest tests_deep
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="needs the 'deep' dependency group: uv sync --group deep")

from sklearn.base import clone  # noqa: E402
from sklearn.exceptions import NotFittedError  # noqa: E402

from predmaint.models.cnn import CNNRegressor  # noqa: E402
from predmaint.training.cv import cross_validate  # noqa: E402

SENSORS = ("a", "b", "c")
COLUMNS = ["unit", "cycle", *SENSORS]


def tiny_cnn(**overrides: object) -> CNNRegressor:
    params = dict(
        window=10,
        sensors=SENSORS,
        channels=(4, 8),
        kernel_size=3,
        dense=8,
        max_epochs=3,
        batch_size=64,
        device="cpu",
    )
    return CNNRegressor(**{**params, **overrides})


@pytest.fixture
def fleet() -> pd.DataFrame:
    """6 engines of 30-55 cycles; sensor 'a' drifts towards failure."""
    rng = np.random.default_rng(0)
    frames = []
    for unit in range(1, 7):
        n = 25 + 5 * unit
        rul = np.arange(n - 1, -1, -1)
        frames.append(
            pd.DataFrame(
                {
                    "unit": unit,
                    "cycle": np.arange(1, n + 1),
                    "a": 50 - 0.3 * rul + rng.normal(0, 1, n),
                    "b": rng.normal(10, 2, n),
                    "c": rng.normal(-5, 1, n),
                    "rul": rul,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_predict_returns_one_finite_value_per_row(fleet: pd.DataFrame) -> None:
    model = tiny_cnn().fit(fleet[COLUMNS], fleet["rul"])
    pred = model.predict(fleet[COLUMNS])
    assert pred.shape == (len(fleet),)
    assert pred.dtype == np.float64
    assert np.isfinite(pred).all()


def test_scaler_is_fitted_only_on_fit_data(fleet: pd.DataFrame) -> None:
    train, other = fleet[fleet["unit"] <= 4], fleet[fleet["unit"] > 4]
    model = tiny_cnn().fit(train[COLUMNS], train["rul"])
    np.testing.assert_allclose(model.scaler_.mean_, train[list(SENSORS)].mean().to_numpy())
    mean_before = model.scaler_.mean_.copy()
    model.predict(other[COLUMNS])
    np.testing.assert_array_equal(model.scaler_.mean_, mean_before)


def test_early_stopping_engines_come_from_training_data(fleet: pd.DataFrame) -> None:
    train = fleet[fleet["unit"] <= 4]
    model = tiny_cnn().fit(train[COLUMNS], train["rul"])
    assert set(model.early_stopping_units_) <= {1, 2, 3, 4}
    assert len(model.early_stopping_units_) == 1  # about 15% of 4 engines, at least one
    assert 1 <= model.n_epochs_ <= 3


def test_same_seed_gives_same_predictions_on_cpu(fleet: pd.DataFrame) -> None:
    a = tiny_cnn(seed=7).fit(fleet[COLUMNS], fleet["rul"]).predict(fleet[COLUMNS])
    b = tiny_cnn(seed=7).fit(fleet[COLUMNS], fleet["rul"]).predict(fleet[COLUMNS])
    c = tiny_cnn(seed=8).fit(fleet[COLUMNS], fleet["rul"]).predict(fleet[COLUMNS])
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_predictions_do_not_depend_on_future_cycles(fleet: pd.DataFrame) -> None:
    # Like the test set: an engine cut off at cycle 20 must get the same predictions for
    # cycles 1-20 as the full engine.
    model = tiny_cnn().fit(fleet[COLUMNS], fleet["rul"])
    engine = fleet[fleet["unit"] == 3]
    full = model.predict(engine[COLUMNS])
    truncated = model.predict(engine[engine["cycle"] <= 20][COLUMNS])
    # Batch size changes float32 summation order (~1e-7 cycles); a real leak would be far larger.
    np.testing.assert_allclose(truncated, full[:20], rtol=0, atol=1e-4)


def test_scikit_learn_compatibility(fleet: pd.DataFrame) -> None:
    model = tiny_cnn()
    assert clone(model).get_params() == model.get_params()
    with pytest.raises(NotFittedError):
        model.predict(fleet[COLUMNS])


def test_missing_columns_raise(fleet: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unit"):
        tiny_cnn().fit(fleet[list(SENSORS)], fleet["rul"])


def test_works_inside_cross_validate(fleet: pd.DataFrame) -> None:
    result = cross_validate(lambda: tiny_cnn(), fleet, cap=30, features=COLUMNS, n_splits=3)
    assert len(result.folds) == 3
    assert result.folds["rmse"].notna().all()
