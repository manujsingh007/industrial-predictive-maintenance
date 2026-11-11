"""Tests for predmaint.models.gbdt (synthetic data; needs LightGBM's OpenMP runtime)."""

import numpy as np
import pandas as pd
import pytest

from predmaint.models.gbdt import LGBM_DEFAULTS, make_lgbm
from predmaint.training.cv import cross_validate
from predmaint.training.run_cv import model_hyperparams


@pytest.fixture
def data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=["a", "b", "c", "d"])
    y = 50 + 10 * X["a"] - 5 * X["b"] + rng.normal(0, 1, 300)
    return X, y.to_numpy()


def test_fits_and_predicts_with_right_shape(data: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, y = data
    model = make_lgbm(n_estimators=50).fit(X.iloc[:200], y[:200])
    pred = model.predict(X.iloc[200:])
    assert pred.shape == (100,)
    assert np.isfinite(pred).all()
    assert np.corrcoef(pred, y[200:])[0, 1] > 0.8  # it actually learned the signal


def test_defaults_and_overrides() -> None:
    params = make_lgbm(num_leaves=15).get_params()
    assert params["num_leaves"] == 15
    assert {k: params[k] for k in LGBM_DEFAULTS if k != "num_leaves"} == {
        k: v for k, v in LGBM_DEFAULTS.items() if k != "num_leaves"
    }


def test_unknown_override_is_rejected() -> None:
    with pytest.raises(TypeError, match="num_leaf"):
        make_lgbm(num_leaf=15)


def test_same_seed_gives_identical_predictions(data: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, y = data
    a = make_lgbm(n_estimators=50).fit(X, y).predict(X)
    b = make_lgbm(n_estimators=50).fit(X, y).predict(X)
    np.testing.assert_array_equal(a, b)


def test_works_in_cross_validate_and_logs_hyperparams() -> None:
    rng = np.random.default_rng(1)
    frames = [
        pd.DataFrame(
            {
                "unit": u,
                "cycle": np.arange(1, 41),
                "rul": np.arange(39, -1, -1),
                "a": rng.normal(size=40),
            }
        )
        for u in range(1, 7)
    ]
    df = pd.concat(frames, ignore_index=True)
    result = cross_validate(lambda: make_lgbm(n_estimators=20), df, features=["a"], n_splits=3)
    assert len(result.folds) == 3
    logged = model_hyperparams("lgbm", result)
    assert logged["lgbm_n_estimators"] == 20  # read from the fitted model, not the defaults
    assert set(logged) == {f"lgbm_{k}" for k in LGBM_DEFAULTS}
    assert model_hyperparams("linear", result) == {}
