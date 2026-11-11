"""Tests for the baselines and cross-validation, using small synthetic fleets."""

import numpy as np
import pandas as pd
import pytest

from predmaint.evaluation.splits import engine_folds
from predmaint.features.selection import KEPT_SENSORS
from predmaint.models.baselines import make_linear_baseline, make_mean_baseline
from predmaint.training.cv import cross_validate


@pytest.fixture
def fleet() -> pd.DataFrame:
    """10 engines living 160-205 cycles, so some rows have RUL > 150.

    Sensor levels differ per engine, so the mean of any training fold differs from the
    mean of all rows. ``oracle`` equals the uncapped RUL, except where RUL > 150, where it
    is deliberately 1000 cycles off.
    """
    rng = np.random.default_rng(0)
    frames = []
    for unit in range(1, 11):
        lifetime = 155 + 5 * unit
        cycle = np.arange(1, lifetime + 1)
        rul = lifetime - cycle
        sensors = rng.normal(loc=unit * 3.0, scale=1.0, size=(lifetime, len(KEPT_SENSORS)))
        frame = pd.DataFrame(sensors, columns=KEPT_SENSORS)
        frame.insert(0, "unit", unit)
        frame.insert(1, "cycle", cycle)
        frame["rul"] = rul
        frame["oracle"] = np.where(rul <= 150, rul, rul + 1000).astype(float)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class Oracle:
    """Predicts the ``oracle`` column; records the largest training target it was given."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Oracle":
        self.max_target_ = float(np.max(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["oracle"].to_numpy()


# --- baselines -------------------------------------------------------------------------


def test_factories_return_fresh_unfitted_models() -> None:
    assert make_mean_baseline() is not make_mean_baseline()
    assert make_linear_baseline() is not make_linear_baseline()
    assert not hasattr(make_linear_baseline().named_steps["scale"], "mean_")


def test_mean_baseline_predicts_training_mean() -> None:
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    model = make_mean_baseline().fit(X, [10, 20, 60])
    assert model.predict(X).tolist() == [30, 30, 30]


def test_linear_pipeline_scales_then_regresses() -> None:
    assert list(make_linear_baseline().named_steps) == ["scale", "regress"]


def test_scaler_statistics_do_not_change_when_predicting(fleet: pd.DataFrame) -> None:
    train, val = fleet[fleet["unit"] <= 5], fleet[fleet["unit"] > 5]
    model = make_linear_baseline().fit(train[KEPT_SENSORS], train["rul"])
    mean_before = model.named_steps["scale"].mean_.copy()
    model.predict(val[KEPT_SENSORS])
    np.testing.assert_array_equal(model.named_steps["scale"].mean_, mean_before)
    np.testing.assert_allclose(mean_before, train[KEPT_SENSORS].mean().to_numpy())


def test_scaler_in_each_fold_is_fitted_on_training_rows_only(fleet: pd.DataFrame) -> None:
    result = cross_validate(make_linear_baseline, fleet)
    X = fleet[KEPT_SENSORS]
    folds = list(engine_folds(fleet, n_splits=5, seed=42))
    for model, (train_idx, _) in zip(result.models, folds, strict=True):
        scaler = model.named_steps["scale"]
        np.testing.assert_allclose(scaler.mean_, X.iloc[train_idx].mean().to_numpy())
        np.testing.assert_allclose(scaler.scale_, X.iloc[train_idx].std(ddof=0).to_numpy())
        assert not np.allclose(scaler.mean_, X.mean().to_numpy())  # all rows would differ


# --- cross_validate -------------------------------------------------------------------


@pytest.mark.parametrize("n_splits", [5, 3])
def test_returns_one_result_per_fold(fleet: pd.DataFrame, n_splits: int) -> None:
    result = cross_validate(make_mean_baseline, fleet, n_splits=n_splits)
    assert len(result.folds) == n_splits
    assert len(result.models) == n_splits
    assert result.folds["n_val_engines"].sum() == fleet["unit"].nunique()


def test_never_scores_rows_above_150_and_uses_uncapped_truth(fleet: pd.DataFrame) -> None:
    # The oracle is exact on RUL <= 150 (uncapped, so also exact on 125 < RUL <= 150) and
    # 1000 cycles off above 150. Zero error proves only RUL <= 150 rows were scored, and that
    # they were scored against the uncapped RUL (capped truth would give errors up to 25).
    result = cross_validate(Oracle, fleet, cap=125, features=["oracle"])
    assert (result.folds["rmse"] == 0).all()
    assert (result.folds["mae"] == 0).all()


def test_scored_row_count_matches_rows_with_rul_up_to_150(fleet: pd.DataFrame) -> None:
    result = cross_validate(make_mean_baseline, fleet)
    assert result.folds["n_val_rows_scored"].sum() == (fleet["rul"] <= 150).sum()


def test_models_are_trained_on_capped_target(fleet: pd.DataFrame) -> None:
    result = cross_validate(Oracle, fleet, cap=110, features=["oracle"])
    assert all(m.max_target_ == 110 for m in result.models)


def test_summary_has_mean_and_std(fleet: pd.DataFrame) -> None:
    summary = cross_validate(make_mean_baseline, fleet).summary()
    assert list(summary.index) == ["mean", "std"]
    assert list(summary.columns) == ["rmse", "mae"]


def test_missing_columns_raise(fleet: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="rul"):
        cross_validate(make_mean_baseline, fleet.drop(columns="rul"))
