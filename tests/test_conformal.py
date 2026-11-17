"""Tests for conformal intervals and their cross-fold coverage evaluation (synthetic data)."""

import math

import numpy as np
import pandas as pd
import pytest

from predmaint.evaluation.coverage import coverage_metrics, cross_fold_coverage
from predmaint.models.conformal import conformal_quantile, interval

# --- conformal_quantile -------------------------------------------------------------------


def test_quantile_hand_calculated() -> None:
    residuals = [1, -2, 3, -4, 5, -6, 7, -8, 9]  # |r| = 1..9, n = 9
    assert conformal_quantile(residuals, alpha=0.1) == 9  # k = ceil(10 * 0.9) = 9
    assert conformal_quantile(residuals, alpha=0.2) == 8  # k = ceil(10 * 0.8) = 8
    assert conformal_quantile(residuals, alpha=0.5) == 5  # k = ceil(10 * 0.5) = 5


def test_quantile_is_infinite_when_too_few_points() -> None:
    # n = 5, alpha = 0.1: k = ceil(6 * 0.9) = 6 > 5, so no finite q guarantees 90%
    assert conformal_quantile([1, 2, 3, 4, 5], alpha=0.1) == math.inf


def test_q_increases_as_alpha_decreases() -> None:
    residuals = np.random.default_rng(0).normal(0, 10, 500)
    alphas = [0.5, 0.3, 0.2, 0.1, 0.05, 0.01]
    qs = [conformal_quantile(residuals, a) for a in alphas]
    assert qs == sorted(qs)  # never decreases
    assert qs[-1] > qs[0]


@pytest.mark.parametrize(
    "bad",
    [
        {"residuals": [], "alpha": 0.1},
        {"residuals": [1, 2], "alpha": 0.0},
        {"residuals": [1, 2], "alpha": 1.0},
        {"residuals": [1, np.nan], "alpha": 0.1},
    ],
)
def test_invalid_inputs_raise(bad: dict) -> None:
    with pytest.raises(ValueError):
        conformal_quantile(**bad)


# --- interval -------------------------------------------------------------------------------


def test_interval_bounds_and_clipping() -> None:
    lower, upper = interval([0.0, 3.0, 50.0], q=5.0)
    assert lower.tolist() == [0.0, 0.0, 45.0]
    assert upper.tolist() == [5.0, 8.0, 55.0]


def test_lower_bound_never_negative() -> None:
    rng = np.random.default_rng(1)
    preds = rng.normal(10, 30, 10_000)  # includes negative predictions
    for q in (0.0, 1.0, 25.0, 1e6, math.inf):
        lower, upper = interval(preds, q)
        assert (lower >= 0).all()
        assert (upper >= preds).all()


# --- coverage on exchangeable data ----------------------------------------------------------


def test_coverage_close_to_target_on_exchangeable_data() -> None:
    # Expected coverage is k / (n + 1), which lies in [0.9, 0.9 + 1/(n + 1)].
    rng = np.random.default_rng(2)
    coverages = []
    for _ in range(200):
        calibration = rng.standard_t(df=4, size=1_000) * 10  # heavy-tailed residuals
        test = rng.standard_t(df=4, size=1_000) * 10
        coverages.append(np.mean(np.abs(test) <= conformal_quantile(calibration, 0.1)))
    assert np.mean(coverages) == pytest.approx(0.9, abs=0.005)
    # Spread across trials: coverage given a calibration set is Beta(901, 100) (sd 0.0095),
    # plus sampling of 1000 test points (sd 0.0095): total sd ~0.0134. A minimum over 200
    # trials is expected to fall a few sd below 0.9, so test the spread, not the minimum.
    assert np.std(coverages) == pytest.approx(0.0134, rel=0.25)


def synthetic_oof(n_per_fold: int = 2_000, n_folds: int = 5, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rul = rng.uniform(0, 150, n_per_fold * n_folds)
    return pd.DataFrame(
        {
            "fold": np.repeat(np.arange(n_folds), n_per_fold),
            "rul": rul,
            "pred": rul + rng.normal(0, 12, rul.size),  # exchangeable, unbiased errors
        }
    )


def test_cross_fold_coverage_close_to_target_on_exchangeable_data() -> None:
    report = cross_fold_coverage(synthetic_oof(), alpha=0.1)
    assert report.overall["coverage"] == pytest.approx(0.9, abs=0.015)
    assert report.per_fold["coverage"].between(0.87, 0.93).all()
    assert report.overall["n"] == 10_000


def test_each_fold_is_calibrated_only_on_other_folds() -> None:
    oof = synthetic_oof()
    fold0 = oof["fold"] == 0
    oof.loc[fold0, "pred"] = oof.loc[fold0, "rul"] + 1_000  # fold 0 is wildly wrong
    report = cross_fold_coverage(oof, alpha=0.1)
    q0 = report.per_fold.set_index("fold").loc[0, "q"]
    # If fold 0's own residuals had been used, q0 would be ~1000 and fold 0 fully covered.
    assert q0 < 50
    assert report.per_fold.set_index("fold").loc[0, "coverage"] == 0.0


def test_rows_above_max_eval_rul_are_ignored() -> None:
    oof = synthetic_oof()
    extra = pd.DataFrame({"fold": [0, 1], "rul": [200.0, 300.0], "pred": [0.0, 0.0]})
    report = cross_fold_coverage(pd.concat([oof, extra], ignore_index=True), alpha=0.1)
    assert report.overall["n"] == len(oof)
    assert report.rows["rul"].max() <= 150


def test_rul_buckets_are_right_closed() -> None:
    rul = [0, 25, 26, 50, 51, 75, 100, 101, 150]
    oof = pd.DataFrame({"fold": [0, 1] * 4 + [0], "rul": rul, "pred": rul})
    report = cross_fold_coverage(oof, alpha=0.5)
    counts = dict(zip(report.by_bucket["bucket"], report.by_bucket["n"], strict=True))
    assert counts == {"0-25": 2, "25-50": 2, "50-75": 2, "75-100": 1, "100-150": 2}


def test_needs_two_folds_and_columns() -> None:
    with pytest.raises(ValueError, match="two folds"):
        cross_fold_coverage(pd.DataFrame({"fold": [0, 0], "rul": [1.0, 2.0], "pred": [1.0, 2.0]}))
    with pytest.raises(ValueError, match="pred"):
        cross_fold_coverage(pd.DataFrame({"fold": [0, 1], "rul": [1.0, 2.0]}))


def test_coverage_metrics_keys() -> None:
    metrics = coverage_metrics(cross_fold_coverage(synthetic_oof(), alpha=0.1))
    assert {"conformal_coverage", "conformal_mean_width", "conformal_q_mean"} <= set(metrics)
    assert len(metrics["conformal_coverage_fold"]) == 5
    assert "conformal_coverage_rul_0_25" in metrics
    assert "conformal_coverage_rul_100_150" in metrics
