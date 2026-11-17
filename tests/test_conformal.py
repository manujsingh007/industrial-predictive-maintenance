"""Tests for conformal intervals and their cross-fold coverage evaluation (synthetic data)."""

import math

import numpy as np
import pandas as pd
import pytest

from predmaint.evaluation.coverage import coverage_metrics, cross_fold_coverage
from predmaint.models.conformal import (
    conformal_quantile,
    conformal_score_quantile,
    cqr_interval,
    cqr_scores,
    interval,
    mondrian_bin_labels,
    mondrian_bins,
    mondrian_interval,
    mondrian_quantiles,
)
from predmaint.models.gbdt import make_lgbm_quantile

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


# --- CQR ----------------------------------------------------------------------------------


def test_cqr_scores_hand_calculated() -> None:
    # band [10, 20]: inside -> negative distance to the nearer edge, outside -> distance out
    scores = cqr_scores([15, 25, 5, 10], [10, 10, 10, 10], [20, 20, 20, 20])
    assert scores.tolist() == [-5, 5, 5, 0]


def test_cqr_interval_clips_and_handles_crossed_quantiles() -> None:
    lower, upper = cqr_interval([10.0, 2.0, 30.0], [20.0, 4.0, 25.0], q=3.0)
    assert lower.tolist() == [7.0, 0.0, 22.0]  # third row: crossed lo/hi swapped first
    assert upper.tolist() == [23.0, 7.0, 33.0]
    # negative q shrinks the band; upper never below the clipped lower bound
    lower, upper = cqr_interval([1.0, 40.0], [2.0, 60.0], q=-5.0)
    assert lower.tolist() == [6.0, 45.0] and upper.tolist() == [6.0, 55.0]
    assert (upper >= lower).all() and (lower >= 0).all()


def test_score_quantile_keeps_sign() -> None:
    # all scores negative (bands too wide) -> negative q shrinks them
    assert conformal_score_quantile([-9, -8, -7, -6, -5, -4, -3, -2, -1], alpha=0.1) == -1


def heteroscedastic(n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Noise sd grows tenfold with x: 1 at x = 0, 10 at x = 1."""
    x = rng.uniform(0, 1, n)
    y = 60 + 40 * x + (1 + 9 * x) * rng.normal(size=n)
    return pd.DataFrame({"x": x}), y


def test_cqr_adapts_width_on_heteroscedastic_data() -> None:
    rng = np.random.default_rng(5)
    (X_tr, y_tr), (X_cal, y_cal), (X_te, y_te) = (heteroscedastic(3_000, rng) for _ in range(3))
    lo_model = make_lgbm_quantile(0.05, n_estimators=200).fit(X_tr, y_tr)
    hi_model = make_lgbm_quantile(0.95, n_estimators=200).fit(X_tr, y_tr)

    q = conformal_score_quantile(
        cqr_scores(y_cal, lo_model.predict(X_cal), hi_model.predict(X_cal)), alpha=0.1
    )
    lower, upper = cqr_interval(lo_model.predict(X_te), hi_model.predict(X_te), q)
    width = upper - lower
    covered = (lower <= y_te) & (y_te <= upper)

    quiet, noisy = X_te["x"] < 0.2, X_te["x"] > 0.8
    assert covered.mean() == pytest.approx(0.9, abs=0.02)
    assert width[quiet].mean() < 0.4 * width[noisy].mean()  # true sd ratio is ~0.15
    # coverage holds in both regions, not just on average
    assert covered[quiet].mean() > 0.8 and covered[noisy].mean() > 0.8


def test_cross_fold_cqr_adapts_where_split_cannot() -> None:
    # Oracle quantiles on data whose noise grows with RUL: CQR width tracks the noise,
    # split conformal has one width everywhere.
    rng = np.random.default_rng(6)
    mu = rng.uniform(20, 150, 10_000)
    sd = 2 + mu / 10
    oof = pd.DataFrame(
        {
            "fold": np.repeat(np.arange(5), 2_000),
            "rul": mu + sd * rng.normal(size=mu.size),
            "pred": mu,
            "lo": mu - 1.645 * sd,
            "hi": mu + 1.645 * sd,
        }
    )
    split = cross_fold_coverage(oof, 0.1, method="split")
    cqr = cross_fold_coverage(oof, 0.1, method="cqr")
    assert cqr.overall["coverage"] == pytest.approx(0.9, abs=0.015)
    assert split.overall["coverage"] == pytest.approx(0.9, abs=0.015)
    cqr_w = cqr.by_bucket.set_index("bucket")["mean_width"]
    split_w = split.by_bucket.set_index("bucket")["mean_width"]
    assert cqr_w["25-50"] < cqr_w["100-150"] / 2
    assert split_w.max() - split_w.min() < 1.0  # constant width (bucket means of pred ± q)
    gap = lambda r: (r.by_bucket["coverage"] - 0.9).abs().max()  # noqa: E731
    assert gap(cqr) < gap(split)  # CQR is closer to 90% in every bucket's worst case


def test_cqr_requires_lo_hi_and_known_method() -> None:
    oof = synthetic_oof()
    with pytest.raises(ValueError, match="lo"):
        cross_fold_coverage(oof, 0.1, method="cqr")
    with pytest.raises(ValueError, match="method"):
        cross_fold_coverage(oof, 0.1, method="bogus")


# --- Mondrian ---------------------------------------------------------------------------------


def test_mondrian_bins_use_left_closed_edges() -> None:
    bins = mondrian_bins([-5, 0, 29.9, 30, 59, 60, 90, 500])
    assert bins.tolist() == [0, 0, 0, 1, 1, 2, 3, 3]
    assert mondrian_bin_labels() == ["0-30", "30-60", "60-90", "90+"]


def mondrian_data(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Predictions spread over all bins; error sd 2 below 30 cycles, 6, 10, then 20 above 90."""
    pred = rng.uniform(0, 150, n)
    sd = np.select([pred < 30, pred < 60, pred < 90], [2.0, 6.0, 10.0], 20.0)
    return pred, pred + sd * rng.normal(size=n)  # (pred, true RUL); RUL may dip below 0


def test_per_bin_q_grows_with_bin_errors() -> None:
    pred, y = mondrian_data(20_000, np.random.default_rng(7))
    q, fallback = mondrian_quantiles(y - pred, pred, alpha=0.1)
    assert not fallback.any()
    assert q[0] < q[1] < q[2] < q[3]
    # 90% two-sided normal quantile is 1.645 sd
    np.testing.assert_allclose(q, 1.645 * np.array([2.0, 6.0, 10.0, 20.0]), rtol=0.08)


def test_mondrian_coverage_close_to_90_within_each_predicted_bin() -> None:
    rng = np.random.default_rng(8)
    cal_pred, cal_y = mondrian_data(20_000, rng)
    q, _ = mondrian_quantiles(cal_y - cal_pred, cal_pred, alpha=0.1)
    pred, y = mondrian_data(20_000, rng)
    # compare unclipped intervals: true RUL can dip below 0 in this synthetic data
    q_row = q[mondrian_bins(pred)]
    covered = np.abs(y - pred) <= q_row
    for b in range(4):
        assert covered[mondrian_bins(pred) == b].mean() == pytest.approx(0.9, abs=0.02)


def test_split_misses_per_bin_coverage_that_mondrian_achieves() -> None:
    rng = np.random.default_rng(9)
    pred, y = mondrian_data(10_000, rng)
    oof = pd.DataFrame({"fold": np.repeat(np.arange(5), 2_000), "rul": y + 20, "pred": pred + 20})
    split = cross_fold_coverage(oof, 0.1, method="split", max_eval_rul=1e9)
    mondrian = cross_fold_coverage(oof, 0.1, method="mondrian", max_eval_rul=1e9)
    worst = lambda r: (r.by_pred_bin["coverage"] - 0.9).abs().max()  # noqa: E731
    assert worst(mondrian) < 0.03
    assert worst(split) > 0.05  # one global width over-covers quiet bins, under-covers noisy


def test_mondrian_falls_back_to_global_q_for_small_bins() -> None:
    rng = np.random.default_rng(10)
    pred = np.r_[rng.uniform(0, 29, 30), rng.uniform(30, 150, 2_000)]  # bin 0 has 30 rows
    residuals = np.r_[np.full(30, 100.0), rng.normal(0, 5, 2_000)]
    q, fallback = mondrian_quantiles(residuals, pred, alpha=0.1, min_bin_size=50)
    global_q = conformal_quantile(residuals, alpha=0.1)
    assert fallback.tolist() == [True, False, False, False]
    assert q[0] == global_q  # not ~100, which bin 0's own 30 rows would give
    q, fallback = mondrian_quantiles(residuals, pred, alpha=0.1, min_bin_size=20)
    assert not fallback[0] and q[0] == 100.0


def test_mondrian_interval_uses_each_rows_bin_and_clips() -> None:
    lower, upper = mondrian_interval([10.0, 45.0, 120.0], q_per_bin=[15.0, 5.0, 8.0, 30.0])
    assert lower.tolist() == [0.0, 40.0, 90.0]
    assert upper.tolist() == [25.0, 50.0, 150.0]


def test_cross_fold_mondrian_reports_q_per_fold_and_bin() -> None:
    rng = np.random.default_rng(11)
    pred, y = mondrian_data(10_000, rng)
    oof = pd.DataFrame({"fold": np.repeat(np.arange(5), 2_000), "rul": y + 20, "pred": pred + 20})
    report = cross_fold_coverage(oof, 0.1, method="mondrian", max_eval_rul=1e9)
    assert len(report.q_by_bin) == 5 * 4
    assert report.per_fold["q"].isna().all()
    metrics = coverage_metrics(report)
    assert len(metrics["conformal_q_bin_90_plus_fold"]) == 5


def test_mondrian_bins_calibration_rows_by_prediction_not_truth() -> None:
    # Every prediction is in bin 0-30 while true RUL spreads over 0-150. Binning by the
    # prediction gives bin 0 all calibration rows; binning by truth would leave it ~20% and
    # starve it below the threshold in small folds, forcing the global fallback.
    rng = np.random.default_rng(12)
    rul = rng.uniform(0, 150, 300)
    oof = pd.DataFrame(
        {"fold": np.repeat(np.arange(5), 60), "rul": rul, "pred": np.full(300, 10.0)}
    )
    report = cross_fold_coverage(oof, 0.1, method="mondrian", min_bin_size=200)
    bin0 = report.q_by_bin[report.q_by_bin["bin"] == "0-30"]
    assert (bin0["n_calibration"] == 240).all()
    assert not bin0["fallback"].any()
