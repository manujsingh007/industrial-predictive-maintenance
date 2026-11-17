"""One-time evaluation on the official test set, exactly as frozen in docs/evaluation.md and
ADRs 0001-0003. No modelling decision is made here.

Run with: ``uv run python -m predmaint.training.final_eval --subset FD001``

Steps:

1. Calibrate Mondrian q per predicted-RUL bin from the 5-fold out-of-fold residuals of the
   training engines (rows with true RUL <= 150, as in ADR 0003's evaluation).
2. Train LightGBM (window features, cap 125, default hyperparameters) and the two baselines
   on ALL training engines.
3. Build causal window features on each test engine's full available history and predict at
   each engine's LAST cycle; compare with the uncapped true RUL from RUL_FD00x.txt.
4. Report RMSE, MAE and PHM08 with 95% bootstrap intervals over engines, the paired-bootstrap
   RMSE difference LightGBM minus linear, and interval coverage and width per predicted bin.

**Run-once guard.** The test set is a one-time, unbiased check. If it can be re-run casually,
it is tempting to adjust something, re-run and keep the better number. That quietly turns the
test set into a second validation set and makes the reported score optimistic. So the script
refuses to run once ``reports/final_test_<subset>.md`` exists. ``--force`` overrides it, and a
forced run is labelled as such in the report and in MLflow, so a second look is always
deliberate and visible.
"""

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from predmaint.data.download import PROJECT_ROOT
from predmaint.data.load import add_rul, load_test, load_test_rul, load_train
from predmaint.evaluation.metrics import (
    bootstrap_ci,
    mae,
    paired_bootstrap_diff,
    phm08_score,
    rmse,
)
from predmaint.features.selection import DEFAULT_RUL_CAP, KEPT_SENSORS, SUBSETS_WITH_ADR, cap_rul
from predmaint.models.baselines import make_linear_baseline, make_mean_baseline
from predmaint.models.conformal import (
    MIN_BIN_CALIBRATION,
    MONDRIAN_EDGES,
    mondrian_bin_labels,
    mondrian_bins,
    mondrian_interval,
    mondrian_quantiles,
)
from predmaint.models.gbdt import make_lgbm
from predmaint.training.cv import MAX_EVAL_RUL, cross_validate
from predmaint.training.run_cv import DEFAULT_WINDOWS, N_SPLITS, SEED, build_features

REPORTS_DIR = PROJECT_ROOT / "reports"
ALPHA = 0.1
N_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42
MODEL_FACTORY = make_lgbm  # replaced by a small model in tests only
METRICS = {"RMSE": rmse, "MAE": mae, "PHM08": phm08_score}


class FinalEvalAlreadyDone(RuntimeError):
    """Raised when the final evaluation already has a report and ``--force`` was not given."""


def report_paths(subset: str, report_dir: Path = REPORTS_DIR) -> tuple[Path, Path]:
    """Paths of the markdown report and the per-engine predictions CSV."""
    name = subset.lower()
    return report_dir / f"final_test_{name}.md", report_dir / f"test_predictions_{name}.csv"


def check_not_already_run(report: Path, force: bool) -> None:
    """Refuse a second run unless forced (see the module docstring for why)."""
    if report.exists() and not force:
        raise FinalEvalAlreadyDone(
            f"{report} already exists: the official test set has been evaluated. "
            "Re-running invites tuning on the test set. Use --force only for a deliberate, "
            "documented re-run."
        )


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion; reliable for small n (e.g. engines per bin)."""
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class FinalResult:
    """Everything the final evaluation computed."""

    subset: str
    predictions: pd.DataFrame  # unit, true_rul, prediction, lower, upper, pred_bin, baselines
    point_metrics: pd.DataFrame  # model, metric, value, low, high
    rmse_diff: tuple[float, float, float]  # LightGBM minus linear: point, low, high
    q_by_bin: pd.DataFrame  # bin, q, n_calibration, fallback
    coverage_by_bin: pd.DataFrame  # bin, engines, covered, coverage, low, high, mean_width
    overall_coverage: dict[str, float]  # engines, covered, coverage, low, high, mean_width
    forced: bool


def _last_cycle_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df.groupby("unit")["cycle"].idxmax()].sort_values("unit")


def evaluate(subset: str, *, forced: bool = False) -> FinalResult:
    """Run the frozen final evaluation (no guard, no files, no logging)."""
    train = add_rul(load_train(subset))
    data, features = build_features(train, "window", DEFAULT_WINDOWS)
    y_capped = cap_rul(data["rul"], DEFAULT_RUL_CAP)

    # 1. Mondrian calibration from out-of-fold residuals (ADR 0003).
    cv = cross_validate(
        MODEL_FACTORY, data, cap=DEFAULT_RUL_CAP, features=features, n_splits=N_SPLITS, seed=SEED
    )
    calibration = cv.oof[cv.oof["rul"] <= MAX_EVAL_RUL]
    q_bins, fallback = mondrian_quantiles(
        calibration["rul"] - calibration["pred"],
        calibration["pred"],
        ALPHA,
        MONDRIAN_EDGES,
        MIN_BIN_CALIBRATION,
    )
    labels = mondrian_bin_labels(MONDRIAN_EDGES)
    q_by_bin = pd.DataFrame(
        {
            "bin": labels,
            "q": q_bins,
            "n_calibration": np.bincount(
                mondrian_bins(calibration["pred"], MONDRIAN_EDGES), minlength=len(labels)
            ),
            "fallback": fallback,
        }
    )

    # 2. Final models on ALL training engines.
    model = MODEL_FACTORY().fit(data[features], y_capped)
    mean_model = make_mean_baseline().fit(train[KEPT_SENSORS], y_capped)
    linear_model = make_linear_baseline().fit(train[KEPT_SENSORS], y_capped)

    # 3. Test: causal window features on each engine's full history; predict at its last cycle.
    test = load_test(subset)
    test_data, _ = build_features(test, "window", DEFAULT_WINDOWS)
    last = _last_cycle_rows(test_data)
    truth = load_test_rul(subset)
    if sorted(truth["unit"]) != sorted(last["unit"]):
        raise ValueError("test units and RUL file units do not match")
    last = last.merge(truth, on="unit", validate="one_to_one")

    pred = model.predict(last[features])
    lower, upper = mondrian_interval(pred, q_bins, MONDRIAN_EDGES)
    predictions = pd.DataFrame(
        {
            "unit": last["unit"].to_numpy(),
            "true_rul": last["rul"].to_numpy(),
            "prediction": pred,
            "lower": lower,
            "upper": upper,
            "pred_bin": [labels[b] for b in mondrian_bins(pred, MONDRIAN_EDGES)],
            "mean_baseline": mean_model.predict(last[KEPT_SENSORS]),
            "linear_baseline": linear_model.predict(last[KEPT_SENSORS]),
        }
    )

    # 4. Point metrics with bootstrap intervals over engines (same seed -> same resamples).
    y = predictions["true_rul"].to_numpy(float)
    models = {
        "LightGBM + window features": predictions["prediction"],
        "Linear regression (current sensors)": predictions["linear_baseline"],
        "Mean RUL baseline": predictions["mean_baseline"],
    }
    rows = []
    for model_name, p in models.items():
        for metric_name, fn in METRICS.items():
            value, low, high = bootstrap_ci(fn, y, p, N_RESAMPLES, BOOTSTRAP_SEED)
            rows.append(
                {
                    "model": model_name,
                    "metric": metric_name,
                    "value": value,
                    "low": low,
                    "high": high,
                }
            )
    rmse_diff = paired_bootstrap_diff(
        rmse,
        y,
        predictions["prediction"],
        predictions["linear_baseline"],
        N_RESAMPLES,
        BOOTSTRAP_SEED,
    )

    # 5. Interval coverage, overall and per predicted bin (Wilson intervals: few engines).
    covered = (predictions["lower"] <= y) & (y <= predictions["upper"])
    width = predictions["upper"] - predictions["lower"]
    by_bin = []
    for label in labels:
        in_bin = predictions["pred_bin"] == label
        n, k = int(in_bin.sum()), int(covered[in_bin].sum())
        low, high = wilson_interval(k, n)
        by_bin.append(
            {
                "bin": label,
                "engines": n,
                "covered": k,
                "coverage": k / n if n else math.nan,
                "low": low,
                "high": high,
                "mean_width": width[in_bin].mean() if n else math.nan,
            }
        )
    n_all, k_all = len(predictions), int(covered.sum())
    low_all, high_all = wilson_interval(k_all, n_all)
    overall = {
        "engines": n_all,
        "covered": k_all,
        "coverage": k_all / n_all,
        "low": low_all,
        "high": high_all,
        "mean_width": float(width.mean()),
    }

    return FinalResult(
        subset=subset,
        predictions=predictions,
        point_metrics=pd.DataFrame(rows),
        rmse_diff=rmse_diff,
        q_by_bin=q_by_bin,
        coverage_by_bin=pd.DataFrame(by_bin),
        overall_coverage=overall,
        forced=forced,
    )


def format_markdown(result: FinalResult, git: dict[str, str]) -> str:
    """The results report written to reports/final_test_<subset>.md."""
    pm = result.point_metrics
    ci = "{:.2f} [{:.2f}, {:.2f}]"
    lines = [
        f"# Final test-set evaluation: {result.subset}",
        "",
        f"- **Date (UTC):** {datetime.now(UTC):%Y-%m-%d %H:%M}",
        f"- **Commit:** `{git['git_commit']}` (uncommitted changes: {git['git_dirty']})",
        f"- **Test engines:** {len(result.predictions)}, each evaluated at its last observed "
        "cycle against the uncapped true RUL",
        "- **Frozen setup:**",
        f"  - LightGBM on causal window features (windows {', '.join(map(str, DEFAULT_WINDOWS))}"
        f"), training target capped at {DEFAULT_RUL_CAP}, trained on all training engines",
        "    (ADRs 0001 and 0002).",
        f"  - Intervals: Mondrian split conformal with alpha {ALPHA}, calibrated on the "
        f"{N_SPLITS}-fold out-of-fold",
        f"    residuals of the training engines (rows with true RUL <= {MAX_EVAL_RUL}; ADR 0003).",
        f"- **Uncertainty:** 95% bootstrap intervals over engines ({N_RESAMPLES:,} resamples, "
        f"seed {BOOTSTRAP_SEED}).",
    ]
    if result.forced:
        lines += ["", "> **Forced re-run (`--force`)**: the test set had been evaluated before."]
    lines += [
        "",
        "## Point predictions",
        "",
        "| Model | RMSE | MAE | PHM08 score |",
        "|---|---|---|---|",
    ]
    for model_name in pm["model"].unique():
        cells = []
        for metric_name in METRICS:
            r = pm[(pm["model"] == model_name) & (pm["metric"] == metric_name)].iloc[0]
            cells.append(ci.format(r["value"], r["low"], r["high"]))
        lines.append(f"| {model_name} | " + " | ".join(cells) + " |")
    d = result.rmse_diff
    lines += [
        "",
        f"**RMSE difference, LightGBM minus linear (paired bootstrap):** {ci.format(*d)} cycles. "
        + (
            "The interval is entirely below 0, so LightGBM is better on these engines."
            if d[2] < 0
            else "The interval includes 0, so these engines do not show a clear difference."
        ),
        "",
        "RMSE and MAE are in cycles. The PHM08 score is a sum over engines (lower is better).",
        "",
        "## Prediction intervals (90% target)",
        "",
    ]
    o = result.overall_coverage
    lines += [
        f"Overall: **{o['covered']}/{o['engines']} engines covered ({o['coverage']:.1%}, 95% "
        f"Wilson interval {o['low']:.1%}–{o['high']:.1%})**, mean width {o['mean_width']:.1f} "
        "cycles.",
        "",
        "| Predicted RUL | Engines | Covered | Coverage [95% Wilson] | Mean width | q (cycles) |",
        "|---|---|---|---|---|---|",
    ]
    q = result.q_by_bin.set_index("bin")
    for _, r in result.coverage_by_bin.iterrows():
        coverage = f"{r['coverage']:.0%} [{r['low']:.0%}, {r['high']:.0%}]" if r["engines"] else "–"
        width = f"{r['mean_width']:.1f}" if r["engines"] else "–"
        fallback = " (global fallback)" if q.loc[r["bin"], "fallback"] else ""
        lines.append(
            f"| {r['bin']} | {r['engines']} | {r['covered']} | {coverage} | {width} | "
            f"{q.loc[r['bin'], 'q']:.2f}{fallback} |"
        )
    lines += [
        "",
        "**Uncertainty:** with 100 engines, and fewer in each bin, coverage is estimated "
        "coarsely. For example, 18 of 20 engines covered (90%) has a 95% Wilson interval of "
        "about 70%–97%. Per-bin deviations from 90% of that size are expected noise, not "
        "evidence of miscalibration.",
        "",
        f"Per-engine predictions: `test_predictions_{result.subset.lower()}.csv`.",
        "",
    ]
    return "\n".join(lines)


def final_metrics(result: FinalResult) -> dict[str, float]:
    """Flat metric dict for MLflow."""
    short = {
        "LightGBM + window features": "lgbm",
        "Linear regression (current sensors)": "linear",
        "Mean RUL baseline": "mean",
    }
    metrics: dict[str, float] = {}
    for _, r in result.point_metrics.iterrows():
        key = f"test_{short[r['model']]}_{r['metric'].lower()}"
        metrics[key], metrics[f"{key}_ci_low"], metrics[f"{key}_ci_high"] = (
            r["value"],
            r["low"],
            r["high"],
        )
    d = result.rmse_diff
    metrics["test_rmse_diff_lgbm_minus_linear"] = d[0]
    metrics["test_rmse_diff_ci_low"], metrics["test_rmse_diff_ci_high"] = d[1], d[2]
    o = result.overall_coverage
    metrics["test_interval_coverage"] = o["coverage"]
    metrics["test_interval_coverage_ci_low"] = o["low"]
    metrics["test_interval_coverage_ci_high"] = o["high"]
    metrics["test_interval_mean_width"] = o["mean_width"]
    for _, r in result.coverage_by_bin.iterrows():
        key = r["bin"].replace("-", "_").replace("+", "_plus")
        metrics[f"test_interval_engines_pred_{key}"] = r["engines"]
        if r["engines"]:
            metrics[f"test_interval_coverage_pred_{key}"] = r["coverage"]
            metrics[f"test_interval_width_pred_{key}"] = r["mean_width"]
    return metrics


def run_final_eval(
    subset: str,
    *,
    force: bool = False,
    report_dir: Path = REPORTS_DIR,
    tracking_uri: str | None = None,
) -> tuple[FinalResult, Path, Path, str]:
    """Guard, evaluate, write the report and predictions, then log to MLflow.

    The guard runs first, before any data is read. The report is written only after the whole
    evaluation succeeded, so a crash leaves no report and does not use up the one-time run.
    """
    report, predictions_csv = report_paths(subset, report_dir)
    check_not_already_run(report, force)

    from predmaint.tracking import git_tags, log_run  # slow import

    forced = report.exists()
    result = evaluate(subset, forced=forced)
    git = git_tags()

    report_dir.mkdir(parents=True, exist_ok=True)
    result.predictions[["unit", "true_rul", "prediction", "lower", "upper"]].to_csv(
        predictions_csv, index=False, float_format="%.3f"
    )
    report.write_text(format_markdown(result, git))

    params = {
        "subset": subset,
        "model": "LightGBM",
        "feature_set": "window",
        "windows": ",".join(map(str, DEFAULT_WINDOWS)),
        "cap": DEFAULT_RUL_CAP,
        "conformal_method": "mondrian",
        "conformal_alpha": ALPHA,
        "mondrian_bins": ",".join(mondrian_bin_labels(MONDRIAN_EDGES)),
        "calibration": f"{N_SPLITS}-fold OOF, rows with true RUL <= {MAX_EVAL_RUL}",
        "bootstrap_resamples": N_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_test_engines": len(result.predictions),
    }
    run_id = log_run(
        run_name=f"FINAL TEST {subset} | LightGBM | window | Mondrian",
        experiment_name=f"rul-{subset.lower()}",
        params=params,
        metrics=final_metrics(result),
        tags={"stage": "final_test", "forced_rerun": str(forced).lower()},
        artifacts=[report, predictions_csv],
        **({"tracking_uri": tracking_uri} if tracking_uri else {}),
    )
    return result, report, predictions_csv, run_id


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="One-time evaluation on the official test set.")
    parser.add_argument("--subset", required=True, choices=SUBSETS_WITH_ADR)
    parser.add_argument(
        "--force", action="store_true", help="re-run even though a report exists (deliberate only)"
    )
    args = parser.parse_args(argv)
    try:
        _, report, predictions_csv, run_id = run_final_eval(
            args.subset, force=args.force, report_dir=REPORTS_DIR
        )
    except FinalEvalAlreadyDone as exc:
        print(f"Refusing to run: {exc}", file=sys.stderr)
        return 1
    print(report.read_text())
    print(f"Wrote {report} and {predictions_csv}; MLflow run {run_id} (tag stage=final_test)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
