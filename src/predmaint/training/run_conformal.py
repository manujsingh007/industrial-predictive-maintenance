"""Evaluate split-conformal intervals for the primary model (ADR 0002) with cross-fold calibration.

Run with: ``uv run python -m predmaint.training.run_conformal --subset FD001``

Model and features follow ADR 0002: LightGBM on causal window features (windows 10 and 30),
target capped at 125. Only training data is used; the test files are never read.
"""

import argparse
from collections.abc import Callable

import pandas as pd
from sklearn.base import RegressorMixin

from predmaint.data.load import add_rul, load_train
from predmaint.evaluation.coverage import CoverageReport, coverage_metrics, cross_fold_coverage
from predmaint.features.selection import DEFAULT_RUL_CAP, SUBSETS_WITH_ADR
from predmaint.models.gbdt import make_lgbm
from predmaint.training.cv import MAX_EVAL_RUL, CVResult, cross_validate
from predmaint.training.run_cv import (
    DEFAULT_WINDOWS,
    N_SPLITS,
    SEED,
    build_features,
    model_hyperparams,
)

MODEL_NAME = "LightGBM"
MODEL_FACTORY: Callable[[], RegressorMixin] = make_lgbm
DEFAULT_ALPHA = 0.1


def run(
    subset: str,
    alpha: float = DEFAULT_ALPHA,
    *,
    log: bool = True,
    tracking_uri: str | None = None,
) -> tuple[CVResult, CoverageReport, str | None]:
    """Cross-validate the primary model, evaluate conformal coverage and optionally log it.

    Returns:
        The CV result (with out-of-fold predictions), the coverage report and the MLflow run id
        (None when not logged).
    """
    data, features = build_features(add_rul(load_train(subset)), "window", DEFAULT_WINDOWS)
    result = cross_validate(
        MODEL_FACTORY, data, cap=DEFAULT_RUL_CAP, features=features, n_splits=N_SPLITS, seed=SEED
    )
    report = cross_fold_coverage(result.oof, alpha, max_eval_rul=MAX_EVAL_RUL)

    run_id = None
    if log:
        from predmaint.tracking import log_cv_result  # slow import; skipped for quick runs

        run_id = log_cv_result(
            result,
            model_name=MODEL_NAME,
            subset=subset,
            run_name=f"{MODEL_NAME} | window | conformal alpha={alpha:g}",
            extra_params={
                "feature_set": "window",
                "windows": ",".join(map(str, DEFAULT_WINDOWS)),
                "conformal_alpha": alpha,
                "conformal_method": "split, calibrated on other folds",
                **model_hyperparams("lgbm", result),
            },
            extra_metrics=coverage_metrics(report),
            **({"tracking_uri": tracking_uri} if tracking_uri else {}),
        )
    return result, report, run_id


def format_report(report: CoverageReport) -> str:
    """Per-fold and per-RUL-bucket coverage tables as text."""
    target = 1 - report.alpha
    per_fold = pd.DataFrame(
        {
            "fold": report.per_fold["fold"],
            "q (cycles)": report.per_fold["q"].map("{:.2f}".format),
            "rows": report.per_fold["n"],
            "coverage": report.per_fold["coverage"].map("{:.1%}".format),
            "mean width": report.per_fold["mean_width"].map("{:.1f}".format),
        }
    )
    by_bucket = pd.DataFrame(
        {
            "true RUL": report.by_bucket["bucket"],
            "rows": report.by_bucket["n"],
            "coverage": report.by_bucket["coverage"].map("{:.1%}".format),
            "mean width": report.by_bucket["mean_width"].map("{:.1f}".format),
        }
    )
    o = report.overall
    return (
        f"Target coverage: {target:.0%} (alpha = {report.alpha:g})\n\n"
        f"Per fold (q calibrated on the other folds):\n{per_fold.to_string(index=False)}\n\n"
        f"By true RUL (marginal guarantee only; buckets can differ):\n"
        f"{by_bucket.to_string(index=False)}\n\n"
        f"Overall: coverage {o['coverage']:.1%} over {int(o['n'])} rows, "
        f"mean width {o['mean_width']:.1f} cycles"
    )


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Evaluate conformal RUL intervals.")
    parser.add_argument("--subset", required=True, choices=SUBSETS_WITH_ADR)
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA, help="miscoverage rate (0.1 = 90%%)"
    )
    parser.add_argument("--no-mlflow", action="store_true", help="do not log the run to MLflow")
    args = parser.parse_args()

    _, report, run_id = run(args.subset, args.alpha, log=not args.no_mlflow)
    print(
        f"{args.subset} | {MODEL_NAME} | window features | {N_SPLITS}-fold engine CV (seed {SEED})"
        f", cap {DEFAULT_RUL_CAP}, rows with true RUL <= {MAX_EVAL_RUL}\n"
    )
    print(format_report(report))
    if run_id:
        print(f"\nMLflow run id: {run_id}")


if __name__ == "__main__":
    main()
