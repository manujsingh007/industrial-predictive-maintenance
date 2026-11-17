"""Evaluate conformal RUL intervals for the primary model (ADR 0002) with cross-fold calibration.

Run with: ``uv run python -m predmaint.training.run_conformal --subset FD001 [--method both]``

Model and features follow ADR 0002: LightGBM on causal window features (windows 10 and 30),
target capped at 125. Two methods are compared:

- ``split``: split conformal around the point model's prediction (constant width).
- ``cqr``: conformalized quantile regression on two LightGBM quantile models (adaptive width).

Every model is cross-validated on the same seed-42 engine folds. Only training data is used;
the test files are never read.
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd
from sklearn.base import RegressorMixin

from predmaint.data.load import add_rul, load_train
from predmaint.evaluation.coverage import CoverageReport, coverage_metrics, cross_fold_coverage
from predmaint.features.selection import DEFAULT_RUL_CAP, SUBSETS_WITH_ADR
from predmaint.models.conformal import CQR_QUANTILES
from predmaint.models.gbdt import make_lgbm, make_lgbm_quantile
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
QUANTILE_FACTORY: Callable[[float], RegressorMixin] = make_lgbm_quantile
DEFAULT_ALPHA = 0.1
METHOD_CHOICES = ("split", "cqr", "both")
METHOD_LABELS = {"split": "split conformal", "cqr": "CQR"}


@dataclass
class ConformalRun:
    """Outputs of :func:`run`.

    Attributes:
        point: CV result of the point model; its ``oof`` gains ``lo``/``hi`` columns when CQR
            is evaluated.
        reports: Coverage report per evaluated method (``"split"``, ``"cqr"``).
        run_ids: MLflow run id per method (empty when not logged).
    """

    point: CVResult
    reports: dict[str, CoverageReport]
    run_ids: dict[str, str] = field(default_factory=dict)


def _quantile_oof(data: pd.DataFrame, features: list[str], point: CVResult) -> pd.DataFrame:
    """Out-of-fold predictions of the two CQR quantile models, merged into the point oof."""
    oof = point.oof.copy()
    for column, quantile in zip(("lo", "hi"), CQR_QUANTILES, strict=True):
        result = cross_validate(
            lambda q=quantile: QUANTILE_FACTORY(q),
            data,
            cap=DEFAULT_RUL_CAP,
            features=features,
            n_splits=N_SPLITS,
            seed=SEED,
        )
        same_rows = result.oof.index.equals(oof.index)
        if not same_rows or not result.oof["fold"].equals(oof["fold"]):
            raise RuntimeError("quantile models were not cross-validated on the same folds")
        oof[column] = result.oof["pred"]
    return oof


def run(
    subset: str,
    alpha: float = DEFAULT_ALPHA,
    *,
    method: str = "both",
    log: bool = True,
    tracking_uri: str | None = None,
) -> ConformalRun:
    """Cross-validate the models, evaluate conformal coverage per method and optionally log.

    Each evaluated method is logged as its own MLflow run, so the two can be compared side by
    side in the UI (same metric names, distinguished by the ``conformal_method`` param).
    """
    if method not in METHOD_CHOICES:
        raise ValueError(f"unknown method {method!r}; expected one of {METHOD_CHOICES}")
    methods = ["split", "cqr"] if method == "both" else [method]

    data, features = build_features(add_rul(load_train(subset)), "window", DEFAULT_WINDOWS)
    point = cross_validate(
        MODEL_FACTORY, data, cap=DEFAULT_RUL_CAP, features=features, n_splits=N_SPLITS, seed=SEED
    )
    if "cqr" in methods:
        point.oof = _quantile_oof(data, features, point)

    reports = {
        m: cross_fold_coverage(point.oof, alpha, method=m, max_eval_rul=MAX_EVAL_RUL)
        for m in methods
    }

    run_ids: dict[str, str] = {}
    if log:
        from predmaint.tracking import log_cv_result  # slow import; skipped for quick runs

        for m, report in reports.items():
            extra_params = {
                "feature_set": "window",
                "windows": ",".join(map(str, DEFAULT_WINDOWS)),
                "conformal_alpha": alpha,
                "conformal_method": m,
                "conformal_calibration": "other folds",
                **model_hyperparams("lgbm", point),
            }
            if m == "cqr":
                extra_params["cqr_quantiles"] = ",".join(map(str, CQR_QUANTILES))
            run_ids[m] = log_cv_result(
                point,
                model_name=MODEL_NAME,
                subset=subset,
                run_name=f"{MODEL_NAME} | window | {METHOD_LABELS[m]} alpha={alpha:g}",
                extra_params=extra_params,
                extra_metrics=coverage_metrics(report),
                **({"tracking_uri": tracking_uri} if tracking_uri else {}),
            )
    return ConformalRun(point=point, reports=reports, run_ids=run_ids)


def format_per_fold(report: CoverageReport) -> str:
    """Per-fold table for one method."""
    table = pd.DataFrame(
        {
            "fold": report.per_fold["fold"],
            "q (cycles)": report.per_fold["q"].map("{:.2f}".format),
            "rows": report.per_fold["n"],
            "coverage": report.per_fold["coverage"].map("{:.1%}".format),
            "mean width": report.per_fold["mean_width"].map("{:.1f}".format),
        }
    )
    return f"{METHOD_LABELS[report.method]}, per fold (q from the other folds):\n" + (
        table.to_string(index=False)
    )


def format_by_bucket(reports: dict[str, CoverageReport]) -> str:
    """Coverage and mean width per true-RUL bucket, one column pair per method."""
    first = next(iter(reports.values()))
    table = pd.DataFrame({"true RUL": first.by_bucket["bucket"], "rows": first.by_bucket["n"]})
    for m, report in reports.items():
        label = METHOD_LABELS[m]
        table[f"{label} coverage"] = report.by_bucket["coverage"].map("{:.1%}".format)
        table[f"{label} width"] = report.by_bucket["mean_width"].map("{:.1f}".format)
    overall = {"true RUL": "all", "rows": int(first.overall["n"])}
    for m, report in reports.items():
        overall[f"{METHOD_LABELS[m]} coverage"] = f"{report.overall['coverage']:.1%}"
        overall[f"{METHOD_LABELS[m]} width"] = f"{report.overall['mean_width']:.1f}"
    table = pd.concat([table, pd.DataFrame([overall])], ignore_index=True)
    return "By true RUL (marginal guarantee only; buckets can differ):\n" + table.to_string(
        index=False
    )


def format_report(run_result: ConformalRun) -> str:
    """Full text report: target, per-fold tables, then the side-by-side bucket table."""
    alpha = next(iter(run_result.reports.values())).alpha
    parts = [f"Target coverage: {1 - alpha:.0%} (alpha = {alpha:g}); widths in cycles"]
    parts += [format_per_fold(r) for r in run_result.reports.values()]
    parts.append(format_by_bucket(run_result.reports))
    return "\n\n".join(parts)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Evaluate conformal RUL intervals.")
    parser.add_argument("--subset", required=True, choices=SUBSETS_WITH_ADR)
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA, help="miscoverage rate (0.1 = 90%%)"
    )
    parser.add_argument(
        "--method", choices=METHOD_CHOICES, default="both", help="interval method(s) to evaluate"
    )
    parser.add_argument("--no-mlflow", action="store_true", help="do not log runs to MLflow")
    args = parser.parse_args()

    result = run(args.subset, args.alpha, method=args.method, log=not args.no_mlflow)
    print(
        f"{args.subset} | {MODEL_NAME} | window features | {N_SPLITS}-fold engine CV (seed {SEED})"
        f", cap {DEFAULT_RUL_CAP}, rows with true RUL <= {MAX_EVAL_RUL}\n"
    )
    print(format_report(result))
    for m, run_id in result.run_ids.items():
        print(f"MLflow run ({METHOD_LABELS[m]}): {run_id}")


if __name__ == "__main__":
    main()
