"""Evaluate conformal RUL intervals for the primary model (ADR 0002) with cross-fold calibration.

Run with: ``uv run python -m predmaint.training.run_conformal --subset FD001 [--method all]``

Model and features follow ADR 0002: LightGBM on causal window features (windows 10 and 30),
target capped at 125. Two methods are compared:

- ``split``: split conformal around the point model's prediction (constant width).
- ``cqr``: conformalized quantile regression on two LightGBM quantile models (adaptive width).
- ``mondrian``: split conformal with one q per predicted-RUL bin (width depends on the bin).

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
from predmaint.models.conformal import CQR_QUANTILES, MIN_BIN_CALIBRATION, mondrian_bin_labels
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
METHOD_CHOICES = ("split", "cqr", "mondrian", "both", "all")
METHOD_GROUPS = {"both": ["split", "cqr"], "all": ["split", "cqr", "mondrian"]}
METHOD_LABELS = {"split": "split", "cqr": "CQR", "mondrian": "Mondrian"}


@dataclass
class ConformalRun:
    """Outputs of :func:`run`.

    Attributes:
        point: CV result of the point model; its ``oof`` gains ``lo``/``hi`` columns when CQR
            is evaluated.
        reports: Coverage report per evaluated method (``"split"``, ``"cqr"``, ``"mondrian"``).
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
    method: str = "all",
    log: bool = True,
    tracking_uri: str | None = None,
) -> ConformalRun:
    """Cross-validate the models, evaluate conformal coverage per method and optionally log.

    Each evaluated method is logged as its own MLflow run, so the two can be compared side by
    side in the UI (same metric names, distinguished by the ``conformal_method`` param).
    """
    if method not in METHOD_CHOICES:
        raise ValueError(f"unknown method {method!r}; expected one of {METHOD_CHOICES}")
    methods = METHOD_GROUPS.get(method, [method])

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
            if m == "mondrian":
                extra_params["mondrian_bins"] = ",".join(mondrian_bin_labels())
                extra_params["mondrian_min_bin_size"] = MIN_BIN_CALIBRATION
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
    """Per-fold table for one method; Mondrian adds one q column per predicted-RUL bin."""
    table = pd.DataFrame(
        {
            "fold": report.per_fold["fold"],
            "rows": report.per_fold["n"],
            "coverage": report.per_fold["coverage"].map("{:.1%}".format),
            "mean width": report.per_fold["mean_width"].map("{:.1f}".format),
        }
    )
    if report.method == "mondrian":
        q = report.q_by_bin.assign(
            text=report.q_by_bin["q"].map("{:.2f}".format)
            + report.q_by_bin["fallback"].map({True: "*", False: ""})
        ).pivot(index="fold", columns="bin", values="text")
        for label in mondrian_bin_labels():
            table[f"q pred {label}"] = q[label].to_numpy()
        note = "\n* bin had too few calibration rows; global q used"
    else:
        table.insert(1, "q", report.per_fold["q"].map("{:.2f}".format).to_numpy())
        note = ""
    return (
        f"{METHOD_LABELS[report.method]}, per fold (q from the other folds):\n"
        + table.to_string(index=False)
        + (note if report.method == "mondrian" and report.q_by_bin["fallback"].any() else "")
    )


def format_side_by_side(reports: dict[str, CoverageReport], by: str) -> str:
    """Coverage and mean width per group, one column pair per method.

    ``by`` is ``"bucket"`` (true RUL) or ``"pred_bin"`` (the point model's predicted RUL).
    """
    attr, title = {
        "bucket": ("by_bucket", "By TRUE RUL"),
        "pred_bin": ("by_pred_bin", "By PREDICTED RUL bin"),
    }[by]
    first = getattr(next(iter(reports.values())), attr)
    label = "true RUL" if by == "bucket" else "predicted RUL"
    table = pd.DataFrame({label: first[by], "rows": first["n"]})
    for m, report in reports.items():
        group = getattr(report, attr)
        table[f"{METHOD_LABELS[m]} cov"] = group["coverage"].map("{:.1%}".format).to_numpy()
        table[f"{METHOD_LABELS[m]} width"] = group["mean_width"].map("{:.1f}".format).to_numpy()
    overall = {label: "all", "rows": int(next(iter(reports.values())).overall["n"])}
    for m, report in reports.items():
        overall[f"{METHOD_LABELS[m]} cov"] = f"{report.overall['coverage']:.1%}"
        overall[f"{METHOD_LABELS[m]} width"] = f"{report.overall['mean_width']:.1f}"
    table = pd.concat([table, pd.DataFrame([overall])], ignore_index=True)
    return f"{title} (cov = coverage, width in cycles):\n" + table.to_string(index=False)


def format_report(run_result: ConformalRun) -> str:
    """Full text report: target, per-fold tables, then the side-by-side bucket table."""
    alpha = next(iter(run_result.reports.values())).alpha
    parts = [f"Target coverage: {1 - alpha:.0%} (alpha = {alpha:g}); widths in cycles"]
    parts += [format_per_fold(r) for r in run_result.reports.values()]
    parts.append(format_side_by_side(run_result.reports, "bucket"))
    parts.append(format_side_by_side(run_result.reports, "pred_bin"))
    return "\n\n".join(parts)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Evaluate conformal RUL intervals.")
    parser.add_argument("--subset", required=True, choices=SUBSETS_WITH_ADR)
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA, help="miscoverage rate (0.1 = 90%%)"
    )
    parser.add_argument(
        "--method",
        choices=METHOD_CHOICES,
        default="all",
        help="split, cqr, mondrian, both (split + cqr) or all three",
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
