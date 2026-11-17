"""Honest evaluation of conformal intervals with cross-fold calibration.

For each fold k, the interval half-width q is computed from the out-of-fold residuals of the
**other** folds only, then coverage and width are measured on fold k. Fold k's own residuals never
set its own q, so the evaluation does not grade its own homework.

Only rows with true RUL <= ``max_eval_rul`` (150, docs/evaluation.md) are used, both for
calibration and for evaluation, so the two describe the same population.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from predmaint.models.conformal import conformal_quantile, interval

RUL_BUCKET_EDGES = (0, 25, 50, 75, 100, 150)


@dataclass
class CoverageReport:
    """Results of :func:`cross_fold_coverage`.

    Attributes:
        alpha: Miscoverage rate; target coverage is ``1 - alpha``.
        per_fold: One row per fold: ``fold``, ``q``, ``n``, ``coverage``, ``mean_width``.
        by_bucket: One row per true-RUL bucket: ``bucket``, ``n``, ``coverage``, ``mean_width``.
            Buckets are right-closed: [0, 25], (25, 50], (50, 75], (75, 100], (100, 150].
        overall: ``coverage`` and ``mean_width`` over all evaluated rows, plus ``n``.
        rows: Every evaluated row with its fold, ``rul``, ``pred``, ``residual``, ``lower``,
            ``upper``, ``covered``, ``width`` and ``bucket``.
    """

    alpha: float
    per_fold: pd.DataFrame
    by_bucket: pd.DataFrame
    overall: dict[str, float]
    rows: pd.DataFrame


def cross_fold_coverage(
    oof: pd.DataFrame,
    alpha: float = 0.1,
    *,
    max_eval_rul: float = 150,
    bucket_edges: Sequence[float] = RUL_BUCKET_EDGES,
) -> CoverageReport:
    """Coverage and width of split-conformal intervals, calibrating each fold on the others.

    Args:
        oof: Out-of-fold predictions with ``fold``, ``rul`` (uncapped true RUL) and ``pred``,
            e.g. ``CVResult.oof``.
        alpha: Miscoverage rate; 0.1 targets 90% coverage.
        max_eval_rul: Rows with larger true RUL are ignored for calibration and evaluation.
        bucket_edges: Edges of the true-RUL buckets for the conditional-coverage table.

    Raises:
        ValueError: On missing columns or fewer than two folds.
    """
    missing = {"fold", "rul", "pred"} - set(oof.columns)
    if missing:
        raise ValueError(f"oof is missing columns: {sorted(missing)}")
    data = oof[oof["rul"] <= max_eval_rul].copy()
    folds = sorted(data["fold"].unique())
    if len(folds) < 2:
        raise ValueError("need at least two folds to calibrate on the other folds")

    data["residual"] = data["rul"] - data["pred"]
    per_fold, evaluated = [], []
    for k in folds:
        test = data[data["fold"] == k].copy()
        q = conformal_quantile(data.loc[data["fold"] != k, "residual"], alpha)
        test["lower"], test["upper"] = interval(test["pred"], q)
        test["covered"] = (test["lower"] <= test["rul"]) & (test["rul"] <= test["upper"])
        test["width"] = test["upper"] - test["lower"]
        per_fold.append(
            {
                "fold": k,
                "q": q,
                "n": len(test),
                "coverage": test["covered"].mean(),
                "mean_width": test["width"].mean(),
            }
        )
        evaluated.append(test)

    rows = pd.concat(evaluated)
    labels = [f"{lo:g}-{hi:g}" for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:], strict=True)]
    rows["bucket"] = pd.cut(
        rows["rul"], bins=list(bucket_edges), labels=labels, right=True, include_lowest=True
    )
    by_bucket = (
        rows.groupby("bucket", observed=False)
        .agg(n=("covered", "size"), coverage=("covered", "mean"), mean_width=("width", "mean"))
        .reset_index()
    )
    by_bucket["bucket"] = by_bucket["bucket"].astype(str)
    overall = {
        "n": float(len(rows)),
        "coverage": float(rows["covered"].mean()),
        "mean_width": float(rows["width"].mean()),
    }
    return CoverageReport(
        alpha=alpha,
        per_fold=pd.DataFrame(per_fold),
        by_bucket=by_bucket,
        overall=overall,
        rows=rows,
    )


def coverage_metrics(report: CoverageReport) -> dict[str, float | list[float]]:
    """Flatten a report into MLflow metrics (lists are logged with step = fold)."""
    metrics: dict[str, float | list[float]] = {
        "conformal_coverage": report.overall["coverage"],
        "conformal_mean_width": report.overall["mean_width"],
        "conformal_q_mean": float(np.mean(report.per_fold["q"])),
        "conformal_coverage_fold": report.per_fold["coverage"].tolist(),
        "conformal_q_fold": report.per_fold["q"].tolist(),
        "conformal_width_fold": report.per_fold["mean_width"].tolist(),
    }
    for _, row in report.by_bucket.iterrows():
        key = row["bucket"].replace("-", "_")
        if row["n"] > 0:
            metrics[f"conformal_coverage_rul_{key}"] = float(row["coverage"])
            metrics[f"conformal_width_rul_{key}"] = float(row["mean_width"])
    return metrics
