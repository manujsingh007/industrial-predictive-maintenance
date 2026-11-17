"""Honest evaluation of conformal intervals with cross-fold calibration.

For each fold k, the conformal correction q (one per predicted-RUL bin for Mondrian) is computed
from the out-of-fold conformity scores of the **other** folds only, then coverage and width are
measured on fold k. Fold k's own scores never set its own q, so the evaluation does not grade
its own homework.

Only rows with true RUL <= ``max_eval_rul`` (150, docs/evaluation.md) are used, both for
calibration and for evaluation, so the two describe the same population.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from predmaint.models.conformal import (
    MIN_BIN_CALIBRATION,
    MONDRIAN_EDGES,
    conformal_score_quantile,
    cqr_interval,
    cqr_scores,
    interval,
    mondrian_bin_labels,
    mondrian_bins,
    mondrian_interval,
    mondrian_quantiles,
)

RUL_BUCKET_EDGES = (0, 25, 50, 75, 100, 150)
METHODS = ("split", "cqr", "mondrian")
_REQUIRED = {
    "split": {"fold", "rul", "pred"},
    "cqr": {"fold", "rul", "lo", "hi"},
    "mondrian": {"fold", "rul", "pred"},
}


@dataclass
class CoverageReport:
    """Results of :func:`cross_fold_coverage`.

    Attributes:
        alpha: Miscoverage rate; target coverage is ``1 - alpha``.
        method: ``"split"``, ``"cqr"`` or ``"mondrian"``.
        per_fold: One row per fold: ``fold``, ``q`` (NaN for Mondrian, see ``q_by_bin``),
            ``n``, ``coverage``, ``mean_width``.
        by_bucket: Per **true**-RUL bucket: ``bucket``, ``n``, ``coverage``, ``mean_width``.
            Buckets are right-closed: [0, 25], (25, 50], (50, 75], (75, 100], (100, 150].
        by_pred_bin: Per **predicted**-RUL bin of the point prediction ``pred`` (left-closed:
            [0, 30), [30, 60), [60, 90), [90, inf)); empty if ``pred`` is absent.
        q_by_bin: Mondrian only: one row per (fold, bin) with ``q``, ``n_calibration`` and
            ``fallback`` (True if the bin used the global q); empty for other methods.
        overall: ``coverage`` and ``mean_width`` over all evaluated rows, plus ``n``.
        rows: Every evaluated row with ``lower``, ``upper``, ``covered``, ``width``,
            ``bucket`` and (if ``pred`` is present) ``pred_bin``.
    """

    alpha: float
    method: str
    per_fold: pd.DataFrame
    by_bucket: pd.DataFrame
    by_pred_bin: pd.DataFrame
    q_by_bin: pd.DataFrame
    overall: dict[str, float]
    rows: pd.DataFrame


def _summarise(rows: pd.DataFrame, by: str) -> pd.DataFrame:
    table = (
        rows.groupby(by, observed=False)
        .agg(n=("covered", "size"), coverage=("covered", "mean"), mean_width=("width", "mean"))
        .reset_index()
    )
    table[by] = table[by].astype(str)
    return table


def cross_fold_coverage(
    oof: pd.DataFrame,
    alpha: float = 0.1,
    *,
    method: str = "split",
    max_eval_rul: float = 150,
    bucket_edges: Sequence[float] = RUL_BUCKET_EDGES,
    pred_bin_edges: tuple[float, ...] = MONDRIAN_EDGES,
    min_bin_size: int = MIN_BIN_CALIBRATION,
) -> CoverageReport:
    """Coverage and width of conformal intervals, calibrating each fold on the others.

    Args:
        oof: Out-of-fold predictions with ``fold`` and ``rul`` (uncapped true RUL), plus
            ``pred`` (split, Mondrian) and/or ``lo``/``hi`` quantile predictions (CQR).
        alpha: Miscoverage rate; 0.1 targets 90% coverage.
        method: ``"split"`` (pred ± q), ``"cqr"`` ([lo - q, hi + q]) or ``"mondrian"``
            (pred ± q of the row's predicted-RUL bin).
        max_eval_rul: Rows with larger true RUL are ignored for calibration and evaluation.
        bucket_edges: Edges of the true-RUL buckets.
        pred_bin_edges: Edges of the predicted-RUL bins (Mondrian bins and report table).
        min_bin_size: Mondrian: bins with fewer calibration rows use the global q.

    Raises:
        ValueError: On an unknown method, missing columns or fewer than two folds.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    missing = _REQUIRED[method] - set(oof.columns)
    if missing:
        raise ValueError(f"oof is missing columns: {sorted(missing)}")
    data = oof[oof["rul"] <= max_eval_rul].copy()
    folds = sorted(data["fold"].unique())
    if len(folds) < 2:
        raise ValueError("need at least two folds to calibrate on the other folds")

    if method == "cqr":
        data["score"] = cqr_scores(data["rul"], data["lo"], data["hi"])
    else:
        data["score"] = np.abs(data["rul"] - data["pred"])

    bin_labels = mondrian_bin_labels(pred_bin_edges)
    per_fold, evaluated, q_rows = [], [], []
    for k in folds:
        test = data[data["fold"] == k].copy()
        calibration = data[data["fold"] != k]
        q = np.nan
        if method == "split":
            q = conformal_score_quantile(calibration["score"], alpha)
            test["lower"], test["upper"] = interval(test["pred"], q)
        elif method == "cqr":
            q = conformal_score_quantile(calibration["score"], alpha)
            test["lower"], test["upper"] = cqr_interval(test["lo"], test["hi"], q)
        else:
            q_bins, fallback = mondrian_quantiles(
                calibration["score"], calibration["pred"], alpha, pred_bin_edges, min_bin_size
            )
            test["lower"], test["upper"] = mondrian_interval(test["pred"], q_bins, pred_bin_edges)
            counts = np.bincount(
                mondrian_bins(calibration["pred"], pred_bin_edges), minlength=len(bin_labels)
            )
            q_rows += [
                {"fold": k, "bin": label, "q": qb, "n_calibration": int(n), "fallback": bool(fb)}
                for label, qb, n, fb in zip(bin_labels, q_bins, counts, fallback, strict=True)
            ]
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
    by_pred_bin = pd.DataFrame(columns=["pred_bin", "n", "coverage", "mean_width"])
    if "pred" in rows.columns:
        rows["pred_bin"] = pd.Categorical.from_codes(
            mondrian_bins(rows["pred"], pred_bin_edges), categories=bin_labels
        )
        by_pred_bin = _summarise(rows, "pred_bin")

    return CoverageReport(
        alpha=alpha,
        method=method,
        per_fold=pd.DataFrame(per_fold),
        by_bucket=_summarise(rows, "bucket"),
        by_pred_bin=by_pred_bin,
        q_by_bin=pd.DataFrame(q_rows, columns=["fold", "bin", "q", "n_calibration", "fallback"]),
        overall={
            "n": float(len(rows)),
            "coverage": float(rows["covered"].mean()),
            "mean_width": float(rows["width"].mean()),
        },
        rows=rows,
    )


def _metric_key(label: str) -> str:
    return label.replace("-", "_").replace("+", "_plus")


def coverage_metrics(report: CoverageReport) -> dict[str, float | list[float]]:
    """Flatten a report into MLflow metrics (lists are logged with step = fold)."""
    metrics: dict[str, float | list[float]] = {
        "conformal_coverage": report.overall["coverage"],
        "conformal_mean_width": report.overall["mean_width"],
        "conformal_coverage_fold": report.per_fold["coverage"].tolist(),
        "conformal_width_fold": report.per_fold["mean_width"].tolist(),
    }
    if report.method == "mondrian":
        for label, group in report.q_by_bin.groupby("bin", sort=False):
            metrics[f"conformal_q_bin_{_metric_key(label)}_fold"] = group["q"].tolist()
        metrics["conformal_fallback_bins"] = float(report.q_by_bin["fallback"].sum())
    else:
        metrics["conformal_q_mean"] = float(np.mean(report.per_fold["q"]))
        metrics["conformal_q_fold"] = report.per_fold["q"].tolist()
    for table, column, prefix in (
        (report.by_bucket, "bucket", "rul"),
        (report.by_pred_bin, "pred_bin", "pred"),
    ):
        for _, row in table.iterrows():
            if row["n"] > 0:
                key = _metric_key(row[column])
                metrics[f"conformal_coverage_{prefix}_{key}"] = float(row["coverage"])
                metrics[f"conformal_width_{prefix}_{key}"] = float(row["mean_width"])
    return metrics
