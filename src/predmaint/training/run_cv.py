"""Cross-validate any model / feature-set combination on a training subset (test files unused).

Run with, for example:
``uv run python -m predmaint.training.run_cv --model linear --features window --subset FD001``
"""

import argparse
from collections.abc import Callable, Sequence

import pandas as pd
from sklearn.base import RegressorMixin

from predmaint.data.load import add_rul, load_train
from predmaint.features.selection import DEFAULT_RUL_CAP, KEPT_SENSORS, SUBSETS_WITH_ADR
from predmaint.features.windows import add_window_features, window_feature_names
from predmaint.models.baselines import make_linear_baseline, make_mean_baseline
from predmaint.models.gbdt import LGBM_DEFAULTS, make_lgbm
from predmaint.training.cv import MAX_EVAL_RUL, CVResult, cross_validate

MODELS: dict[str, tuple[str, Callable[[], RegressorMixin]]] = {
    "mean": ("mean RUL", make_mean_baseline),
    "linear": ("linear regression", make_linear_baseline),
    "lgbm": ("LightGBM", make_lgbm),
}


def model_hyperparams(model_key: str, result: CVResult) -> dict[str, object]:
    """Hyperparameters to log, read from the fitted model (empty for the baselines)."""
    if model_key != "lgbm":
        return {}
    params = result.models[0].get_params()
    return {f"lgbm_{k}": params[k] for k in LGBM_DEFAULTS}


FEATURE_SETS = ("current", "window")
DEFAULT_WINDOWS = (10, 30)
N_SPLITS = 5
SEED = 42


def build_features(
    train: pd.DataFrame, feature_set: str, windows: Sequence[int] = DEFAULT_WINDOWS
) -> tuple[pd.DataFrame, list[str]]:
    """Return the data and feature column names for a feature set.

    - ``current``: the 14 kept sensors at the current cycle (ADR 0001).
    - ``window``: the 14 current values plus causal rolling mean, std and slope of each sensor
      over each window.

    Window features depend only on each engine's own past, so computing them before the
    CV split cannot leak information between folds.
    """
    if feature_set == "current":
        return train, list(KEPT_SENSORS)
    if feature_set == "window":
        data = add_window_features(train, KEPT_SENSORS, windows)
        return data, list(KEPT_SENSORS) + window_feature_names(KEPT_SENSORS, windows)
    raise ValueError(f"unknown feature set {feature_set!r}; expected one of {FEATURE_SETS}")


def build_parser() -> argparse.ArgumentParser:
    """Command-line arguments; ``--cap`` and ``--caps`` are mutually exclusive."""
    parser = argparse.ArgumentParser(description="Cross-validate a model and feature set.")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--features", required=True, choices=FEATURE_SETS)
    parser.add_argument("--subset", required=True, choices=SUBSETS_WITH_ADR)
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=list(DEFAULT_WINDOWS),
        help="window lengths in cycles (window feature set only)",
    )
    cap_group = parser.add_mutually_exclusive_group()
    cap_group.add_argument(
        "--cap", type=int, default=DEFAULT_RUL_CAP, help="training RUL cap (cycles)"
    )
    cap_group.add_argument(
        "--caps",
        type=int,
        nargs="+",
        help="sweep: one CV run per cap, e.g. --caps 110 120 125 130 140 150",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="do not log runs to MLflow")
    return parser


def run_caps(
    model_key: str,
    feature_set: str,
    subset: str,
    caps: Sequence[int],
    windows: Sequence[int] = DEFAULT_WINDOWS,
    *,
    log: bool = True,
    tracking_uri: str | None = None,
) -> pd.DataFrame:
    """Cross-validate one model and feature set once per cap; log each cap as its own run.

    Features are built once and reused for every cap; only the training target changes.

    Args:
        model_key: Key of :data:`MODELS`.
        feature_set: One of :data:`FEATURE_SETS`.
        subset: Training subset, e.g. "FD001".
        caps: Caps to evaluate; duplicates are removed and the result is sorted by cap.
        windows: Window lengths for the ``window`` feature set.
        log: Log each run to MLflow.
        tracking_uri: MLflow backend; defaults to the project database (tests pass a temp one).

    Returns:
        One row per cap: ``cap``, ``rmse_mean``, ``rmse_std``, ``mae_mean``, ``mae_std``,
        ``run_id`` (None when not logged).
    """
    caps = sorted(set(caps))
    if not caps or min(caps) <= 0:
        raise ValueError(f"caps must be positive integers, got {caps}")
    model_name, factory = MODELS[model_key]
    data, features = build_features(add_rul(load_train(subset)), feature_set, windows)

    if log:
        from predmaint.tracking import log_cv_result  # slow import; skipped for quick runs

    rows = []
    for cap in caps:
        result = cross_validate(
            factory, data, cap=cap, features=features, n_splits=N_SPLITS, seed=SEED
        )
        s = result.summary()
        run_id = None
        if log:
            extra_params = {
                "feature_set": feature_set,
                "windows": ",".join(map(str, windows)) if feature_set == "window" else "none",
                **model_hyperparams(model_key, result),
            }
            if len(caps) > 1:
                extra_params["cap_sweep"] = ",".join(map(str, caps))
            run_id = log_cv_result(
                result,
                model_name=model_name,
                subset=subset,
                run_name=f"{model_name} | {feature_set} | cap {cap}",
                extra_params=extra_params,
                **({"tracking_uri": tracking_uri} if tracking_uri else {}),
            )
        rows.append(
            {
                "cap": cap,
                "rmse_mean": s.loc["mean", "rmse"],
                "rmse_std": s.loc["std", "rmse"],
                "mae_mean": s.loc["mean", "mae"],
                "mae_std": s.loc["std", "mae"],
                "run_id": run_id,
            }
        )
    return pd.DataFrame(rows)


def format_table(table: pd.DataFrame) -> str:
    """Comparison table sorted by cap; ``*`` marks the lowest mean RMSE (the selection rule)."""
    best = table["rmse_mean"].idxmin()
    lines = [f"{'cap':>5}  {'CV RMSE (mean ± std)':>21}  {'CV MAE (mean ± std)':>20}"]
    lines.append("-" * len(lines[0]))
    for i, r in table.sort_values("cap").iterrows():
        rmse_text = f"{r['rmse_mean']:.2f} ± {r['rmse_std']:.2f}"
        mae_text = f"{r['mae_mean']:.2f} ± {r['mae_std']:.2f}"
        marker = " *" if i == best and len(table) > 1 else ""
        lines.append(f"{int(r['cap']):>5}  {rmse_text:>21}  {mae_text:>20}{marker}")
    if len(table) > 1:
        lines.append("* lowest mean CV RMSE")
    return "\n".join(lines)


def main() -> None:
    """Command-line entry point."""
    args = build_parser().parse_args()
    caps = args.caps or [args.cap]
    model_name, _ = MODELS[args.model]

    table = run_caps(
        args.model, args.features, args.subset, caps, args.windows, log=not args.no_mlflow
    )

    print(
        f"{args.subset} | {model_name} | features: {args.features}"
        + (f" (windows {args.windows})" if args.features == "window" else "")
        + f"\n{N_SPLITS}-fold engine CV (seed {SEED}); each cap applies to the training target "
        f"only; scored on uncapped true RUL <= {MAX_EVAL_RUL}; units: cycles\n"
    )
    print(format_table(table))
    if not args.no_mlflow:
        print(
            "\nMLflow runs: "
            + ", ".join(f"cap {c}: {r}" for c, r in zip(table["cap"], table["run_id"], strict=True))
        )


if __name__ == "__main__":
    main()
