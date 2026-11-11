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


def main() -> None:
    """Command-line entry point."""
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
    parser.add_argument(
        "--cap", type=int, default=DEFAULT_RUL_CAP, help="training RUL cap (cycles)"
    )
    parser.add_argument("--no-mlflow", action="store_true", help="do not log the run to MLflow")
    args = parser.parse_args()

    model_name, factory = MODELS[args.model]
    data, features = build_features(add_rul(load_train(args.subset)), args.features, args.windows)
    result = cross_validate(
        factory, data, cap=args.cap, features=features, n_splits=N_SPLITS, seed=SEED
    )
    s = result.summary()

    print(
        f"{args.subset} | {model_name} | features: {args.features} ({len(features)} columns)"
        + (f", windows {args.windows}" if args.features == "window" else "")
        + f"\n{N_SPLITS}-fold engine CV (seed {SEED}), target capped at {args.cap}, "
        f"scored on uncapped true RUL <= {MAX_EVAL_RUL}\n"
        f"RMSE {s.loc['mean', 'rmse']:.2f} ± {s.loc['std', 'rmse']:.2f}   "
        f"MAE {s.loc['mean', 'mae']:.2f} ± {s.loc['std', 'mae']:.2f}   (cycles, mean ± std)"
    )

    if not args.no_mlflow:
        from predmaint.tracking import log_cv_result  # slow import; skipped for quick runs

        run_id = log_cv_result(
            result,
            model_name=model_name,
            subset=args.subset,
            run_name=f"{model_name} | {args.features}",
            extra_params={
                "feature_set": args.features,
                "windows": ",".join(map(str, args.windows))
                if args.features == "window"
                else "none",
                **model_hyperparams(args.model, result),
            },
        )
        print(f"MLflow run id: {run_id}")


if __name__ == "__main__":
    main()
