"""Cross-validation by engine, following docs/evaluation.md."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd
from sklearn.base import RegressorMixin

from predmaint.evaluation.metrics import mae, rmse
from predmaint.evaluation.splits import engine_folds
from predmaint.features.selection import DEFAULT_RUL_CAP, KEPT_SENSORS, cap_rul

MAX_EVAL_RUL = 150  # validation rows scored: true RUL <= 150 (docs/evaluation.md, section 1)


@dataclass
class CVResult:
    """Cross-validation results.

    Attributes:
        folds: One row per fold: ``fold``, ``n_val_engines``, ``n_val_rows_scored``,
            ``rmse``, ``mae``.
        models: The fitted model of each fold, in fold order.
    """

    folds: pd.DataFrame
    models: list[RegressorMixin]

    def summary(self) -> pd.DataFrame:
        """Mean and (sample) standard deviation of RMSE and MAE across folds."""
        return self.folds[["rmse", "mae"]].agg(["mean", "std"])


def cross_validate(
    model_factory: Callable[[], RegressorMixin],
    train_df: pd.DataFrame,
    cap: float = DEFAULT_RUL_CAP,
    *,
    features: Sequence[str] = KEPT_SENSORS,
    n_splits: int = 5,
    seed: int = 42,
    max_eval_rul: float = MAX_EVAL_RUL,
) -> CVResult:
    """Cross-validate a model with engine folds, capped training and uncapped scoring.

    For each fold:

    1. Build a fresh model with ``model_factory()``.
    2. Fit it on the training engines with target ``min(rul, cap)``.
    3. Predict all rows of the validation engines.
    4. Score only validation rows with true RUL <= ``max_eval_rul``, against the
       **uncapped** true RUL.

    Args:
        model_factory: Zero-argument function returning a new, unfitted estimator.
        train_df: Training rows with ``unit``, ``rul`` (from ``add_rul``) and the feature columns.
        cap: RUL cap applied to the training target only.
        features: Feature columns passed to the model.
        n_splits: Number of engine folds.
        seed: Fold shuffling seed.
        max_eval_rul: Largest true RUL that is scored.

    Returns:
        Per-fold scores and fitted models; see :class:`CVResult`.
    """
    missing = {"unit", "rul", *features} - set(train_df.columns)
    if missing:
        raise ValueError(f"train_df is missing columns: {sorted(missing)}")

    X = train_df[list(features)]
    y_true = train_df["rul"]
    y_train_target = cap_rul(y_true, cap)

    rows, models = [], []
    for fold, (train_idx, val_idx) in enumerate(engine_folds(train_df, n_splits, seed)):
        model = model_factory()
        model.fit(X.iloc[train_idx], y_train_target.iloc[train_idx])

        y_val = y_true.iloc[val_idx]
        scored = (y_val <= max_eval_rul).to_numpy()
        y_pred = model.predict(X.iloc[val_idx][scored])

        rows.append(
            {
                "fold": fold,
                "n_val_engines": train_df["unit"].iloc[val_idx].nunique(),
                "n_val_rows_scored": int(scored.sum()),
                "rmse": rmse(y_val[scored], y_pred),
                "mae": mae(y_val[scored], y_pred),
            }
        )
        models.append(model)

    return CVResult(folds=pd.DataFrame(rows), models=models)
