"""Gradient-boosted decision trees (LightGBM).

The hyperparameters are fixed defaults, not tuned per fold. The model trains for exactly
``n_estimators`` rounds, with **no early stopping on the validation fold**. Stopping when the
validation score stops improving would choose the number of trees using validation labels,
which leaks them into the model and makes the CV score optimistic.

``lightgbm`` is imported inside :func:`make_lgbm`, so the rest of the package works on machines
where LightGBM's OpenMP runtime is missing (on macOS: ``brew install libomp``).
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lightgbm import LGBMRegressor

LGBM_DEFAULTS: dict[str, Any] = {
    "n_estimators": 500,  # number of boosting rounds (trees); fixed, no early stopping
    "learning_rate": 0.05,  # shrinks each tree's contribution; smaller = slower, steadier
    "num_leaves": 31,  # max leaves per tree; controls how complex each tree can be
    "min_child_samples": 50,  # min rows per leaf; stops leaves fitting a few noisy cycles
    "subsample": 0.8,  # each round trains on a random 80% of rows (bagging)
    "subsample_freq": 1,  # re-draw that row sample every round (0 would disable bagging)
    "colsample_bytree": 0.8,  # each tree sees a random 80% of features
    "random_state": 42,  # seed for row and feature sampling
    "deterministic": True,  # identical results on every run, given the same data and seed
    "force_row_wise": True,  # fix the histogram layout; required for determinism
    "verbose": -1,  # silence LightGBM's per-run log output
}


def make_lgbm(**overrides: Any) -> "LGBMRegressor":
    """Return a new, unfitted LGBMRegressor with :data:`LGBM_DEFAULTS`, updated by ``overrides``.

    Raises:
        TypeError: If an override is not one of the default hyperparameters (catches typos
            such as ``num_leaf``, which LightGBM would otherwise silently ignore).
    """
    unknown = set(overrides) - set(LGBM_DEFAULTS)
    if unknown:
        raise TypeError(f"unknown LightGBM overrides: {sorted(unknown)}")
    from lightgbm import LGBMRegressor

    return LGBMRegressor(**{**LGBM_DEFAULTS, **overrides})
