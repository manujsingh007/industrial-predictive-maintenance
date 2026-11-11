"""Baseline models every RUL model must beat (docs/evaluation.md, section 5).

Each factory returns a new, unfitted scikit-learn estimator, so every CV fold starts fresh.
"""

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_mean_baseline() -> DummyRegressor:
    """Always predict the mean of the (capped) training target, ignoring the features."""
    return DummyRegressor(strategy="mean")


def make_linear_baseline() -> Pipeline:
    """Ordinary least squares on standardised current-cycle sensor values.

    The scaler is a pipeline step, so it is fitted only on the rows passed to ``fit``
    (the training fold) and merely applied to the rows passed to ``predict``.
    """
    return Pipeline([("scale", StandardScaler()), ("regress", LinearRegression())])
