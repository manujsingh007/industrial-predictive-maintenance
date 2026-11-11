"""RUL metrics and bootstrap confidence intervals, as defined in docs/evaluation.md.

Conventions:

- ``d = y_pred - y_true``, so d > 0 is a *late* prediction (the engine fails sooner than predicted).
- ``y_true`` is the **uncapped** true RUL.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

Metric = Callable[[ArrayLike, ArrayLike], float]
FloatArray = NDArray[np.float64]

PHM08_LATE_SCALE = 10.0  # d >= 0: exp(d / 10) - 1
PHM08_EARLY_SCALE = 13.0  # d < 0:  exp(-d / 13) - 1


def _as_pair(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Convert inputs to equal-length, non-empty 1-D float arrays."""
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if t.ndim != 1 or p.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-D")
    if t.shape != p.shape:
        raise ValueError(f"length mismatch: y_true has {t.size}, y_pred has {p.size}")
    if t.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    return t, p


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error, in cycles."""
    t, p = _as_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error, in cycles."""
    t, p = _as_pair(y_true, y_pred)
    return float(np.mean(np.abs(p - t)))


def phm08_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Asymmetric PHM08 score (lower is better; 0 is perfect).

    With ``d = y_pred - y_true``, each engine contributes ``exp(d / 10) - 1`` when late (d >= 0)
    and ``exp(-d / 13) - 1`` when early (d < 0), so late predictions cost more. Returns the
    **sum** over engines, which grows with the number of engines: only compare scores computed
    on the same set of engines.
    """
    t, p = _as_pair(y_true, y_pred)
    d = p - t
    penalties = np.where(d >= 0, np.exp(d / PHM08_LATE_SCALE), np.exp(-d / PHM08_EARLY_SCALE)) - 1
    return float(np.sum(penalties))


def bootstrap_ci(
    metric: Metric,
    y_true: ArrayLike,
    y_pred: ArrayLike,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Point estimate and 95% percentile bootstrap interval of ``metric``.

    Each element of ``y_true``/``y_pred`` must be **one engine** (on the test set: the
    prediction at each engine's last cycle), so resampling elements resamples engines. Each
    resample draws ``len(y_true)`` engines with replacement.

    Args:
        metric: Function ``metric(y_true, y_pred) -> float``, e.g. :func:`rmse`.
        y_true: True RUL, one value per engine.
        y_pred: Predicted RUL, one value per engine.
        n_resamples: Number of bootstrap resamples.
        seed: Seed for the random generator; the same seed gives the same interval.

    Returns:
        ``(point, low, high)``: the metric on the original data and the 2.5th / 97.5th
        percentiles of the resampled metrics.
    """
    t, p = _as_pair(y_true, y_pred)
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, t.size, size=(n_resamples, t.size))
    stats = np.array([metric(t[idx], p[idx]) for idx in indices])
    low, high = np.percentile(stats, [2.5, 97.5])
    return metric(t, p), float(low), float(high)
