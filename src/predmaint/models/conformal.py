"""Split-conformal prediction intervals for any point-prediction model.

Given calibration residuals ``y_true - y_pred`` that the model did **not** train on, the interval
``[pred - q, pred + q]`` with ``q = conformal_quantile(residuals, alpha)`` covers a new true value
with probability at least ``1 - alpha``, provided the new point is *exchangeable* with the
calibration points (interchangeable in distribution). The guarantee is **marginal**: it holds on
average over all points, not separately for each RUL range.
"""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def conformal_quantile(residuals: ArrayLike, alpha: float = 0.1) -> float:
    """Finite-sample-corrected (1 - alpha) quantile of the absolute residuals.

    Finite-sample correction, in one line: with n calibration scores we take the
    ceil((n + 1)(1 - alpha))-th smallest instead of the plain (1 - alpha) quantile, because the
    new point is the (n + 1)-th exchangeable score and must also be counted. This makes coverage
    at least 1 - alpha even for small n.

    Args:
        residuals: Calibration residuals ``y_true - y_pred`` (sign is ignored).
        alpha: Miscoverage rate in (0, 1); 0.1 gives 90% intervals.

    Returns:
        The half-width ``q``. Returns ``inf`` when n is too small for the requested coverage,
        i.e. when ceil((n + 1)(1 - alpha)) > n.

    Raises:
        ValueError: If ``alpha`` is not in (0, 1) or ``residuals`` is empty or not finite.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    scores = np.abs(np.asarray(residuals, dtype=float).ravel())
    if scores.size == 0:
        raise ValueError("residuals must not be empty")
    if not np.isfinite(scores).all():
        raise ValueError("residuals must be finite")
    n = scores.size
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return math.inf
    return float(np.partition(scores, k - 1)[k - 1])


def interval(pred: ArrayLike, q: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Symmetric interval ``[pred - q, pred + q]``, with the lower bound clipped at 0 (RUL >= 0)."""
    p = np.asarray(pred, dtype=float)
    return np.maximum(p - q, 0.0), p + q
