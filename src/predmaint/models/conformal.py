"""Conformal prediction intervals: split conformal and conformalized quantile regression (CQR).

**Split conformal.** Given calibration residuals ``y_true - y_pred`` that the model did **not**
train on, the interval ``[pred - q, pred + q]`` with ``q = conformal_quantile(residuals, alpha)``
covers a new true value with probability at least ``1 - alpha``, provided the new point is
*exchangeable* with the calibration points (interchangeable in distribution). Every interval
has the same width.

**CQR** (Romano et al., 2019). Two quantile models predict a lower (5%) and an upper (95%)
quantile, ``lo`` and ``hi``. Their gap has already learned where the target is noisy: wide where
errors are large, narrow where they are small. Calibration then computes one correction ``q``
from the scores ``max(lo - y, y - hi)``, and the interval is ``[lo - q, hi + q]``. The width
therefore **adapts** to each input, while the same finite-sample guarantee holds. ``q`` can be
negative, which shrinks a band that was too wide.

Both guarantees are **marginal**: they hold on average over all points, not separately for
each RUL range.
"""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

CQR_QUANTILES = (0.05, 0.95)  # lower and upper quantile models for 90% intervals


def conformal_score_quantile(scores: ArrayLike, alpha: float = 0.1) -> float:
    """Finite-sample-corrected (1 - alpha) quantile of conformity scores (sign kept).

    Finite-sample correction, in one line: with n calibration scores we take the
    ceil((n + 1)(1 - alpha))-th smallest instead of the plain (1 - alpha) quantile, because the
    new point is the (n + 1)-th exchangeable score and must also be counted. This makes coverage
    at least 1 - alpha even for small n.

    Args:
        scores: Calibration conformity scores; larger means "worse fit".
        alpha: Miscoverage rate in (0, 1); 0.1 gives 90% intervals.

    Returns:
        The correction ``q``; ``inf`` when ceil((n + 1)(1 - alpha)) > n (too few points).

    Raises:
        ValueError: If ``alpha`` is not in (0, 1) or ``scores`` is empty or not finite.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    s = np.asarray(scores, dtype=float).ravel()
    if s.size == 0:
        raise ValueError("scores must not be empty")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    n = s.size
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return math.inf
    return float(np.partition(s, k - 1)[k - 1])


def conformal_quantile(residuals: ArrayLike, alpha: float = 0.1) -> float:
    """Split-conformal half-width: :func:`conformal_score_quantile` of ``|residuals|``.

    Args:
        residuals: Calibration residuals ``y_true - y_pred`` (sign is ignored).
        alpha: Miscoverage rate in (0, 1); 0.1 gives 90% intervals.
    """
    return conformal_score_quantile(np.abs(np.asarray(residuals, dtype=float)), alpha)


def _ordered(lower_pred: ArrayLike, upper_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Swap crossed quantile predictions (two separate models can give lo > hi)."""
    lo = np.asarray(lower_pred, dtype=float)
    hi = np.asarray(upper_pred, dtype=float)
    return np.minimum(lo, hi), np.maximum(lo, hi)


def cqr_scores(y_true: ArrayLike, lower_pred: ArrayLike, upper_pred: ArrayLike) -> np.ndarray:
    """CQR conformity scores ``max(lo - y, y - hi)``: negative inside the band, positive outside."""
    lo, hi = _ordered(lower_pred, upper_pred)
    y = np.asarray(y_true, dtype=float)
    return np.maximum(lo - y, y - hi)


def cqr_interval(
    lower_pred: ArrayLike, upper_pred: ArrayLike, q: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """CQR interval ``[lo - q, hi + q]``, lower bound clipped at 0 and never above the upper.

    A negative ``q`` can push ``hi + q`` below 0 for engines very close to failure. The upper
    bound is then raised to the clipped lower bound (an empty-width interval at 0) rather than
    returning lower > upper.
    """
    lo, hi = _ordered(lower_pred, upper_pred)
    lower = np.maximum(lo - q, 0.0)
    return lower, np.maximum(hi + q, lower)


def interval(pred: ArrayLike, q: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Symmetric interval ``[pred - q, pred + q]``, with the lower bound clipped at 0 (RUL >= 0)."""
    p = np.asarray(pred, dtype=float)
    return np.maximum(p - q, 0.0), p + q


# --- Mondrian (group-conditional) split conformal --------------------------------------------

MONDRIAN_EDGES = (0.0, 30.0, 60.0, 90.0, math.inf)
MIN_BIN_CALIBRATION = 50
"""Fewest calibration rows for a bin to get its own q; smaller bins use the global q.

With alpha = 0.1 a bin needs at least 9 rows for any finite q, but then q is essentially its
largest residual. 50 rows is a pragmatic floor below which a per-bin quantile is too noisy
to beat the global one.
"""


def mondrian_bins(pred: ArrayLike, edges: tuple[float, ...] = MONDRIAN_EDGES) -> np.ndarray:
    """Bin index of each **predicted** RUL; bins are left-closed, e.g. [0, 30), [30, 60).

    Bins must use the prediction, not the true RUL: a new engine's true RUL is unknown when its
    interval is needed, and the per-bin guarantee requires that bin membership does not depend
    on the label. Predictions below the first edge go into the first bin.
    """
    if len(edges) < 2 or list(edges) != sorted(edges):
        raise ValueError(f"edges must be increasing with at least two values, got {edges}")
    return np.searchsorted(np.asarray(edges[1:-1]), np.asarray(pred, dtype=float), side="right")


def mondrian_bin_labels(edges: tuple[float, ...] = MONDRIAN_EDGES) -> list[str]:
    """Readable labels such as "0-30" and "90+"."""
    return [
        f"{lo:g}+" if math.isinf(hi) else f"{lo:g}-{hi:g}"
        for lo, hi in zip(edges[:-1], edges[1:], strict=True)
    ]


def mondrian_quantiles(
    residuals: ArrayLike,
    pred: ArrayLike,
    alpha: float = 0.1,
    edges: tuple[float, ...] = MONDRIAN_EDGES,
    min_bin_size: int = MIN_BIN_CALIBRATION,
) -> tuple[np.ndarray, np.ndarray]:
    """One split-conformal q per predicted-RUL bin, from absolute calibration residuals.

    Args:
        residuals: Calibration residuals ``y_true - y_pred`` (sign is ignored).
        pred: Calibration predictions, which decide each residual's bin.
        alpha: Miscoverage rate in (0, 1).
        edges: Bin edges on the predicted RUL.
        min_bin_size: Bins with fewer calibration rows fall back to the global q.

    Returns:
        ``(q_per_bin, used_fallback)``, both of length ``len(edges) - 1``.
    """
    scores = np.abs(np.asarray(residuals, dtype=float))
    bins = mondrian_bins(pred, edges)
    if len(scores) != len(bins):
        raise ValueError("residuals and pred must have the same length")
    global_q = conformal_score_quantile(scores, alpha)
    n_bins = len(edges) - 1
    q = np.full(n_bins, global_q)
    fallback = np.ones(n_bins, dtype=bool)
    for b in range(n_bins):
        in_bin = scores[bins == b]
        if len(in_bin) >= min_bin_size:
            q[b] = conformal_score_quantile(in_bin, alpha)
            fallback[b] = False
    return q, fallback


def mondrian_interval(
    pred: ArrayLike, q_per_bin: ArrayLike, edges: tuple[float, ...] = MONDRIAN_EDGES
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Interval ``pred ± q[bin(pred)]``, lower bound clipped at 0."""
    q_row = np.asarray(q_per_bin, dtype=float)[mondrian_bins(pred, edges)]
    p = np.asarray(pred, dtype=float)
    return np.maximum(p - q_row, 0.0), p + q_row
