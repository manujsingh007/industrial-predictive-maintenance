"""The packaged RUL predictor: one engine's raw history in, RUL with a 90% interval out.

Features are computed with the **same** functions as in training
(:func:`predmaint.features.windows.add_window_features` and :func:`window_feature_names`), so
there is no separate serving implementation that could drift from training ("training-serving
skew"). A test checks that serving and the training pipeline give identical predictions.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from predmaint.features.windows import add_window_features, window_feature_names
from predmaint.models.conformal import mondrian_bin_labels, mondrian_bins, mondrian_interval


class InvalidHistoryError(ValueError):
    """The engine history passed to :meth:`RULPredictor.predict_engine` is not usable."""


@dataclass(frozen=True)
class RULPrediction:
    """Prediction at an engine's last observed cycle.

    Attributes:
        unit: Engine id from the input.
        rul: Predicted remaining useful life in cycles, ``max(rul_raw, 0)``: a negative
            remaining life is meaningless to a planner.
        rul_raw: The model's raw output; identical to ``rul`` unless the model predicts < 0
            (possible very near failure, because boosted trees can overshoot).
        lower: Lower bound of the 90% interval, ``max(rul_raw - q, 0)``.
        upper: Upper bound of the 90% interval, ``rul_raw + q`` (raised to ``rul`` if lower),
            so that ``0 <= lower <= rul <= upper`` always holds.
        n_cycles_seen: Number of readings in the history (history length).
        last_cycle: Cycle number of the last reading.
        pred_bin: Predicted-RUL bin whose q set the interval width, e.g. "30-60".
    """

    unit: int
    rul: float
    rul_raw: float
    lower: float
    upper: float
    n_cycles_seen: int
    last_cycle: int
    pred_bin: str

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass(frozen=True)
class RULPredictor:
    """Trained LightGBM plus everything needed to reproduce its inputs and intervals.

    Attributes:
        model: Fitted regressor on window features (``predict(DataFrame)``).
        sensors: Kept sensor columns (ADR 0001).
        windows: Window lengths in cycles (ADR 0002).
        cap: RUL cap used for the training target (for reference; predictions are not clipped).
        bin_edges: Mondrian bin edges on the predicted RUL (ADR 0003).
        q_per_bin: Interval half-width per bin, calibrated on out-of-fold residuals.
        alpha: Miscoverage rate the intervals were calibrated for (0.1 = 90%).
        subset: Training subset, e.g. "FD001".
    """

    model: Any
    sensors: tuple[str, ...]
    windows: tuple[int, ...]
    cap: int
    bin_edges: tuple[float, ...]
    q_per_bin: tuple[float, ...]
    alpha: float
    subset: str

    @property
    def feature_names(self) -> list[str]:
        """Model input columns, in the same order as in training (sensors, then windows)."""
        return [*self.sensors, *window_feature_names(self.sensors, self.windows)]

    def validate_history(self, history: pd.DataFrame) -> None:
        """Raise :class:`InvalidHistoryError` with a clear message if ``history`` is unusable."""
        if not isinstance(history, pd.DataFrame):
            raise InvalidHistoryError("history must be a pandas DataFrame")
        if history.empty:
            raise InvalidHistoryError("history is empty; at least one cycle is required")
        missing = [c for c in ("unit", "cycle", *self.sensors) if c not in history.columns]
        if missing:
            raise InvalidHistoryError(f"history is missing required columns: {missing}")
        units = history["unit"].unique()
        if len(units) != 1:
            raise InvalidHistoryError(
                f"history must contain exactly one engine (unit); found {len(units)}: "
                f"{sorted(units.tolist())[:5]}"
            )
        cycles = pd.to_numeric(history["cycle"], errors="coerce")
        if cycles.isna().any() or (cycles != np.floor(cycles)).any() or (cycles < 1).any():
            raise InvalidHistoryError("cycle must contain whole numbers >= 1")
        if cycles.duplicated().any():
            dup = sorted(cycles[cycles.duplicated()].astype(int).unique().tolist())[:5]
            raise InvalidHistoryError(f"cycle values must be unique; duplicated: {dup}")
        if not cycles.is_monotonic_increasing:
            raise InvalidHistoryError("rows must be sorted by cycle (oldest first)")
        values = history[list(self.sensors)].apply(pd.to_numeric, errors="coerce")
        bad = [c for c in self.sensors if not np.isfinite(values[c].to_numpy(dtype=float)).all()]
        if bad:
            raise InvalidHistoryError(f"sensor values must be finite numbers; problems in: {bad}")

    def predict_engine(self, history: pd.DataFrame) -> RULPrediction:
        """Predict RUL at the last cycle of one engine's raw readings.

        Args:
            history: One engine's readings, oldest first: ``unit``, ``cycle`` and the kept
                sensors; other columns (settings, dropped sensors) are ignored. Any length >= 1:
                shorter histories than the windows use shrinking windows, exactly as in training.

        Raises:
            InvalidHistoryError: On missing columns, several units, unsorted or duplicate cycles,
                or non-finite sensor values.
        """
        self.validate_history(history)
        data = history[["unit", "cycle", *self.sensors]].reset_index(drop=True)
        data[list(self.sensors)] = data[list(self.sensors)].astype(float)
        features = add_window_features(data, self.sensors, self.windows)
        last = features.iloc[[-1]][self.feature_names]
        raw = float(np.asarray(self.model.predict(last))[0])
        # Interval from the raw prediction, exactly as in the final evaluation (ADR 0003).
        lower, upper = mondrian_interval([raw], self.q_per_bin, self.bin_edges)
        rul = max(raw, 0.0)
        labels = mondrian_bin_labels(self.bin_edges)
        return RULPrediction(
            unit=int(data["unit"].iloc[0]),
            rul=rul,
            rul_raw=raw,
            lower=float(lower[0]),
            upper=max(float(upper[0]), rul),
            n_cycles_seen=len(data),
            last_cycle=int(data["cycle"].iloc[-1]),
            pred_bin=labels[int(mondrian_bins([raw], self.bin_edges)[0])],
        )

    def predict_engines(self, histories: pd.DataFrame) -> pd.DataFrame:
        """Predict the last cycle of every engine in ``histories`` (grouped by ``unit``)."""
        if "unit" not in histories.columns:
            raise InvalidHistoryError("history is missing required columns: ['unit']")
        rows = [
            self.predict_engine(group.sort_values("cycle")).as_dict()
            for _, group in histories.groupby("unit", sort=True)
        ]
        return pd.DataFrame(rows)


def make_predictor(
    model: Any,
    *,
    sensors: Sequence[str],
    windows: Sequence[int],
    cap: int,
    bin_edges: Sequence[float],
    q_per_bin: Sequence[float],
    alpha: float,
    subset: str,
) -> RULPredictor:
    """Build a :class:`RULPredictor`, checking that the q values match the bins."""
    if len(q_per_bin) != len(bin_edges) - 1:
        raise ValueError(f"need {len(bin_edges) - 1} q values for {len(bin_edges)} bin edges")
    return RULPredictor(
        model=model,
        sensors=tuple(sensors),
        windows=tuple(int(w) for w in windows),
        cap=int(cap),
        bin_edges=tuple(float(e) for e in bin_edges),
        q_per_bin=tuple(float(q) for q in q_per_bin),
        alpha=float(alpha),
        subset=subset,
    )
