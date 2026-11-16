"""A small 1D-CNN for RUL, wrapped as a scikit-learn regressor (needs the ``deep`` group).

It plugs into :func:`predmaint.training.cv.cross_validate` like any other model. Each row of
``X`` must carry ``unit`` and ``cycle``, which are used **only** to build each row's input
sequence (see :mod:`predmaint.features.sequences`), plus the sensor columns. Each row's input
is its engine's last ``window`` cycles x ``len(sensors)`` sensors, and the output is that
row's RUL.

Leakage safeguards inside :meth:`CNNRegressor.fit`:

- The StandardScaler is fitted on the rows passed to ``fit`` (the CV training fold) only.
- Early stopping holds out about 15% of the **training** engines (split by engine). The CV
  validation fold is never seen during training.

Reproducibility: weights, dropout, batch order and the early-stopping split all come from
``seed``. Results are bit-for-bit repeatable on CPU. On Apple MPS (the default when available)
they can differ slightly between runs, because some GPU kernels are not deterministic.
"""

import copy
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from torch import nn

from predmaint.features.selection import KEPT_SENSORS
from predmaint.features.sequences import build_sequences


class _ConvNet(nn.Module):
    """Conv1d blocks over time, then a dense head that outputs one value per sequence."""

    def __init__(
        self,
        n_sensors: int,
        window: int,
        channels: Sequence[int],
        kernel_size: int,
        dense: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = n_sensors
        for out_channels in channels:
            layers += [nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"), nn.ReLU()]
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * window, dense),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, sensors, window)
        return self.head(self.conv(x)).squeeze(-1)


def _pick_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(device)


class CNNRegressor(RegressorMixin, BaseEstimator):
    """1D-CNN RUL regressor over causal per-engine sequences.

    Args:
        window: Sequence length in cycles; shorter histories are left-padded.
        sensors: Sensor columns used as input channels.
        channels: Output channels of each conv layer (2-3 layers).
        kernel_size: Conv kernel size in cycles.
        dense: Width of the hidden dense layer.
        dropout: Dropout rate before the output layer.
        learning_rate: Adam learning rate.
        batch_size: Mini-batch size.
        max_epochs: Upper limit on training epochs.
        patience: Stop after this many epochs without improvement on the early-stopping engines.
        early_stopping_fraction: Share of training engines held out for early stopping.
        seed: Seed for weights, dropout, batch order and the early-stopping split.
        device: "auto" (MPS if available, else CPU), "cpu" or "mps".
    """

    def __init__(
        self,
        window: int = 30,
        sensors: Sequence[str] = tuple(KEPT_SENSORS),
        channels: Sequence[int] = (32, 64, 64),
        kernel_size: int = 5,
        dense: int = 64,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 40,
        patience: int = 5,
        early_stopping_fraction: float = 0.15,
        seed: int = 42,
        device: str = "auto",
    ) -> None:
        # scikit-learn convention: store arguments unchanged; all work happens in fit().
        self.window = window
        self.sensors = sensors
        self.channels = channels
        self.kernel_size = kernel_size
        self.dense = dense
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.early_stopping_fraction = early_stopping_fraction
        self.seed = seed
        self.device = device

    # --- helpers -------------------------------------------------------------------------

    def _check_columns(self, X: pd.DataFrame) -> None:
        missing = {"unit", "cycle", *self.sensors} - set(X.columns)
        if missing:
            raise ValueError(f"X is missing columns: {sorted(missing)}")

    def _sequences(self, X: pd.DataFrame) -> torch.Tensor:
        """Scaled, causal sequences shaped (rows, sensors, window) for Conv1d."""
        scaled = self.scaler_.transform(X[list(self.sensors)].to_numpy())
        seq = build_sequences(X["unit"].to_numpy(), X["cycle"].to_numpy(), scaled, self.window)
        return torch.from_numpy(seq).permute(0, 2, 1).contiguous()

    def _predict_scaled(self, seq: torch.Tensor) -> torch.Tensor:
        self.model_.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(seq), 4096):
                outputs.append(self.model_(seq[start : start + 4096].to(self.device_)).cpu())
        return torch.cat(outputs) if outputs else torch.empty(0)

    # --- scikit-learn API ----------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> "CNNRegressor":
        """Train on the rows of ``X`` (one CV training fold) with targets ``y``."""
        self._check_columns(X)
        y = np.asarray(y, dtype=np.float32)
        if len(y) != len(X):
            raise ValueError("X and y have different lengths")
        units = X["unit"].to_numpy()
        engines = np.unique(units)
        if len(engines) < 2:
            raise ValueError("need at least 2 engines (one is held out for early stopping)")

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.device_ = _pick_device(self.device)

        # Early-stopping engines: an inner split by engine of the TRAINING rows only.
        n_es = min(len(engines) - 1, max(1, round(self.early_stopping_fraction * len(engines))))
        self.early_stopping_units_ = np.sort(rng.choice(engines, size=n_es, replace=False))
        is_es = np.isin(units, self.early_stopping_units_)

        # Scaler fitted on the rows given to fit() only; never on validation rows.
        self.scaler_ = StandardScaler().fit(X[list(self.sensors)].to_numpy())
        # Target scaling keeps the loss near 1 for stable Adam steps; undone in predict().
        self.y_scale_ = float(max(np.abs(y).max(), 1.0))

        seq = self._sequences(X)  # built per engine, so held-out engines never mix in
        target = torch.from_numpy(y / self.y_scale_)
        train_seq, train_y = seq[~is_es], target[~is_es]
        es_seq, es_y = seq[is_es], target[is_es]

        self.model_ = _ConvNet(
            len(self.sensors),
            self.window,
            self.channels,
            self.kernel_size,
            self.dense,
            self.dropout,
        ).to(self.device_)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        batch_order = torch.Generator().manual_seed(self.seed)

        best_loss, best_state, epochs_without_gain = float("inf"), None, 0
        self.history_: list[dict[str, float]] = []
        for epoch in range(self.max_epochs):
            self.model_.train()
            for idx in torch.randperm(len(train_seq), generator=batch_order).split(self.batch_size):
                optimizer.zero_grad()
                pred = self.model_(train_seq[idx].to(self.device_))
                loss = loss_fn(pred, train_y[idx].to(self.device_))
                loss.backward()
                optimizer.step()

            es_loss = float(loss_fn(self._predict_scaled(es_seq), es_y))
            self.history_.append({"epoch": epoch, "early_stopping_mse": es_loss})
            if es_loss < best_loss:
                best_loss, epochs_without_gain = es_loss, 0
                best_state = copy.deepcopy(self.model_.state_dict())
                self.best_epoch_ = epoch
            else:
                epochs_without_gain += 1
                if epochs_without_gain >= self.patience:
                    break

        if best_state is None:  # early-stopping loss was NaN from the first epoch
            raise RuntimeError("training diverged: early-stopping loss was never finite")
        self.model_.load_state_dict(best_state)
        self.n_epochs_ = len(self.history_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict RUL for every row of ``X``, in row order.

        Each row's sequence is built from the rows of ``X`` of the same engine, so pass all
        available earlier cycles of an engine (``cross_validate`` passes whole engines).
        """
        check_is_fitted(self, "model_")
        self._check_columns(X)
        pred = self._predict_scaled(self._sequences(X)).numpy().astype(np.float64)
        return pred * self.y_scale_
