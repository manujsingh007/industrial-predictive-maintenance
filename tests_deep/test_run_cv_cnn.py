"""run_cv with the CNN: hyperparameters logged to a temporary MLflow database."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="needs the 'deep' dependency group: uv sync --group deep")

from mlflow.tracking import MlflowClient  # noqa: E402

from predmaint.features.selection import KEPT_SENSORS  # noqa: E402
from predmaint.models.cnn import CNNRegressor  # noqa: E402
from predmaint.tracking import TRACKING_DB  # noqa: E402
from predmaint.training import run_cv  # noqa: E402


@pytest.fixture(autouse=True)
def real_db_untouched() -> Iterator[None]:
    def snapshot() -> tuple[int, int] | None:
        return (
            (TRACKING_DB.stat().st_size, TRACKING_DB.stat().st_mtime_ns)
            if TRACKING_DB.exists()
            else None
        )

    before = snapshot()
    yield
    assert snapshot() == before


@pytest.fixture
def synthetic_train(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(0)
    frames = []
    for unit in range(1, 7):
        n = 160 + 5 * unit
        frame = pd.DataFrame(rng.normal(size=(n, len(KEPT_SENSORS))), columns=KEPT_SENSORS)
        frame.insert(0, "unit", unit)
        frame.insert(1, "cycle", np.arange(1, n + 1))
        frames.append(frame)
    fake = pd.concat(frames, ignore_index=True)
    monkeypatch.setattr(run_cv, "load_train", lambda subset: fake.copy())


def test_cnn_run_logs_cnn_hyperparams(
    synthetic_train: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def tiny() -> CNNRegressor:
        return CNNRegressor(window=5, channels=(4,), dense=4, max_epochs=2, device="cpu")

    monkeypatch.setitem(run_cv.MODELS, "cnn", ("1D-CNN", tiny))
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    table = run_cv.run_caps("cnn", "sequence", "FD001", [125], tracking_uri=uri)
    params = MlflowClient(tracking_uri=uri).get_run(table["run_id"].iloc[0]).data.params
    assert params["feature_set"] == "sequence"
    assert params["cnn_window"] == "5"
    assert params["cnn_max_epochs"] == "2"
    assert params["cnn_device_used"] == "cpu"
    assert "cnn_sensors" not in params


def test_cnn_run_never_loads_lightgbm(
    synthetic_train: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # PyTorch and LightGBM bundle different OpenMP runtimes, which crash in one process on
    # macOS. run_cv must import lightgbm only for --model lgbm.
    import sys

    def tiny() -> CNNRegressor:
        return CNNRegressor(window=5, channels=(4,), dense=4, max_epochs=1, device="cpu")

    monkeypatch.setitem(run_cv.MODELS, "cnn", ("1D-CNN", tiny))
    table = run_cv.run_caps("cnn", "sequence", "FD001", [125], log=False)
    assert len(table) == 1
    assert "lightgbm" not in sys.modules
