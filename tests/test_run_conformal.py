"""run_conformal end to end on synthetic engines, logging to a temporary MLflow database."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from predmaint.features.selection import KEPT_SENSORS
from predmaint.models.gbdt import make_lgbm
from predmaint.tracking import TRACKING_DB
from predmaint.training import run_conformal


@pytest.fixture(autouse=True)
def real_db_untouched() -> Iterator[None]:
    def snapshot() -> tuple[int, int] | None:
        if not TRACKING_DB.exists():
            return None
        return TRACKING_DB.stat().st_size, TRACKING_DB.stat().st_mtime_ns

    before = snapshot()
    yield
    assert snapshot() == before


@pytest.fixture
def synthetic(monkeypatch: pytest.MonkeyPatch) -> None:
    """10 synthetic engines and a small LightGBM, so the test runs in seconds."""
    rng = np.random.default_rng(0)
    frames = []
    for unit in range(1, 11):
        n = 130 + 7 * unit
        frame = pd.DataFrame(rng.normal(size=(n, len(KEPT_SENSORS))), columns=KEPT_SENSORS)
        frame[KEPT_SENSORS[0]] += np.linspace(0, 6, n)
        frame.insert(0, "unit", unit)
        frame.insert(1, "cycle", np.arange(1, n + 1))
        frames.append(frame)
    fake = pd.concat(frames, ignore_index=True)
    monkeypatch.setattr(run_conformal, "load_train", lambda subset: fake.copy())
    monkeypatch.setattr(run_conformal, "MODEL_FACTORY", lambda: make_lgbm(n_estimators=20))


def test_run_without_logging(synthetic: None) -> None:
    result, report, run_id = run_conformal.run("FD001", alpha=0.1, log=False)
    assert run_id is None
    assert len(report.per_fold) == 5
    assert 0 <= report.overall["coverage"] <= 1
    assert (result.oof["scored"] == (result.oof["rul"] <= 150)).all()
    text = run_conformal.format_report(report)
    assert "Target coverage: 90%" in text and "Overall: coverage" in text


def test_run_logs_coverage_metrics(synthetic: None, tmp_path: Path) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    _, report, run_id = run_conformal.run("FD001", alpha=0.2, tracking_uri=uri)
    client = MlflowClient(tracking_uri=uri)
    run = client.get_run(run_id)
    assert run.data.params["conformal_alpha"] == "0.2"
    assert run.data.params["feature_set"] == "window"
    assert run.data.metrics["conformal_coverage"] == pytest.approx(report.overall["coverage"])
    assert "conformal_coverage_rul_0_25" in run.data.metrics
    history = client.get_metric_history(run_id, "conformal_coverage_fold")
    assert sorted(m.step for m in history) == [0, 1, 2, 3, 4]
