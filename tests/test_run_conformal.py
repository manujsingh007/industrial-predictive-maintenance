"""run_conformal end to end on synthetic engines, logging to a temporary MLflow database."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from predmaint.features.selection import KEPT_SENSORS
from predmaint.models.gbdt import make_lgbm, make_lgbm_quantile
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
    monkeypatch.setattr(
        run_conformal, "QUANTILE_FACTORY", lambda q: make_lgbm_quantile(q, n_estimators=20)
    )


def test_split_only_without_logging(synthetic: None) -> None:
    result = run_conformal.run("FD001", alpha=0.1, method="split", log=False)
    assert result.run_ids == {}
    assert set(result.reports) == {"split"}
    assert "lo" not in result.point.oof.columns  # quantile models not trained for split only
    report = result.reports["split"]
    assert len(report.per_fold) == 5
    assert 0 <= report.overall["coverage"] <= 1
    text = run_conformal.format_report(result)
    assert "Target coverage: 90%" in text and "split conformal coverage" in text


def test_both_methods_side_by_side(synthetic: None) -> None:
    result = run_conformal.run("FD001", alpha=0.1, method="both", log=False)
    assert set(result.reports) == {"split", "cqr"}
    oof = result.point.oof
    assert {"lo", "hi"} <= set(oof.columns) and oof[["lo", "hi"]].notna().all().all()
    # both methods are evaluated on exactly the same rows
    for m in ("split", "cqr"):
        assert result.reports[m].overall["n"] == result.reports["split"].overall["n"]
    text = run_conformal.format_by_bucket(result.reports)
    for column in (
        "split conformal coverage",
        "split conformal width",
        "CQR coverage",
        "CQR width",
    ):
        assert column in text
    assert "all" in text


def test_each_method_logged_as_its_own_run(synthetic: None, tmp_path: Path) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    result = run_conformal.run("FD001", alpha=0.2, method="both", tracking_uri=uri)
    client = MlflowClient(tracking_uri=uri)
    assert set(result.run_ids) == {"split", "cqr"}
    for m, run_id in result.run_ids.items():
        run = client.get_run(run_id)
        assert run.data.params["conformal_method"] == m
        assert run.data.params["conformal_alpha"] == "0.2"
        expected = result.reports[m].overall["coverage"]
        assert run.data.metrics["conformal_coverage"] == pytest.approx(expected)
        steps = sorted(m_.step for m_ in client.get_metric_history(run_id, "conformal_q_fold"))
        assert steps == [0, 1, 2, 3, 4]
    cqr_params = client.get_run(result.run_ids["cqr"]).data.params
    assert cqr_params["cqr_quantiles"] == "0.05,0.95"


def test_unknown_method_raises(synthetic: None) -> None:
    with pytest.raises(ValueError, match="method"):
        run_conformal.run("FD001", method="bogus", log=False)
