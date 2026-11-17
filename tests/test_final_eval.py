"""Tests for the one-time final evaluation: run-once guard and outputs (synthetic data only).

No real test data is read: every loader is replaced by synthetic engines, reports go to
tmp_path, and MLflow logs to a temporary database.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from predmaint.features.selection import KEPT_SENSORS
from predmaint.models.gbdt import make_lgbm
from predmaint.tracking import TRACKING_DB
from predmaint.training import final_eval
from predmaint.training.final_eval import (
    FinalEvalAlreadyDone,
    check_not_already_run,
    report_paths,
    run_final_eval,
    wilson_interval,
)


@pytest.fixture(autouse=True)
def real_outputs_untouched() -> Iterator[None]:
    """Fail any test that touches the real mlflow.db or the real reports/ folder."""

    def snapshot() -> tuple:
        db = (
            (TRACKING_DB.stat().st_size, TRACKING_DB.stat().st_mtime_ns)
            if TRACKING_DB.exists()
            else None
        )
        reports = (
            sorted(p.name for p in final_eval.REPORTS_DIR.glob("*"))
            if final_eval.REPORTS_DIR.exists()
            else None
        )
        return db, reports

    before = snapshot()
    yield
    assert snapshot() == before


def engine(unit: int, n: int, rng: np.random.Generator) -> pd.DataFrame:
    frame = pd.DataFrame(rng.normal(size=(n, len(KEPT_SENSORS))), columns=KEPT_SENSORS)
    frame[KEPT_SENSORS[0]] += np.linspace(0, 6, n)  # drifts towards failure
    frame.insert(0, "unit", unit)
    frame.insert(1, "cycle", np.arange(1, n + 1))
    return frame


@pytest.fixture
def synthetic(monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    """Synthetic train/test/RUL loaders, a small LightGBM and fewer bootstrap resamples."""
    rng = np.random.default_rng(0)
    train = pd.concat([engine(u, 140 + 6 * u, rng) for u in range(1, 13)], ignore_index=True)
    test_frames, rul = [], []
    for u in range(1, 21):
        lifetime = 150 + 3 * u
        observed = int(rng.integers(20, lifetime - 5))  # cut off before failure
        test_frames.append(engine(u, lifetime, rng).iloc[:observed])
        rul.append(lifetime - observed)
    test = pd.concat(test_frames, ignore_index=True)
    truth = pd.DataFrame({"unit": range(1, 21), "rul": rul})

    monkeypatch.setattr(final_eval, "load_train", lambda subset: train.copy())
    monkeypatch.setattr(final_eval, "load_test", lambda subset: test.copy())
    monkeypatch.setattr(final_eval, "load_test_rul", lambda subset: truth.copy())
    monkeypatch.setattr(final_eval, "MODEL_FACTORY", lambda: make_lgbm(n_estimators=20))
    monkeypatch.setattr(final_eval, "N_RESAMPLES", 200)
    return truth


def forbid_data_access(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(subset: str) -> None:
        raise AssertionError("data must not be read when the guard refuses")

    for name in ("load_train", "load_test", "load_test_rul"):
        monkeypatch.setattr(final_eval, name, refuse)


# --- run-once guard ----------------------------------------------------------------------


def test_guard_allows_first_run_and_refuses_second(tmp_path: Path) -> None:
    report, _ = report_paths("FD001", tmp_path)
    check_not_already_run(report, force=False)  # no report yet: allowed
    report.write_text("done")
    with pytest.raises(FinalEvalAlreadyDone, match="--force"):
        check_not_already_run(report, force=False)
    check_not_already_run(report, force=True)  # deliberate override


def test_refused_run_reads_no_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report, _ = report_paths("FD001", tmp_path)
    report.write_text("done")
    forbid_data_access(monkeypatch)
    with pytest.raises(FinalEvalAlreadyDone):
        run_final_eval("FD001", report_dir=tmp_path)


def test_main_exits_with_1_when_already_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(final_eval, "REPORTS_DIR", tmp_path)
    report_paths("FD001", tmp_path)[0].write_text("done")
    forbid_data_access(monkeypatch)
    assert final_eval.main(["--subset", "FD001"]) == 1
    assert "Refusing to run" in capsys.readouterr().err


def test_crash_leaves_no_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def crash(subset: str, *, forced: bool = False) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(final_eval, "evaluate", crash)
    with pytest.raises(RuntimeError):
        run_final_eval("FD001", report_dir=tmp_path)
    report, predictions = report_paths("FD001", tmp_path)
    assert not report.exists() and not predictions.exists()  # the one run is not used up


# --- end to end on synthetic engines ---------------------------------------------------------


def test_full_run_writes_outputs_logs_and_then_refuses(
    synthetic: pd.DataFrame, tmp_path: Path
) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    result, report, predictions_csv, run_id = run_final_eval(
        "FD001", report_dir=tmp_path / "reports", tracking_uri=uri
    )

    csv = pd.read_csv(predictions_csv)
    assert list(csv.columns) == ["unit", "true_rul", "prediction", "lower", "upper"]
    assert csv["unit"].tolist() == list(range(1, 21))  # one row per test engine
    assert csv["true_rul"].tolist() == synthetic["rul"].tolist()  # uncapped truth
    assert (csv["lower"] >= 0).all() and (csv["lower"] <= csv["upper"]).all()

    text = report.read_text()
    for needle in (
        "LightGBM + window features",
        "Linear regression",
        "Mean RUL baseline",
        "paired bootstrap",
        "Wilson",
        "90+",
    ):
        assert needle in text
    assert "Forced re-run" not in text
    assert result.overall_coverage["engines"] == 20
    assert result.coverage_by_bin["engines"].sum() == 20

    run = MlflowClient(tracking_uri=uri).get_run(run_id)
    assert run.data.tags["stage"] == "final_test"
    assert run.data.tags["forced_rerun"] == "false"
    assert "test_lgbm_rmse" in run.data.metrics and "test_rmse_diff_ci_high" in run.data.metrics
    artifacts = {a.path for a in MlflowClient(tracking_uri=uri).list_artifacts(run_id)}
    assert artifacts == {report.name, predictions_csv.name}

    with pytest.raises(FinalEvalAlreadyDone):  # second run refused
        run_final_eval("FD001", report_dir=tmp_path / "reports", tracking_uri=uri)


def test_forced_rerun_is_labelled(synthetic: pd.DataFrame, tmp_path: Path) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    reports = tmp_path / "reports"
    run_final_eval("FD001", report_dir=reports, tracking_uri=uri)
    _, report, _, run_id = run_final_eval("FD001", force=True, report_dir=reports, tracking_uri=uri)
    assert "Forced re-run" in report.read_text()
    assert MlflowClient(tracking_uri=uri).get_run(run_id).data.tags["forced_rerun"] == "true"


class FeatureEcho:
    """Stand-in model whose "prediction" is one window feature, to inspect the test inputs."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureEcho":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["s_2_mean_30"].to_numpy()


def test_predicts_at_last_cycle_from_full_causal_history(
    synthetic: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(final_eval, "MODEL_FACTORY", FeatureEcho)
    result = final_eval.evaluate("FD001")
    test = final_eval.load_test("FD001")
    expected = (
        test.sort_values(["unit", "cycle"])
        .groupby("unit")["s_2"]
        .apply(lambda s: s.tail(30).mean())
    )
    np.testing.assert_allclose(result.predictions["prediction"], expected.to_numpy())


# --- Wilson interval ---------------------------------------------------------------------------


def test_wilson_interval_known_values() -> None:
    low, high = wilson_interval(18, 20)
    assert low == pytest.approx(0.699, abs=0.002) and high == pytest.approx(0.972, abs=0.002)
    low, high = wilson_interval(20, 20)  # all covered: upper bound 1, lower well below
    assert high == 1.0 and 0.8 < low < 0.85
    assert all(np.isnan(wilson_interval(0, 0)))
