"""Tests for predmaint.tracking. Every test uses a temporary MLflow database in tmp_path."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from predmaint.models.baselines import make_mean_baseline
from predmaint.tracking import TRACKING_DB, git_tags, log_cv_result
from predmaint.training.cv import CVResult, cross_validate


def _snapshot(path: Path) -> tuple[bool, int, int] | None:
    return (True, path.stat().st_size, path.stat().st_mtime_ns) if path.exists() else None


@pytest.fixture(autouse=True)
def real_db_untouched() -> Iterator[None]:
    """Fail any test in this module that creates or modifies the real mlflow.db."""
    before = _snapshot(TRACKING_DB)
    yield
    assert _snapshot(TRACKING_DB) == before, f"test modified the real {TRACKING_DB}"


@pytest.fixture
def tmp_tracking(tmp_path: Path) -> dict[str, str]:
    return {
        "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
        "artifact_location": (tmp_path / "mlruns").as_uri(),
    }


@pytest.fixture
def cv_result() -> CVResult:
    rng = np.random.default_rng(0)
    frames = []
    for unit in range(1, 7):
        n = 20 + unit
        frames.append(
            pd.DataFrame(
                {
                    "unit": unit,
                    "cycle": np.arange(1, n + 1),
                    "rul": np.arange(n - 1, -1, -1),
                    "a": rng.normal(size=n),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    return cross_validate(make_mean_baseline, df, cap=15, features=["a"], n_splits=3, seed=7)


def test_creates_finished_run_in_named_experiment(
    cv_result: CVResult, tmp_tracking: dict[str, str]
) -> None:
    run_id = log_cv_result(cv_result, model_name="mean RUL", subset="FD001", **tmp_tracking)
    client = MlflowClient(tracking_uri=tmp_tracking["tracking_uri"])
    run = client.get_run(run_id)
    assert client.get_experiment(run.info.experiment_id).name == "rul-fd001"
    assert run.info.run_name == "mean RUL"
    assert run.info.status == "FINISHED"


def test_logs_expected_params(cv_result: CVResult, tmp_tracking: dict[str, str]) -> None:
    run_id = log_cv_result(cv_result, model_name="mean RUL", subset="FD001", **tmp_tracking)
    params = MlflowClient(tracking_uri=tmp_tracking["tracking_uri"]).get_run(run_id).data.params
    assert params == {
        "model": "mean RUL",
        "subset": "FD001",
        "cap": "15",
        "n_splits": "3",
        "seed": "7",
        "n_features": "1",
        "max_eval_rul": "150",
    }


def test_logs_per_fold_metrics_with_fold_as_step(
    cv_result: CVResult, tmp_tracking: dict[str, str]
) -> None:
    run_id = log_cv_result(cv_result, model_name="m", subset="FD001", **tmp_tracking)
    client = MlflowClient(tracking_uri=tmp_tracking["tracking_uri"])
    for key in ("rmse", "mae"):
        history = sorted(client.get_metric_history(run_id, key), key=lambda m: m.step)
        assert [m.step for m in history] == [0, 1, 2]
        assert [m.value for m in history] == pytest.approx(cv_result.folds[key].tolist())


def test_logs_summary_metrics(cv_result: CVResult, tmp_tracking: dict[str, str]) -> None:
    run_id = log_cv_result(cv_result, model_name="m", subset="FD001", **tmp_tracking)
    metrics = MlflowClient(tracking_uri=tmp_tracking["tracking_uri"]).get_run(run_id).data.metrics
    summary = cv_result.summary()
    for key in ("rmse", "mae"):
        assert metrics[f"cv_{key}_mean"] == pytest.approx(summary.loc["mean", key])
        assert metrics[f"cv_{key}_std"] == pytest.approx(summary.loc["std", key])


def test_logs_git_tags(cv_result: CVResult, tmp_tracking: dict[str, str]) -> None:
    run_id = log_cv_result(cv_result, model_name="m", subset="FD001", **tmp_tracking)
    tags = MlflowClient(tracking_uri=tmp_tracking["tracking_uri"]).get_run(run_id).data.tags
    assert tags["git_commit"] == git_tags()["git_commit"]
    assert len(tags["git_commit"]) == 40  # this test suite runs inside the repo
    assert tags["git_dirty"] in {"true", "false"}


def test_second_run_reuses_experiment(cv_result: CVResult, tmp_tracking: dict[str, str]) -> None:
    a = log_cv_result(cv_result, model_name="a", subset="FD001", **tmp_tracking)
    b = log_cv_result(cv_result, model_name="b", subset="FD001", **tmp_tracking)
    client = MlflowClient(tracking_uri=tmp_tracking["tracking_uri"])
    assert client.get_run(a).info.experiment_id == client.get_run(b).info.experiment_id


def test_git_tags_outside_a_repo_are_unknown(tmp_path: Path) -> None:
    assert git_tags(tmp_path) == {"git_commit": "unknown", "git_dirty": "unknown"}


def test_extra_params_and_run_name_are_logged(
    cv_result: CVResult, tmp_tracking: dict[str, str]
) -> None:
    run_id = log_cv_result(
        cv_result,
        model_name="linear regression",
        subset="FD001",
        run_name="linear regression | window",
        extra_params={"feature_set": "window", "windows": "10,30"},
        **tmp_tracking,
    )
    run = MlflowClient(tracking_uri=tmp_tracking["tracking_uri"]).get_run(run_id)
    assert run.info.run_name == "linear regression | window"
    assert run.data.params["feature_set"] == "window"
    assert run.data.params["windows"] == "10,30"


def test_extra_params_cannot_override_standard_params(
    cv_result: CVResult, tmp_tracking: dict[str, str]
) -> None:
    with pytest.raises(ValueError, match="cap"):
        log_cv_result(
            cv_result, model_name="m", subset="FD001", extra_params={"cap": 999}, **tmp_tracking
        )
