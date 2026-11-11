"""Tests for the run_cv command line and cap sweep (synthetic data, temporary MLflow DB)."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from predmaint.features.selection import KEPT_SENSORS
from predmaint.tracking import TRACKING_DB
from predmaint.training import run_cv
from predmaint.training.run_cv import build_parser, format_table, run_caps

BASE_ARGS = ["--model", "linear", "--features", "current", "--subset", "FD001"]


@pytest.fixture(autouse=True)
def real_db_untouched() -> Iterator[None]:
    """Fail any test here that creates or modifies the real mlflow.db."""

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
    """Make run_cv load 8 synthetic engines (lifetimes 170-205) instead of the real files."""
    rng = np.random.default_rng(0)
    frames = []
    for unit in range(1, 9):
        n = 165 + 5 * unit
        frame = pd.DataFrame(rng.normal(size=(n, len(KEPT_SENSORS))), columns=KEPT_SENSORS)
        frame[KEPT_SENSORS[0]] += np.linspace(0, 5, n)  # a sensor that drifts towards failure
        frame.insert(0, "unit", unit)
        frame.insert(1, "cycle", np.arange(1, n + 1))
        frames.append(frame)
    fake = pd.concat(frames, ignore_index=True)
    monkeypatch.setattr(run_cv, "load_train", lambda subset: fake.copy())


# --- argument parsing -----------------------------------------------------------------


def test_default_cap_is_125() -> None:
    args = build_parser().parse_args(BASE_ARGS)
    assert args.cap == 125 and args.caps is None


def test_caps_parsed_as_integers() -> None:
    args = build_parser().parse_args([*BASE_ARGS, "--caps", "110", "125", "150"])
    assert args.caps == [110, 125, 150]


def test_cap_and_caps_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([*BASE_ARGS, "--cap", "120", "--caps", "110", "130"])


# --- sweep ---------------------------------------------------------------------------


def test_sweep_returns_one_sorted_row_per_unique_cap(synthetic_train: None) -> None:
    table = run_caps("linear", "current", "FD001", [140, 110, 125, 110], log=False)
    assert table["cap"].tolist() == [110, 125, 140]
    assert table["run_id"].isna().all()
    assert (table[["rmse_mean", "mae_mean"]] > 0).all().all()


def test_sweep_logs_each_cap_as_its_own_run(synthetic_train: None, tmp_path: Path) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    table = run_caps("linear", "window", "FD001", [110, 130], windows=(5,), tracking_uri=uri)
    client = MlflowClient(tracking_uri=uri)
    assert table["run_id"].nunique() == 2
    for cap, run_id, rmse_mean in zip(
        table["cap"], table["run_id"], table["rmse_mean"], strict=True
    ):
        run = client.get_run(run_id)
        assert run.data.params["cap"] == str(cap)
        assert run.data.params["cap_sweep"] == "110,130"
        assert run.data.params["feature_set"] == "window"
        assert run.data.params["windows"] == "5"
        assert run.info.run_name == f"linear regression | window | cap {cap}"
        assert run.data.metrics["cv_rmse_mean"] == pytest.approx(rmse_mean)


def test_single_cap_run_has_no_sweep_param(synthetic_train: None, tmp_path: Path) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    table = run_caps("mean", "current", "FD001", [125], tracking_uri=uri)
    params = MlflowClient(tracking_uri=uri).get_run(table["run_id"].iloc[0]).data.params
    assert params["cap"] == "125"
    assert "cap_sweep" not in params


@pytest.mark.parametrize("caps", [[], [0, 125], [-5]])
def test_invalid_caps_raise(synthetic_train: None, caps: list[int]) -> None:
    with pytest.raises(ValueError, match="caps"):
        run_caps("mean", "current", "FD001", caps, log=False)


# --- table ---------------------------------------------------------------------------


def test_format_table_sorted_and_marks_lowest_rmse() -> None:
    table = pd.DataFrame(
        {
            "cap": [150, 110, 125],
            "rmse_mean": [20.0, 19.0, 18.5],
            "rmse_std": [1.0, 1.0, 1.0],
            "mae_mean": [15.0, 14.0, 14.5],
            "mae_std": [0.5, 0.5, 0.5],
        }
    )
    lines = format_table(table).splitlines()
    assert [line.split()[0] for line in lines[2:5]] == ["110", "125", "150"]
    assert lines[3].endswith("*") and lines[3].startswith("  125")
    assert not lines[2].endswith("*") and not lines[4].endswith("*")
    assert "18.50 ± 1.00" in lines[3]
