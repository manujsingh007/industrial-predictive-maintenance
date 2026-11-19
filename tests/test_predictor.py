"""Tests for the serving predictor, the build step and the MLflow registry round trip.

Synthetic data only; MLflow uses a temporary database; the real mlflow.db is guarded.
"""

from collections.abc import Iterator
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from predmaint.features.selection import KEPT_SENSORS
from predmaint.models.conformal import MONDRIAN_EDGES
from predmaint.models.gbdt import make_lgbm
from predmaint.serving.predictor import InvalidHistoryError, make_predictor
from predmaint.serving.registry import load_champion, register_champion
from predmaint.tracking import TRACKING_DB
from predmaint.training import build_model, final_eval
from predmaint.training.run_cv import build_features

WINDOWS = (10, 30)
Q = (6.0, 16.0, 29.0, 33.0)


@pytest.fixture(autouse=True)
def real_db_untouched() -> Iterator[None]:
    def snapshot() -> tuple[int, int] | None:
        if not TRACKING_DB.exists():
            return None
        return TRACKING_DB.stat().st_size, TRACKING_DB.stat().st_mtime_ns

    mlruns = TRACKING_DB.parent / "mlruns"

    def artifacts() -> list[str]:
        return sorted(p.name for p in mlruns.glob("*")) if mlruns.exists() else []

    before, uri_before, artifacts_before = snapshot(), mlflow.get_tracking_uri(), artifacts()
    yield
    assert snapshot() == before
    assert artifacts() == artifacts_before  # no artifacts in the real mlruns/
    assert mlflow.get_tracking_uri() == uri_before  # registry helpers must not leak global state


def engine(unit: int, n: int, rng: np.random.Generator) -> pd.DataFrame:
    frame = pd.DataFrame(rng.normal(size=(n, len(KEPT_SENSORS))), columns=KEPT_SENSORS)
    frame[KEPT_SENSORS[0]] += np.linspace(0, 6, n)
    frame.insert(0, "unit", unit)
    frame.insert(1, "cycle", np.arange(1, n + 1))
    frame.insert(2, "setting_1", rng.normal(0, 0.002, n))  # raw columns serving should ignore
    frame["s_1"] = 518.67
    return frame


@pytest.fixture
def fleet() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(training engines with rul, truncated test engines without rul)."""
    rng = np.random.default_rng(0)
    train = pd.concat([engine(u, 140 + 6 * u, rng) for u in range(1, 13)], ignore_index=True)
    train["rul"] = train.groupby("unit")["cycle"].transform("max") - train["cycle"]
    test = pd.concat(
        [engine(u, 150 + 3 * u, rng).iloc[: 5 + 9 * u] for u in range(1, 16)], ignore_index=True
    )
    return train, test


@pytest.fixture
def predictor(fleet: tuple[pd.DataFrame, pd.DataFrame]):
    train, _ = fleet
    data, features = build_features(train, "window", WINDOWS)
    model = make_lgbm(n_estimators=30).fit(data[features], np.minimum(data["rul"], 125))
    return make_predictor(
        model,
        sensors=KEPT_SENSORS,
        windows=WINDOWS,
        cap=125,
        bin_edges=MONDRIAN_EDGES,
        q_per_bin=Q,
        alpha=0.1,
        subset="FD001",
    )


# --- training-serving skew -------------------------------------------------------------------


def test_serving_matches_training_pipeline_at_last_cycle(predictor, fleet) -> None:
    _, test = fleet
    data, features = build_features(test, "window", WINDOWS)
    expected = predictor.model.predict(final_eval._last_cycle_rows(data)[features])
    served = predictor.predict_engines(test)
    np.testing.assert_allclose(served["rul_raw"], expected, rtol=0, atol=1e-9)
    assert served["n_cycles_seen"].tolist() == test.groupby("unit").size().tolist()


def test_serving_matches_training_pipeline_for_every_history_prefix(predictor, fleet) -> None:
    # Live use: an engine is predicted after each new cycle. Every prefix must equal the
    # training pipeline's (causal) features at that cycle.
    _, test = fleet
    one = test[test["unit"] == 15].reset_index(drop=True)  # 140 cycles
    data, features = build_features(one, "window", WINDOWS)
    expected = predictor.model.predict(data[features])
    for k in (1, 2, 9, 10, 11, 29, 30, 31, 100, len(one)):
        served = predictor.predict_engine(one.iloc[:k])
        assert served.rul_raw == pytest.approx(expected[k - 1], abs=1e-9)
        assert served.n_cycles_seen == k and served.last_cycle == k


def test_extra_columns_order_and_index_do_not_matter(predictor, fleet) -> None:
    _, test = fleet
    one = test[test["unit"] == 7]
    shuffled = one[list(reversed(one.columns))].set_index(one.index + 1000)
    trimmed = one[["unit", "cycle", *KEPT_SENSORS]]
    a, b, c = (predictor.predict_engine(x) for x in (one, shuffled, trimmed))
    assert a == b == c


def test_build_predictor_reproduces_final_eval_exactly(
    fleet, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: the served model and the evaluated model give identical outputs."""
    train, test = fleet
    test_ids = test["unit"].unique()
    truth = pd.DataFrame({"unit": test_ids, "rul": np.full(len(test_ids), 50)})
    monkeypatch.setattr(final_eval, "load_train", lambda s: train.drop(columns="rul"))
    monkeypatch.setattr(final_eval, "load_test", lambda s: test.copy())
    monkeypatch.setattr(final_eval, "load_test_rul", lambda s: truth.copy())
    monkeypatch.setattr(final_eval, "MODEL_FACTORY", lambda: make_lgbm(n_estimators=30))
    monkeypatch.setattr(final_eval, "N_RESAMPLES", 50)

    evaluated = final_eval.evaluate("FD001").predictions
    served_predictor, params, metrics = build_model.build_predictor("FD001")
    served = served_predictor.predict_engines(test)
    np.testing.assert_allclose(served["rul_raw"], evaluated["prediction"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(served["lower"], evaluated["lower"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(served["upper"], evaluated["upper"], rtol=0, atol=1e-9)
    assert params["windows"] == "10,30" and set(metrics) == {
        "q_0_30",
        "q_30_60",
        "q_60_90",
        "q_90_plus",
    }


# --- input validation ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.iloc[0:0], "empty"),
        (lambda d: d.drop(columns="s_11"), "missing required columns: \\['s_11'\\]"),
        (lambda d: d.drop(columns="cycle"), "missing required columns: \\['cycle'\\]"),
        (lambda d: d.assign(unit=[1] * 5 + [2] * (len(d) - 5)), "exactly one engine"),
        (lambda d: d.iloc[::-1], "sorted by cycle"),
        (lambda d: pd.concat([d, d.iloc[[-1]]]), "unique; duplicated: \\[20\\]"),
        (lambda d: d.assign(cycle=d["cycle"] - 1), "whole numbers >= 1"),
        (lambda d: d.assign(cycle=d["cycle"] + 0.5), "whole numbers >= 1"),
        (lambda d: d.assign(s_4=np.r_[np.nan, d["s_4"].iloc[1:]]), "finite numbers.*s_4"),
        (lambda d: d.assign(s_7="high"), "finite numbers.*s_7"),
    ],
    ids=[
        "empty",
        "missing sensor",
        "missing cycle",
        "two units",
        "unsorted",
        "duplicate",
        "cycle 0",
        "fractional cycle",
        "NaN sensor",
        "text sensor",
    ],
)
def test_invalid_histories_give_clear_errors(predictor, fleet, mutate, message) -> None:
    _, test = fleet
    one = test[test["unit"] == 2].reset_index(drop=True)  # 23 cycles
    one = one.iloc[:20]
    with pytest.raises(InvalidHistoryError, match=message):
        predictor.predict_engine(mutate(one))


def test_non_dataframe_is_rejected(predictor) -> None:
    with pytest.raises(InvalidHistoryError, match="DataFrame"):
        predictor.predict_engine([{"unit": 1, "cycle": 1}])


# --- intervals ------------------------------------------------------------------------------------


def test_interval_contains_point_and_lower_non_negative(predictor, fleet) -> None:
    _, test = fleet
    out = predictor.predict_engines(test)
    assert (out["lower"] >= 0).all()
    assert ((out["lower"] <= out["rul"]) & (out["rul"] <= out["upper"])).all()


class NegativeModel:
    """Predicts -3: like an engine at failure where boosted trees overshoot below 0."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), -3.0)


def test_negative_raw_prediction_is_clipped_consistently(fleet) -> None:
    _, test = fleet
    p = make_predictor(
        NegativeModel(),
        sensors=KEPT_SENSORS,
        windows=WINDOWS,
        cap=125,
        bin_edges=MONDRIAN_EDGES,
        q_per_bin=(2.0, 16.0, 29.0, 33.0),
        alpha=0.1,
        subset="FD001",
    )
    out = p.predict_engine(test[test["unit"] == 1])
    assert out.rul_raw == -3.0 and out.rul == 0.0
    assert out.lower == 0.0 and out.upper == 0.0  # raw + q = -1, raised to rul = 0
    assert out.pred_bin == "0-30"


def test_q_count_must_match_bins() -> None:
    with pytest.raises(ValueError, match="q values"):
        make_predictor(
            None,
            sensors=KEPT_SENSORS,
            windows=WINDOWS,
            cap=125,
            bin_edges=MONDRIAN_EDGES,
            q_per_bin=(1.0, 2.0),
            alpha=0.1,
            subset="FD001",
        )


# --- registry round trip ------------------------------------------------------------


def test_register_and_load_champion_round_trip(predictor, fleet, tmp_path: Path) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifacts = (tmp_path / "mlruns").as_uri()
    _, test = fleet

    v1 = register_champion(
        predictor,
        params={"subset": "FD001"},
        metrics={"q_0_30": 6.0},
        tracking_uri=uri,
        artifact_location=artifacts,
    )
    loaded = load_champion(tracking_uri=uri)
    pd.testing.assert_frame_equal(loaded.predict_engines(test), predictor.predict_engines(test))
    assert loaded.q_per_bin == predictor.q_per_bin and loaded.windows == predictor.windows

    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    assert str(client.get_model_version_by_alias("rul-lgbm-fd001", "champion").version) == v1
    run = client.get_run(client.get_model_version("rul-lgbm-fd001", v1).run_id)
    assert run.data.tags["stage"] == "build_model" and "git_commit" in run.data.tags

    v2 = register_champion(predictor, tracking_uri=uri, artifact_location=artifacts)
    assert int(v2) == int(v1) + 1
    assert str(client.get_model_version_by_alias("rul-lgbm-fd001", "champion").version) == v2


def test_pyfunc_interface_predicts_several_engines(predictor, fleet, tmp_path: Path) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    register_champion(predictor, tracking_uri=uri, artifact_location=(tmp_path / "a").as_uri())
    from predmaint.serving.registry import _tracking_uri

    _, test = fleet
    with _tracking_uri(uri):
        pyfunc = mlflow.pyfunc.load_model("models:/rul-lgbm-fd001@champion")
    out = pyfunc.predict(test)
    assert len(out) == test["unit"].nunique()
    assert {"rul", "lower", "upper", "n_cycles_seen"} <= set(out.columns)


def test_reference_profile_travels_with_the_model_version(predictor, fleet, tmp_path: Path) -> None:
    from predmaint.monitoring.drift import build_reference_profile
    from predmaint.serving.registry import load_champion_info

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifacts = (tmp_path / "mlruns").as_uri()
    train, _ = fleet
    profile = build_reference_profile(train, ["setting_1", *KEPT_SENSORS])

    v1 = register_champion(predictor, tracking_uri=uri, artifact_location=artifacts)
    assert load_champion_info(tracking_uri=uri).reference_profile is None  # older-style model

    v2 = register_champion(
        predictor, reference_profile=profile, tracking_uri=uri, artifact_location=artifacts
    )
    assert int(v2) == int(v1) + 1  # rebuilding creates a new version ...
    loaded = load_champion_info(tracking_uri=uri)
    assert loaded.version == v2  # ... and the champion alias moves to it
    assert loaded.reference_profile == profile

    # The profile is a readable JSON file inside the registered model version itself.
    from predmaint.monitoring.drift import ReferenceProfile
    from predmaint.serving.registry import _tracking_uri

    with _tracking_uri(uri):
        local = mlflow.artifacts.download_artifacts(
            f"models:/rul-lgbm-fd001/{v2}", dst_path=str(tmp_path / "download")
        )
    files = list(Path(local).rglob("reference_profile.json"))
    assert len(files) == 1
    assert ReferenceProfile.from_json(files[0].read_text()) == profile
