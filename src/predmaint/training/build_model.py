"""Train the frozen model on all training engines and register it as the serving champion.

Run with: ``uv run python -m predmaint.training.build_model --subset FD001``

Training and calibration call :func:`predmaint.training.final_eval.train_frozen`, the exact code
used for the official test evaluation, so the served model is the evaluated model. No test data
is read. The predictor is registered in MLflow as ``rul-lgbm-fd001`` with the alias
``champion`` (see :mod:`predmaint.serving.registry` for the format choice), together
with the drift reference profile of the training readings. Rebuilding registers a new version
and moves the alias to it.
"""

import argparse

from predmaint.features.selection import DEFAULT_RUL_CAP, KEPT_SENSORS, SUBSETS_WITH_ADR
from predmaint.models.conformal import MONDRIAN_EDGES, mondrian_bin_labels
from predmaint.monitoring.drift import build_reference_profile
from predmaint.serving.predictor import RULPredictor, make_predictor
from predmaint.training import final_eval
from predmaint.training.run_cv import DEFAULT_WINDOWS


def build_predictor(subset: str) -> tuple[RULPredictor, dict[str, object], dict[str, float]]:
    """Train via ``final_eval.train_frozen`` and wrap the result; return params and metrics too."""
    frozen = final_eval.train_frozen(subset)
    predictor = make_predictor(
        frozen.model,
        sensors=KEPT_SENSORS,
        windows=DEFAULT_WINDOWS,
        cap=DEFAULT_RUL_CAP,
        bin_edges=MONDRIAN_EDGES,
        q_per_bin=frozen.q_per_bin,
        alpha=final_eval.ALPHA,
        subset=subset,
    )
    if predictor.feature_names != frozen.features:
        raise RuntimeError("serving feature order differs from training; refusing to register")
    params = {
        "subset": subset,
        "model": "LightGBM",
        "feature_set": "window",
        "windows": ",".join(map(str, DEFAULT_WINDOWS)),
        "cap": DEFAULT_RUL_CAP,
        "conformal_method": "mondrian",
        "conformal_alpha": final_eval.ALPHA,
        "mondrian_bins": ",".join(mondrian_bin_labels(MONDRIAN_EDGES)),
    }
    metrics = {
        f"q_{label.replace('-', '_').replace('+', '_plus')}": q
        for label, q in zip(frozen.q_by_bin["bin"], frozen.q_per_bin, strict=True)
    }
    return predictor, params, metrics


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Build and register the serving model.")
    parser.add_argument("--subset", required=True, choices=SUBSETS_WITH_ADR)
    args = parser.parse_args(argv)

    from predmaint.serving.registry import CHAMPION_ALIAS, MODEL_NAME, register_champion

    predictor, params, metrics = build_predictor(args.subset)
    profile = build_reference_profile(final_eval.load_train(args.subset))
    version = register_champion(
        predictor, reference_profile=profile, params=params, metrics=metrics
    )
    print(f"Registered {MODEL_NAME} version {version} with alias '{CHAMPION_ALIAS}'.")
    print(
        f"Drift reference profile: {len(profile.columns)} columns from {profile.n_rows} "
        "training readings (stored with the model)."
    )
    print(
        "q per predicted-RUL bin:",
        dict(zip(params["mondrian_bins"].split(","), predictor.q_per_bin, strict=True)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
