"""Prepare a fresh deployment: data download and champion model, skipped if already done.

Run with: ``python -m predmaint.training.bootstrap --subset FD001`` (the Docker ``model`` service).

1. If the MLflow registry (``MLFLOW_TRACKING_URI``) already has a ``champion`` model, stop.
   Restarts are fast, and an existing model is never replaced by accident.
2. Otherwise download the C-MAPSS data into ``PREDMAINT_DATA_DIR`` (checksum-verified; skipped
   if the files are already there) and run ``build_model``, which trains, calibrates and
   registers the champion together with its drift reference profile.
"""

import argparse

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from predmaint import config
from predmaint.data.download import DEFAULT_RAW_DIR, download_cmapss
from predmaint.features.selection import SUBSETS_WITH_ADR
from predmaint.serving.registry import CHAMPION_ALIAS, MODEL_NAME
from predmaint.tracking import ARTIFACT_ROOT, TRACKING_DB, TRACKING_URI
from predmaint.training import build_model


def champion_exists(tracking_uri: str = TRACKING_URI) -> str | None:
    """Version the champion alias points to, or None if there is none yet."""
    try:
        client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
        return str(client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS).version)
    except MlflowException:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download data and build the model if needed.")
    parser.add_argument("--subset", required=True, choices=SUBSETS_WITH_ADR)
    args = parser.parse_args(argv)

    for directory in (
        TRACKING_DB.parent if TRACKING_DB else None,
        config.artifact_path(ARTIFACT_ROOT),
    ):
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)

    version = champion_exists()
    if version is not None:
        print(f"{MODEL_NAME}@{CHAMPION_ALIAS} already exists (version {version}); nothing to do.")
        return 0
    print(f"No {MODEL_NAME}@{CHAMPION_ALIAS} yet: downloading data and building the model.")
    path = download_cmapss(DEFAULT_RAW_DIR)
    print(f"C-MAPSS data ready in {path}")
    return build_model.main(["--subset", args.subset])


if __name__ == "__main__":
    raise SystemExit(main())
