"""Register the RUL predictor in the MLflow model registry and load it back by alias.

**Format: an ``mlflow.pyfunc`` model wrapping the pickled :class:`RULPredictor`.** Compared with
pickling the predictor as a plain run artifact, pyfunc gives:

- **registry versions and aliases.** ``models:/rul-lgbm-fd001@champion`` always resolves to
  whichever version currently holds the alias, so serving never hard-codes a run id.
- **MLflow's standard ``predict(DataFrame)`` interface**, which tools such as
  ``mlflow models serve`` understand.

Either way, the predictor is pickled, so the ``predmaint`` package (same version) must be
importable wherever the model is loaded, e.g. inside the serving Docker image.
"""

import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.tracking import MlflowClient

from predmaint.monitoring.drift import ReferenceProfile
from predmaint.serving.predictor import RULPredictor
from predmaint.tracking import ARTIFACT_ROOT, TRACKING_URI, _get_or_create_experiment, git_tags

MODEL_NAME = "rul-lgbm-fd001"
PROFILE_ARTIFACT = "reference_profile"
CHAMPION_ALIAS = "champion"
RUNTIME_PACKAGES = ("lightgbm", "numpy", "pandas", "scikit-learn")


class RULPyfuncModel(mlflow.pyfunc.PythonModel):
    """MLflow wrapper: ``predict`` takes readings of one or more engines, one row per engine out."""

    def __init__(self, predictor: RULPredictor) -> None:
        self.predictor = predictor
        self.reference_profile: ReferenceProfile | None = None

    def load_context(self, context: Any) -> None:
        """Read the drift reference profile shipped with this model version (if any).

        Models registered before drift monitoring existed have no such artifact; they still
        load, with ``reference_profile = None``.
        """
        path = (context.artifacts or {}).get(PROFILE_ARTIFACT)
        self.reference_profile = (
            ReferenceProfile.from_json(Path(path).read_text()) if path else None
        )

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        return self.predictor.predict_engines(model_input)


@contextmanager
def _tracking_uri(uri: str) -> Iterator[None]:
    """Point MLflow's global tracking and registry URIs at ``uri``; restore them afterwards."""
    previous_tracking, previous_registry = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    try:
        yield
    finally:
        mlflow.set_tracking_uri(previous_tracking)
        mlflow.set_registry_uri(previous_registry)


def register_champion(
    predictor: RULPredictor,
    *,
    reference_profile: ReferenceProfile | None = None,
    model_name: str = MODEL_NAME,
    alias: str = CHAMPION_ALIAS,
    params: Mapping[str, object] | None = None,
    metrics: Mapping[str, float] | None = None,
    experiment_name: str | None = None,
    tracking_uri: str = TRACKING_URI,
    artifact_location: str = ARTIFACT_ROOT,
) -> str:
    """Log ``predictor`` as a pyfunc model, register a new version and point ``alias`` at it.

    The drift ``reference_profile`` is stored as a JSON artifact **inside the model**, so it
    belongs to exactly this model version and can be read in the MLflow UI.

    Returns:
        The new registered model version (as a string).
    """
    with _tracking_uri(tracking_uri):
        client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
        experiment_id = _get_or_create_experiment(
            client, experiment_name or f"rul-{predictor.subset.lower()}", artifact_location
        )
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"build {model_name}"):
            mlflow.set_tags({**git_tags(), "stage": "build_model"})
            mlflow.log_params(dict(params or {}))
            mlflow.log_metrics({k: float(v) for k, v in (metrics or {}).items()})
            with tempfile.TemporaryDirectory() as tmp:
                artifacts = {}
                if reference_profile is not None:
                    profile_path = Path(tmp) / "reference_profile.json"
                    profile_path.write_text(reference_profile.to_json())
                    artifacts[PROFILE_ARTIFACT] = str(profile_path)
                info = mlflow.pyfunc.log_model(  # copies the artifact into the model
                    name="model",
                    python_model=RULPyfuncModel(predictor),
                    artifacts=artifacts or None,
                    registered_model_name=model_name,
                    # Pinned explicitly so MLflow does not try to infer requirements by
                    # importing and probing the model; predmaint must be installed separately.
                    pip_requirements=[f"{p}=={version(p)}" for p in RUNTIME_PACKAGES],
                )
        model_version = str(info.registered_model_version)
        client.set_registered_model_alias(model_name, alias, model_version)
    return model_version


@dataclass(frozen=True)
class LoadedModel:
    """A loaded predictor plus which registered model version it is."""

    predictor: RULPredictor
    name: str
    version: str
    reference_profile: ReferenceProfile | None = None  # None for models without drift profile


def load_champion_info(
    model_name: str = MODEL_NAME, *, alias: str = CHAMPION_ALIAS, tracking_uri: str = TRACKING_URI
) -> LoadedModel:
    """Resolve ``alias`` to a version number once, then load exactly that version.

    Loading by number (not by alias) means the reported version is guaranteed to be the one
    loaded, even if someone moves the alias at the same moment.
    """
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    model_version = str(client.get_model_version_by_alias(model_name, alias).version)
    with _tracking_uri(tracking_uri):
        pyfunc_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")
    wrapper = pyfunc_model.unwrap_python_model()
    return LoadedModel(
        wrapper.predictor,
        model_name,
        model_version,
        getattr(wrapper, "reference_profile", None),
    )


def load_champion(
    model_name: str = MODEL_NAME, *, alias: str = CHAMPION_ALIAS, tracking_uri: str = TRACKING_URI
) -> RULPredictor:
    """Load the :class:`RULPredictor` that ``alias`` (default "champion") currently points to."""
    return load_champion_info(model_name, alias=alias, tracking_uri=tracking_uri).predictor
