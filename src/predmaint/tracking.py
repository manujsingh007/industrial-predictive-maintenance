"""MLflow experiment tracking: the one place where the tracking backend is configured.

All runs go to a SQLite database at ``<project root>/mlflow.db`` (git-ignored). The URI is
built from this file's location, so it is the same no matter which directory code runs from.
Open the UI with: ``uv run mlflow ui --backend-store-uri sqlite:///mlflow.db``
"""

import subprocess
import time
from pathlib import Path

from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient

from predmaint.data.download import PROJECT_ROOT
from predmaint.training.cv import CVResult

TRACKING_DB = PROJECT_ROOT / "mlflow.db"
TRACKING_URI = f"sqlite:///{TRACKING_DB}"
ARTIFACT_ROOT = (PROJECT_ROOT / "mlruns").as_uri()


def git_tags(repo_dir: Path = PROJECT_ROOT) -> dict[str, str]:
    """Return ``git_commit`` (HEAD hash) and ``git_dirty`` ("true"/"false") for ``repo_dir``.

    "Dirty" means uncommitted changes or untracked files, so the run may not be reproducible
    from the commit alone. Falls back to "unknown" if git is unavailable.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "git_dirty": "unknown"}
    return {"git_commit": commit, "git_dirty": "true" if status.strip() else "false"}


def _get_or_create_experiment(client: MlflowClient, name: str, artifact_location: str) -> str:
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    return client.create_experiment(name, artifact_location=artifact_location)


def log_cv_result(
    result: CVResult,
    *,
    model_name: str,
    subset: str,
    experiment_name: str | None = None,
    tracking_uri: str = TRACKING_URI,
    artifact_location: str = ARTIFACT_ROOT,
) -> str:
    """Log a cross-validation result as one finished MLflow run.

    Logs params (model, subset, cap, n_splits, seed, n_features, max_eval_rul), per-fold
    ``rmse``/``mae`` with ``step`` = fold number, the summary metrics ``cv_rmse_mean``,
    ``cv_rmse_std``, ``cv_mae_mean``, ``cv_mae_std``, and the git tags from :func:`git_tags`.

    Args:
        result: Output of :func:`predmaint.training.cv.cross_validate`.
        model_name: Human-readable model name; also used as the run name.
        subset: Data subset, e.g. "FD001".
        experiment_name: Defaults to ``rul-<subset>`` in lower case, e.g. "rul-fd001".
        tracking_uri: MLflow backend; tests pass a temporary database here.
        artifact_location: Artifact root used if the experiment has to be created.

    Returns:
        The MLflow run id.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = _get_or_create_experiment(
        client, experiment_name or f"rul-{subset.lower()}", artifact_location
    )
    run = client.create_run(experiment_id, run_name=model_name)
    run_id = run.info.run_id

    params = {
        "model": model_name,
        "subset": subset,
        "cap": result.cap,
        "n_splits": result.n_splits,
        "seed": result.seed,
        "n_features": len(result.features),
        "max_eval_rul": result.max_eval_rul,
    }
    now = int(time.time() * 1000)
    metrics = [
        Metric(key, float(row[key]), now, int(row["fold"]))
        for _, row in result.folds.iterrows()
        for key in ("rmse", "mae")
    ]
    summary = result.summary()
    metrics += [
        Metric(f"cv_{key}_{stat}", float(summary.loc[stat, key]), now, 0)
        for key in ("rmse", "mae")
        for stat in ("mean", "std")
    ]

    try:
        client.log_batch(
            run_id,
            metrics=metrics,
            params=[Param(k, str(v)) for k, v in params.items()],
            tags=[RunTag(k, v) for k, v in git_tags().items()],
        )
    except Exception:
        client.set_terminated(run_id, status="FAILED")
        raise
    client.set_terminated(run_id)
    return run_id
