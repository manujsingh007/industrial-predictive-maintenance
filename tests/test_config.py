"""Tests for predmaint.config: every location comes from env vars or the project directory."""

from pathlib import Path

import pytest

from predmaint import config

SRC = Path(__file__).parents[1] / "src" / "predmaint"
ENV_VARS = [
    "PREDMAINT_HOME",
    "PREDMAINT_DATA_DIR",
    "PREDMAINT_REPORTS_DIR",
    "MLFLOW_TRACKING_URI",
    "PREDMAINT_ARTIFACT_ROOT",
    "API_BASE_URL",
]


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (tmp_path / "notebooks").mkdir()
    monkeypatch.chdir(tmp_path / "notebooks")  # like running a notebook from a subfolder
    return tmp_path.resolve()


def test_project_root_is_found_upwards_from_cwd(project: Path) -> None:
    assert config.project_root() == project


def test_project_root_falls_back_to_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PREDMAINT_HOME", raising=False)
    monkeypatch.chdir(tmp_path)  # no pyproject.toml here or above (tmp dirs)
    if any((d / "pyproject.toml").exists() for d in (tmp_path, *tmp_path.parents)):
        pytest.skip("a pyproject.toml exists above the temporary directory")
    assert config.project_root() == tmp_path.resolve()


def test_defaults_live_in_the_project(project: Path) -> None:
    assert config.data_dir() == project / "data"
    assert config.raw_data_dir() == project / "data" / "raw" / "cmapss"
    assert config.reports_dir() == project / "reports"
    assert config.tracking_uri() == f"sqlite:///{project / 'mlflow.db'}"
    assert config.artifact_root() == (project / "mlruns").as_uri()
    assert config.env_file() == project / ".env"
    assert config.api_base_url() == "http://127.0.0.1:8000"


def test_environment_variables_win(
    project: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PREDMAINT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PREDMAINT_DATA_DIR", "/state/data")
    monkeypatch.setenv("PREDMAINT_REPORTS_DIR", "/state/reports")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:////state/mlflow/mlflow.db")
    monkeypatch.setenv("PREDMAINT_ARTIFACT_ROOT", "/state/mlflow/artifacts")
    monkeypatch.setenv("API_BASE_URL", "http://api:8000/")
    assert config.project_root() == (tmp_path / "home").resolve()
    assert config.raw_data_dir() == Path("/state/data/raw/cmapss")
    assert config.reports_dir() == Path("/state/reports")
    assert config.tracking_uri() == "sqlite:////state/mlflow/mlflow.db"
    assert config.sqlite_path(config.tracking_uri()) == Path("/state/mlflow/mlflow.db")
    assert config.artifact_root() == "file:///state/mlflow/artifacts"  # path -> file URI
    assert config.artifact_path(config.artifact_root()) == Path("/state/mlflow/artifacts")
    assert config.api_base_url() == "http://api:8000"


@pytest.mark.parametrize("uri", ["file:///data/mlruns", "s3://bucket/mlruns"])
def test_artifact_root_uris_pass_through(monkeypatch: pytest.MonkeyPatch, uri: str) -> None:
    monkeypatch.setenv("PREDMAINT_ARTIFACT_ROOT", uri)
    assert config.artifact_root() == uri


def test_non_local_stores_have_no_local_path() -> None:
    assert config.sqlite_path("postgresql://host/mlflow") is None
    assert config.artifact_path("s3://bucket/mlruns") is None


def test_no_module_derives_paths_from_its_source_location() -> None:
    """Paths from __file__ break once the package is installed (e.g. in a Docker image)."""
    offenders = [str(p.relative_to(SRC)) for p in SRC.rglob("*.py") if "__file__" in p.read_text()]
    assert offenders == []
