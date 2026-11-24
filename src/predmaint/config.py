"""Every filesystem location and service URL, configurable by environment variables.

Nothing is derived from where the source code lives. When installed in a Docker image the
package sits in site-packages, where such paths would be meaningless.

Resolution order for each setting: its environment variable, then a default inside the
*project directory*. The project directory is ``PREDMAINT_HOME`` if set, otherwise the nearest
directory at or above the current working directory that contains a ``pyproject.toml`` (as git
finds its repository), otherwise the current directory. So scripts and notebooks run from any
subfolder of the project find the same data.

| Variable                  | Default                               |
|---------------------------|---------------------------------------|
| PREDMAINT_DATA_DIR        | <project>/data                        |
| PREDMAINT_REPORTS_DIR     | <project>/reports                     |
| MLFLOW_TRACKING_URI       | sqlite:///<project>/mlflow.db         |
| PREDMAINT_ARTIFACT_ROOT   | <project>/mlruns (path or file:// URI)|
| DATABASE_URL              | from the environment or <project>/.env|
| API_BASE_URL (dashboard)  | http://127.0.0.1:8000                 |
"""

import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


def project_root() -> Path:
    """``PREDMAINT_HOME``, else the nearest directory upwards from cwd with a pyproject.toml."""
    home = os.environ.get("PREDMAINT_HOME")
    if home:
        return Path(home).expanduser().resolve()
    cwd = Path.cwd().resolve()
    for directory in (cwd, *cwd.parents):
        if (directory / "pyproject.toml").is_file():
            return directory
    return cwd


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


def data_dir() -> Path:
    return _path_from_env("PREDMAINT_DATA_DIR", project_root() / "data")


def raw_data_dir() -> Path:
    """Where the downloaded C-MAPSS files live: ``<data dir>/raw/cmapss``."""
    return data_dir() / "raw" / "cmapss"


def reports_dir() -> Path:
    return _path_from_env("PREDMAINT_REPORTS_DIR", project_root() / "reports")


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{project_root() / 'mlflow.db'}"


def sqlite_path(uri: str) -> Path | None:
    """The database file of a ``sqlite:///...`` URI, or None for other backends."""
    return Path(uri.removeprefix("sqlite:///")) if uri.startswith("sqlite:///") else None


def artifact_root() -> str:
    """MLflow artifact location as a URI; ``PREDMAINT_ARTIFACT_ROOT`` may be a path or URI."""
    value = os.environ.get("PREDMAINT_ARTIFACT_ROOT")
    if not value:
        return (project_root() / "mlruns").as_uri()
    if urlparse(value).scheme in ("", "file") and not value.startswith("file:"):
        return Path(value).expanduser().resolve().as_uri()
    return value


def artifact_path(uri: str) -> Path | None:
    """Local directory of a ``file://`` artifact URI, or None for remote stores."""
    parsed = urlparse(uri)
    return Path(url2pathname(parsed.path)) if parsed.scheme == "file" else None


def env_file() -> Path:
    return project_root() / ".env"


def api_base_url() -> str:
    return (os.environ.get("API_BASE_URL") or DEFAULT_API_BASE_URL).rstrip("/")
