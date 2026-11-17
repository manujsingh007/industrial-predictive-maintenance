"""Database settings, read from the environment or the project's ``.env`` file."""

from pydantic_settings import BaseSettings, SettingsConfigDict

from predmaint.data.download import PROJECT_ROOT


class DatabaseSettings(BaseSettings):
    """``DATABASE_URL``, e.g. ``postgresql+psycopg://user:password@localhost:5432/predmaint``.

    Environment variables take precedence over ``.env``. When unset, the database endpoints
    answer 503 and the stateless ``/predict`` still works.
    """

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    database_url: str | None = None


def get_database_url() -> str | None:
    return DatabaseSettings().database_url
