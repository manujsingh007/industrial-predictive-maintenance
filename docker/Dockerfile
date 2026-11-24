# syntax=docker/dockerfile:1
#
# Two images from one Dockerfile:
#   docker build -f docker/Dockerfile --target app .        API, migrations, model build, replay
#   docker build -f docker/Dockerfile --target dashboard .  Streamlit dashboard
#
# Multi-stage: "build-*" stages contain uv and create the virtual environment; the final
# stages copy only that environment and the code onto a slim base, so no build tools ship.

# uv from its official image, pinned to the version this project is developed with.
FROM ghcr.io/astral-sh/uv:0.13.7 AS uv

# --- runtime base: what every final image needs ----------------------------------------------
FROM python:3.12-slim AS base
# libgomp1: the OpenMP runtime that LightGBM's Linux wheel needs (libomp on macOS).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# Non-root user. /state is where the named volume (data + MLflow store) is mounted; creating
# it here owned by "app" means a fresh volume inherits that ownership.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /state \
    && chown app:app /state
ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app

# --- build stages: locked dependencies first, code last (layer caching) -----------------------
FROM base AS build
COPY --from=uv /uv /usr/local/bin/uv
# Compile .pyc once at build time; copy files instead of hard-linking out of the cache mount;
# use the image's Python (never download one).
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock ./

FROM build AS build-app
# Dependencies only: this layer is reused as long as pyproject.toml and uv.lock are unchanged.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project
COPY README.md alembic.ini ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

FROM build AS build-dashboard
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group dashboard --no-install-project
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group dashboard

# --- final images ---------------------------------------------------------------------------
FROM base AS app
COPY --from=build-app --chown=app:app /app /app
USER app
ENV PREDMAINT_HOME=/app \
    PREDMAINT_DATA_DIR=/state/data \
    PREDMAINT_REPORTS_DIR=/state/reports \
    MLFLOW_TRACKING_URI=sqlite:////state/mlflow/mlflow.db \
    PREDMAINT_ARTIFACT_ROOT=/state/mlflow/artifacts
EXPOSE 8000
CMD ["uvicorn", "predmaint.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS dashboard
COPY --from=build-dashboard --chown=app:app /app /app
USER app
ENV API_BASE_URL=http://api:8000
EXPOSE 8501
CMD ["streamlit", "run", "src/predmaint/dashboard/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
