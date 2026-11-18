"""FastAPI RUL service: a stateless ``/predict`` plus PostgreSQL-backed unit endpoints.

Run locally: ``uv run uvicorn predmaint.serving.api:app --reload``. The interactive docs are at
http://127.0.0.1:8000/docs.

- The champion model is loaded **once at startup** (FastAPI lifespan), from the MLflow
  registry at ``MLFLOW_TRACKING_URI`` (default: the project's ``mlflow.db``).
- ``POST /predict`` is stateless: readings in, prediction out, nothing stored.
- ``/units/...`` endpoints store readings and predictions in PostgreSQL (``DATABASE_URL`` from
  the environment or ``.env``; schema created by ``alembic upgrade head``).
- If the model or the database is unavailable, the service still starts. The endpoints that
  need the missing part answer 503 with a clear message, instead of the process crashing.
"""

import logging
import os
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from predmaint.db import repository
from predmaint.db.settings import get_database_url
from predmaint.serving.predictor import InvalidHistoryError
from predmaint.serving.registry import LoadedModel
from predmaint.serving.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ReadingsBatch,
    ReadingsStored,
    StoredPrediction,
    StoredReading,
    UnitOverview,
    UnitPrediction,
    UnitSummary,
)

logger = logging.getLogger(__name__)

ModelLoader = Callable[[], LoadedModel]
UnitId = Annotated[int, Path(ge=1, description="Engine identifier")]


def load_from_registry() -> LoadedModel:
    """Default loader: the champion from the MLflow registry."""
    from predmaint.serving.registry import load_champion_info
    from predmaint.tracking import TRACKING_URI

    return load_champion_info(tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", TRACKING_URI))


def get_model(request: Request) -> LoadedModel:
    """FastAPI dependency: the loaded model, or 503 if none. Tests override this."""
    model: LoadedModel | None = request.app.state.model
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No model loaded: {request.app.state.load_error}",
        )
    return model


def get_session(request: Request) -> Iterator[Session]:
    """FastAPI dependency: a database session per request, or 503 if no database is set up."""
    factory: sessionmaker | None = request.app.state.session_factory
    if factory is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not configured: set DATABASE_URL (see .env.example)",
        )
    session = factory()
    try:
        yield session
    finally:
        session.close()


ModelDep = Annotated[LoadedModel, Depends(get_model)]
SessionDep = Annotated[Session, Depends(get_session)]


def get_repository(session: SessionDep) -> repository.Repository:
    """FastAPI dependency: database operations for one request. Unit tests override it."""
    return repository.Repository(session)


RepoDep = Annotated[repository.Repository, Depends(get_repository)]


def create_app(
    load_model: ModelLoader = load_from_registry, database_url: str | None = None
) -> FastAPI:
    """Build the app.

    Args:
        load_model: Runs once at startup; tests pass a fake loader.
        database_url: SQLAlchemy URL for PostgreSQL. None disables the ``/units`` endpoints
            (503); the stateless ``/predict`` still works.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model, app.state.load_error = None, None
        app.state.engine, app.state.session_factory = None, None
        try:
            app.state.model = load_model()
            logger.info("Loaded %s version %s", app.state.model.name, app.state.model.version)
        except Exception as exc:  # keep serving; endpoints that need the model answer 503
            app.state.load_error = f"{type(exc).__name__}: {exc}"
            logger.error("Could not load model: %s", app.state.load_error)
        if database_url:
            # pool_pre_ping replaces connections the database has closed in the meantime.
            app.state.engine = create_engine(
                database_url, pool_pre_ping=True, connect_args={"connect_timeout": 5}
            )
            app.state.session_factory = sessionmaker(app.state.engine, expire_on_commit=False)
        yield
        if app.state.engine is not None:
            app.state.engine.dispose()

    app = FastAPI(
        title="Predictive maintenance: RUL service",
        description="Remaining useful life of a turbofan engine (C-MAPSS FD001) with a 90% "
        "prediction interval. Either send readings to the stateless /predict, or store them "
        "per unit and ask for a prediction from the stored history.",
        version="1.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(OperationalError)
    def database_unavailable(request: Request, exc: OperationalError) -> JSONResponse:
        """A configured but unreachable database is a 503, not an opaque 500."""
        logger.error("Database error: %s", exc)
        return JSONResponse(
            {"detail": "Database unavailable; check that PostgreSQL is running"},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    def database_status(request: Request) -> str:
        if request.app.state.engine is None:
            return "not configured"
        try:
            with request.app.state.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception:
            return "unavailable"
        return "ok"

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    def health(request: Request) -> JSONResponse:
        """``ok`` (200) when a model is loaded, otherwise ``unavailable`` (503).

        The database status is reported too, but does not change the status code: the
        stateless ``/predict`` works without it.
        """
        model: LoadedModel | None = request.app.state.model
        body = HealthResponse(
            status="ok" if model else "unavailable",
            model_name=model.name if model else None,
            model_version=model.version if model else None,
            database=database_status(request),
        )
        code = status.HTTP_200_OK if model else status.HTTP_503_SERVICE_UNAVAILABLE
        return JSONResponse(body.model_dump(), status_code=code)

    @app.post(
        "/predict",
        response_model=PredictResponse,
        responses={422: {"description": "Invalid readings"}, 503: {"description": "No model"}},
        tags=["stateless"],
    )
    def predict(body: PredictRequest, model: ModelDep) -> PredictResponse:
        """Predict RUL at the last reading; nothing is stored. A sync ``def``: FastAPI runs it
        in a worker thread, so CPU-bound prediction does not block the server."""
        history = pd.DataFrame([r.model_dump() for r in body.readings])
        history.insert(0, "unit", body.unit_id)
        result = _predict(model, history)
        return PredictResponse(
            unit_id=body.unit_id,
            rul=result.rul,
            lower=result.lower,
            upper=result.upper,
            n_cycles_seen=result.n_cycles_seen,
            model_name=model.name,
            model_version=model.version,
        )

    @app.post(
        "/units/{unit_id}/readings",
        response_model=ReadingsStored,
        status_code=status.HTTP_201_CREATED,
        responses={409: {"description": "Cycle already stored"}, 503: {"description": "No DB"}},
        tags=["units"],
    )
    def store_readings(unit_id: UnitId, body: ReadingsBatch, repo: RepoDep) -> ReadingsStored:
        """Store readings for a unit, all or nothing. 409 if any cycle is already stored."""
        try:
            n = repo.add_readings(unit_id, [r.model_dump() for r in body.readings])
        except repository.DuplicateCyclesError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        total, last_cycle = repo.count_readings(unit_id)
        return ReadingsStored(unit_id=unit_id, n_stored=n, n_total=total, last_cycle=last_cycle)

    @app.get(
        "/units/{unit_id}",
        response_model=UnitSummary,
        responses={404: {"description": "Unknown unit"}, 503: {"description": "No DB"}},
        tags=["units"],
    )
    def unit_status(unit_id: UnitId, repo: RepoDep) -> UnitSummary:
        """How many readings and predictions are stored for the unit (used to resume)."""
        if not repo.unit_known(unit_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"unit {unit_id} is unknown")
        return UnitSummary(**repo.unit_summary(unit_id))

    @app.get(
        "/units/{unit_id}/prediction",
        response_model=UnitPrediction,
        responses={404: {"description": "Unknown unit"}, 503: {"description": "No model/DB"}},
        tags=["units"],
    )
    def predict_unit(unit_id: UnitId, model: ModelDep, repo: RepoDep) -> UnitPrediction:
        """Predict from all stored readings of the unit, store the prediction and return it."""
        history = repo.get_history(unit_id)
        if history.empty:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, detail=f"unit {unit_id} has no stored readings"
            )
        result = _predict(model, history)
        row = repo.add_prediction(
            unit_id=unit_id,
            cycle=int(history["cycle"].iloc[-1]),
            rul=result.rul,
            lower=result.lower,
            upper=result.upper,
            model_name=model.name,
            model_version=model.version,
        )
        stored = StoredPrediction.model_validate(row)
        return UnitPrediction(**stored.model_dump(), n_cycles_seen=result.n_cycles_seen)

    @app.get(
        "/units/{unit_id}/predictions",
        response_model=list[StoredPrediction],
        responses={404: {"description": "Unknown unit"}, 503: {"description": "No DB"}},
        tags=["units"],
    )
    def prediction_history(
        unit_id: UnitId,
        repo: RepoDep,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    ) -> list[StoredPrediction]:
        """The unit's stored predictions, newest first."""
        if not repo.unit_known(unit_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"unit {unit_id} is unknown")
        rows = repo.list_predictions(unit_id, limit)
        return [StoredPrediction.model_validate(r) for r in rows]

    @app.get(
        "/units",
        response_model=list[UnitOverview],
        responses={503: {"description": "No DB"}},
        tags=["units"],
    )
    def list_units(repo: RepoDep) -> list[UnitOverview]:
        """Every unit with stored readings, its last cycle and its latest stored prediction."""
        return [UnitOverview(**u) for u in repo.list_units()]

    @app.get(
        "/units/{unit_id}/readings",
        response_model=list[StoredReading],
        responses={404: {"description": "Unknown unit"}, 503: {"description": "No DB"}},
        tags=["units"],
    )
    def list_readings(
        unit_id: UnitId,
        repo: RepoDep,
        limit: Annotated[
            int | None, Query(ge=1, le=10_000, description="Only the most recent N readings")
        ] = None,
    ) -> list[StoredReading]:
        """The unit's stored readings in cycle order (all, or the most recent ``limit``)."""
        rows = repo.get_readings(unit_id, limit)
        if not rows:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, detail=f"unit {unit_id} has no stored readings"
            )
        return [StoredReading.model_validate(r) for r in rows]

    return app


def _predict(model: LoadedModel, history: pd.DataFrame):
    try:
        return model.predictor.predict_engine(history)
    except InvalidHistoryError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


app = create_app(database_url=get_database_url())
