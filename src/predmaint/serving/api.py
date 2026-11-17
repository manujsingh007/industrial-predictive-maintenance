"""Stateless FastAPI service: one engine's readings in, RUL with a 90% interval out.

Run locally: ``uv run uvicorn predmaint.serving.api:app --reload``. The interactive docs are at
http://127.0.0.1:8000/docs.

- The champion model is loaded **once at startup** (FastAPI lifespan), from the MLflow
  registry at ``MLFLOW_TRACKING_URI`` (default: the project's ``mlflow.db``).
- If loading fails, the service still starts. ``/health`` and ``/predict`` then answer 503
  with a clear message, instead of the process crashing.
- The service is stateless: nothing is stored between requests.
"""

import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from predmaint.serving.predictor import InvalidHistoryError
from predmaint.serving.registry import LoadedModel
from predmaint.serving.schemas import HealthResponse, PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

ModelLoader = Callable[[], LoadedModel]


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


ModelDep = Annotated[LoadedModel, Depends(get_model)]


def create_app(load_model: ModelLoader = load_from_registry) -> FastAPI:
    """Build the app; ``load_model`` runs once at startup (tests pass a fake loader)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model, app.state.load_error = None, None
        try:
            app.state.model = load_model()
            logger.info("Loaded %s version %s", app.state.model.name, app.state.model.version)
        except Exception as exc:  # keep serving /health; /predict answers 503
            app.state.load_error = f"{type(exc).__name__}: {exc}"
            logger.error("Could not load model: %s", app.state.load_error)
        yield

    app = FastAPI(
        title="Predictive maintenance: RUL service",
        description="Remaining useful life of a turbofan engine (C-MAPSS FD001) with a 90% "
        "prediction interval. Send one engine's readings, oldest first.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> JSONResponse:
        """``ok`` (200) when a model is loaded, otherwise ``unavailable`` (503)."""
        model: LoadedModel | None = request.app.state.model
        if model is None:
            body = HealthResponse(status="unavailable")
            return JSONResponse(body.model_dump(), status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        body = HealthResponse(status="ok", model_name=model.name, model_version=model.version)
        return JSONResponse(body.model_dump())

    @app.post(
        "/predict",
        response_model=PredictResponse,
        responses={422: {"description": "Invalid readings"}, 503: {"description": "No model"}},
    )
    def predict(body: PredictRequest, model: ModelDep) -> PredictResponse:
        """Predict RUL at the last reading. A sync ``def``: FastAPI runs it in a worker thread,
        so CPU-bound prediction does not block the server."""
        history = pd.DataFrame([r.model_dump() for r in body.readings])
        history.insert(0, "unit", body.unit_id)
        try:
            result = model.predictor.predict_engine(history)
        except InvalidHistoryError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
        return PredictResponse(
            unit_id=body.unit_id,
            rul=result.rul,
            lower=result.lower,
            upper=result.upper,
            n_cycles_seen=result.n_cycles_seen,
            model_name=model.name,
            model_version=model.version,
        )

    return app


app = create_app()
