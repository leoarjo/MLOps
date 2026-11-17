"""API de inferência do churn (Encontro 12).

Rodar::

    uv run uvicorn churn.api:app --host 0.0.0.0 --port 8000
    # documentação interativa em http://localhost:8000/docs

Rotas:

``GET  /health``         o container está de pé e o modelo carregou?
``GET  /model``          qual champion está servindo (versão, run, linhagem)
``POST /predict``        um cliente  -> probabilidade de churn
``POST /predict/batch``  N clientes  -> N probabilidades

O modelo é carregado UMA vez, no ``lifespan`` — nunca por requisição. Se o
bundle não existir, a API não sobe: um container que responde 200 no
``/health`` sem modelo é pior do que um que cai com a mensagem certa no log.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from churn import __version__
from churn.config import settings
from churn.serving import CustomerRecord, ModelBundle, Prediction

logger = logging.getLogger(__name__)

MAX_BATCH = 10_000


class BatchRequest(BaseModel):
    records: list[CustomerRecord] = Field(min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    model_version: str | None
    predictions: list[Prediction]


def create_app(
    model_dir: Path | None = None, threshold: float | None = None
) -> FastAPI:
    """Fábrica da aplicação — os testes passam um bundle temporário."""
    model_dir = model_dir or settings.model_dir
    threshold = settings.decision_threshold if threshold is None else threshold

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bundle = ModelBundle.load(model_dir)
        app.state.threshold = threshold
        yield

    app = FastAPI(
        title="Churn API",
        version=__version__,
        description="Previsão de churn servindo o `models:/churn@champion`.",
        lifespan=lifespan,
    )

    def bundle(request: Request) -> ModelBundle:
        return request.app.state.bundle

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/model")
    def model_info(request: Request) -> dict[str, Any]:
        return {
            **bundle(request).metadata,
            "decision_threshold": request.app.state.threshold,
        }

    @app.post("/predict")
    def predict(record: CustomerRecord, request: Request) -> Prediction:
        return bundle(request).predict([record], request.app.state.threshold)[0]

    @app.post("/predict/batch")
    def predict_batch(body: BatchRequest, request: Request) -> BatchResponse:
        b = bundle(request)
        return BatchResponse(
            model_version=b.metadata.get("version"),
            predictions=b.predict(body.records, request.app.state.threshold),
        )

    return app


app = create_app()
