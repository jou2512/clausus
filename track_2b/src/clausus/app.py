from __future__ import annotations

import logging

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from clausus.config import Settings
from clausus.errors import ClaususError
from clausus.llm import ApertusClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def create_app(settings: Settings | None = None, http_client: httpx.Client | None = None) -> FastAPI:
    """App factory. Without arguments it reads the environment and fails loudly
    if configuration is missing (run with `uvicorn --factory clausus.app:create_app`)."""
    settings = settings or Settings.from_env()
    apertus = ApertusClient(settings, http_client)
    app = FastAPI(title="Clausus", version="0.1.0")

    @app.get("/health")
    def health() -> JSONResponse:
        try:
            models = apertus.list_models()
        except ClaususError as e:
            return JSONResponse(
                status_code=503,
                content={"status": "down", "apertus": e.code, "model": settings.llm_name,
                         "detail": e.message},
            )
        available = settings.llm_name in models
        return JSONResponse(
            content={
                "status": "ok" if available else "degraded",
                "apertus": "ok",
                "model": settings.llm_name,
                "model_available": available,
                "models": models,
            }
        )

    return app
