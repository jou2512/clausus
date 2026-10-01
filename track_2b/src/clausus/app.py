from __future__ import annotations

import logging
import uuid
from collections import Counter

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from clausus.config import Settings
from clausus.detect import detect
from clausus.errors import ClaususError
from clausus.llm import ApertusClient
from clausus.models import SpanType
from clausus.receipt import PlaceholderInfo, Receipt, write_receipt
from clausus.vault import Vault

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("clausus.app")


class KnownEntity(BaseModel):
    text: str
    type: SpanType = "PERSON"


class RedactRequest(BaseModel):
    text: str = Field(min_length=1)
    known_entities: list[KnownEntity] = Field(default_factory=list)


def create_app(settings: Settings | None = None, http_client: httpx.Client | None = None) -> FastAPI:
    """App factory. Without arguments it reads the environment and fails loudly
    if configuration is missing (run with `uvicorn --factory clausus.app:create_app`)."""
    settings = settings or Settings.from_env()
    apertus = ApertusClient(settings, http_client)
    app = FastAPI(title="Clausus", version="0.1.0")

    def failure(e: ClaususError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"status": "down", "error": e.code,
                                                      "detail": e.message})

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

    @app.post("/redact")
    def redact(req: RedactRequest) -> JSONResponse:
        """Find what is confidential and mask it. Nothing leaves the house here; the
        receipt records the placeholders, never the originals."""
        vault = Vault()
        vault.preload([(k.text, k.type) for k in req.known_entities])
        try:
            spans = detect(req.text, apertus, person_model=settings.fast_model)
        except ClaususError as e:
            log.warning("redact failed closed: %s", e.code)
            return failure(e)
        masked = vault.pseudonymize(req.text, spans)

        sources: dict[str, Counter[str]] = {}
        for s in spans:
            sources.setdefault(vault.token_for(s.text, s.type), Counter())[s.source] += 1
        placeholders = [
            PlaceholderInfo(token=t, type=typ, sources=sorted(sources.get(t, {"known": 1})))
            for t, typ in vault.tokens
        ]
        run_id = f"redact-{uuid.uuid4().hex[:12]}"
        write_receipt(Receipt(run_id=run_id, model=settings.llm_name, placeholders=placeholders,
                              outgoing=[]), settings.runs_dir)
        log.info("redact ok: %d spans, %d placeholders, run %s", len(spans), len(placeholders), run_id)
        return JSONResponse(content={
            "masked_text": masked,
            "placeholders": [{"token": p.token, "type": p.type} for p in placeholders],
            "receipt_id": run_id,
        })

    return app
