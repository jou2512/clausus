from __future__ import annotations

import json
import logging
import uuid
from collections import Counter
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from clausus.assemble import assemble
from clausus.config import Settings
from clausus.detect import detect
from clausus.errors import ClaususError
from clausus.final_check import final_check
from clausus.hear import HearClient
from clausus.llm import ApertusClient, ChatClient, Recorded
from clausus.models import Span, SpanType
from clausus.receipt import Outgoing, PlaceholderInfo, Receipt, write_receipt
from clausus.router import Routes, route
from clausus.tasks.protocol import make_protocol
from clausus.vault import Vault

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("clausus.app")


class KnownEntity(BaseModel):
    text: str
    type: SpanType = "PERSON"


class RedactRequest(BaseModel):
    text: str = Field(min_length=1)
    known_entities: list[KnownEntity] = Field(default_factory=list)


class MeetingRequest(BaseModel):
    transcript: str = Field(min_length=1)
    known_entities: list[KnownEntity] = Field(default_factory=list)


def _restore_deep(value: Any, vault: Vault) -> Any:
    if isinstance(value, str):
        return vault.restore(value)
    if isinstance(value, list):
        return [_restore_deep(v, vault) for v in value]
    if isinstance(value, dict):
        return {k: _restore_deep(v, vault) for k, v in value.items()}
    return value


def _placeholders(vault: Vault, spans: list[Span]) -> list[PlaceholderInfo]:
    sources: dict[str, Counter[str]] = {}
    for s in spans:
        sources.setdefault(vault.token_for(s.text, s.type), Counter())[s.source] += 1
    return [PlaceholderInfo(token=t, type=typ, sources=sorted(sources.get(t, {"known": 1})))
            for t, typ in vault.tokens]


def create_app(settings: Settings | None = None, http_client: httpx.Client | None = None,
               external_http: httpx.Client | None = None,
               hear_http: httpx.Client | None = None) -> FastAPI:
    """App factory. Without arguments it reads the environment and fails loudly
    if configuration is missing (run with `uvicorn --factory clausus.app:create_app`)."""
    settings = settings or Settings.from_env()
    apertus = ApertusClient(settings, http_client)
    hear = HearClient(settings.hear_url, hear_http)
    external = (ChatClient(settings.external_base_url, settings.external_api_key,
                           settings.external_model or settings.llm_name, "external", external_http)
                if settings.external_configured else None)
    routes = Routes.load(settings.routes_path) if settings.routes_path else Routes.default()
    app = FastAPI(title="Clausus", version="0.1.0")

    def failure(e: ClaususError) -> JSONResponse:
        log.warning("failed closed: %s", e.code)
        return JSONResponse(status_code=503, content={"status": "down", "error": e.code,
                                                      "detail": e.message})

    def mask(text: str, known: list[KnownEntity], outgoing: list[Outgoing]) -> tuple[Vault, list[Span], str]:
        vault = Vault()
        vault.preload([(k.text, k.type) for k in known])
        spans = detect(text, Recorded(apertus, outgoing, "original"), person_model=settings.fast_model)
        return vault, spans, vault.pseudonymize(text, spans)

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
        hear_status = hear.health()
        return JSONResponse(content={
            "status": "ok" if available and hear_status == "ok" else "degraded",
            "apertus": "ok",
            "model": settings.llm_name,
            "model_available": available,
            "models": models,
            "hear": hear_status,
        })

    @app.post("/redact")
    def redact(req: RedactRequest) -> JSONResponse:
        """Find what is confidential and mask it."""
        outgoing: list[Outgoing] = []
        try:
            vault, spans, masked = mask(req.text, req.known_entities, outgoing)
        except ClaususError as e:
            return failure(e)
        placeholders = _placeholders(vault, spans)
        run_id = f"redact-{uuid.uuid4().hex[:12]}"
        write_receipt(Receipt(run_id=run_id, model=settings.llm_name, placeholders=placeholders,
                              outgoing=outgoing), settings.runs_dir)
        log.info("redact ok: %d spans, run %s", len(spans), run_id)
        return JSONResponse(content={
            "masked_text": masked,
            "placeholders": [{"token": p.token, "type": p.type} for p in placeholders],
            "receipt_id": run_id,
        })

    @app.post("/meeting")
    def meeting(req: MeetingRequest) -> JSONResponse:
        """Transcript in, protocol with real names out. The protocol model only ever sees
        the masked transcript; external only after a clean final check."""
        outgoing: list[Outgoing] = []
        notes: list[str] = []
        try:
            vault, spans, masked = mask(req.transcript, req.known_entities, outgoing)
            decision = route("protocol", routes, external is not None)
            notes.append(decision.reason)
            destination = decision.destination
            if destination == "external":
                check = final_check(masked, Recorded(apertus, outgoing, "masked"))
                if not check.clean:
                    destination = "local"
                    notes.append(f"final check found {len(check.leftovers)} leftover(s); kept local")
            target = external if destination == "external" and external else apertus
            protocol = make_protocol(masked, Recorded(target, outgoing, "masked"))
        except ClaususError as e:
            return failure(e)


        result = assemble(json.dumps(protocol, ensure_ascii=False), vault)
        restored = _restore_deep(protocol, vault)
        run_id = f"meeting-{uuid.uuid4().hex[:12]}"
        write_receipt(Receipt(run_id=run_id, model=settings.llm_name, destination=destination,
                              placeholders=_placeholders(vault, spans), outgoing=outgoing,
                              notes=notes), settings.runs_dir)
        log.info("meeting ok: %d spans, destination %s, run %s", len(spans), destination, run_id)
        return JSONResponse(content={
            "protocol": restored,
            "destination": destination,
            "seen_by_model": masked,
            "warnings": {"invented": result.invented, "missing": result.missing},
            "notes": notes,
            "receipt_id": run_id,
        })

    return app
