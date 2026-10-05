from __future__ import annotations

import hmac
import json
import logging
import re
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

import httpx
from fastapi import FastAPI, File, Form, Header, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from clausus.assemble import assemble
from clausus.config import Settings
from clausus.detect import detect
from clausus.errors import ClaususError
from clausus.final_check import final_check
from clausus.guardrail import GuardrailRequest, GuardrailStore
from clausus.hear import EndpointHearClient, HearClient
from clausus.llm import ApertusClient, ChatClient, Recorded
from clausus.models import Span, SpanType
from clausus.receipt import Outgoing, PlaceholderInfo, Receipt, write_receipt
from clausus.router import Routes, route
from clausus.see import IMAGE_TYPES, read_image
from clausus.tasks.protocol import make_protocol
from clausus.vault import Vault

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("clausus.app")

_CLIENT_ERRORS = {"unsupported_image": 415, "hear_bad_audio": 422}
_RUN_ID = re.compile(r"(?:redact|meeting|guardrail)-[0-9a-f]{12}")


class KnownEntity(BaseModel):
    text: str
    type: SpanType = "PERSON"


class RedactRequest(BaseModel):
    text: str = Field(min_length=1)
    known_entities: list[KnownEntity] = Field(default_factory=list)


class PlaygroundRequest(BaseModel):
    text: str = Field(min_length=1)
    instruction: str = ""


_TOKEN = re.compile(r"\[[A-Z]+_\d+\]")
_ANY_PLACEHOLDER = re.compile(r"\[[A-ZÄÖÜ][A-ZÄÖÜ_ ]+(?:_\d+)?\]")
_PAGE = Path(__file__).parent / "static" / "playground.html"
_DEFAULT_INSTRUCTION = ("Antworte hilfreich und knapp auf die Nachricht. Platzhalter in eckigen Klammern "
                        "wie [PERSON_1] stehen für vertrauliche Angaben: übernimm sie wörtlich und "
                        "erfinde keine neuen.")


def split_original(original: str, masked: str) -> list[dict[str, str | None]]:
    """Align the masked text back to what was really said, so the playground can show
    which words became which placeholder ("Brunner" stays "Brunner", not the canonical name).
    The literal text between placeholders is identical in both; the gaps are the originals."""
    parts = _TOKEN.split(masked)
    tokens = _TOKEN.findall(masked)
    segs: list[dict[str, str | None]] = []
    pos = 0
    if not original.startswith(parts[0]):
        return [{"text": original, "token": None}]
    if parts[0]:
        segs.append({"text": parts[0], "token": None})
    pos = len(parts[0])
    for i, token in enumerate(tokens):
        literal = parts[i + 1]
        if i == len(tokens) - 1 and not literal:
            end = len(original)
        elif not literal:
            end = pos  # adjacent placeholders: the next one takes the shared text
        else:
            end = original.find(literal, pos + 1) if i < len(tokens) - 1 else original.rfind(literal)
            if end < pos:
                return [{"text": original, "token": None}]
        if end > pos:
            segs.append({"text": original[pos:end], "token": token})
        if literal:
            segs.append({"text": literal, "token": None})
        pos = end + len(literal)
    if "".join(str(s["text"]) for s in segs) != original:
        return [{"text": original, "token": None}]
    return segs


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
               hear_http: httpx.Client | None = None,
               guardrail_store: GuardrailStore | None = None) -> FastAPI:
    """App factory. Without arguments it reads the environment and fails loudly
    if configuration is missing (run with `uvicorn --factory clausus.app:create_app`)."""
    settings = settings or Settings.from_env()
    apertus = ApertusClient(settings, http_client)
    hear: HearClient | EndpointHearClient = (
        EndpointHearClient(settings.stt_base_url, settings.stt_api_key, settings.stt_model, hear_http)
        if settings.stt_base_url else HearClient(settings.hear_url, hear_http))
    store = guardrail_store if guardrail_store is not None else GuardrailStore()
    external = (ChatClient(settings.external_base_url, settings.external_api_key,
                           settings.external_model or settings.llm_name, "external", external_http)
                if settings.external_configured else None)
    routes = Routes.load(settings.routes_path) if settings.routes_path else Routes.default()
    app = FastAPI(title="Clausus", version="0.1.0")

    def failure(e: ClaususError) -> JSONResponse:
        """Input problems are 4xx, everything else 503. Either way nothing went further."""
        status = _CLIENT_ERRORS.get(e.code, 503)
        log.warning("failed closed: %s (%d)", e.code, status)
        return JSONResponse(status_code=status, content={"status": "error" if status < 500 else "down",
                                                         "error": e.code, "detail": e.message})

    def mask(text: str, known: list[KnownEntity], outgoing: list[Outgoing]) -> tuple[Vault, list[Span], str]:
        vault = Vault()
        vault.preload([(k.text, k.type) for k in known])
        spans = detect(text, Recorded(apertus, outgoing, "original"), person_model=settings.fast_model)
        return vault, spans, vault.pseudonymize(text, spans)

    models_cache: dict[str, Any] = {"at": 0.0, "models": None}

    @app.get("/live")
    def live() -> JSONResponse:
        """Liveness only, calls nothing. Used by the Docker healthcheck so that polling
        never spends the model endpoint's rate limit."""
        return JSONResponse(content={"status": "alive"})

    @app.get("/health")
    def health() -> JSONResponse:
        try:
            if models_cache["models"] is None or time.monotonic() - models_cache["at"] > 60:
                models_cache["models"] = apertus.list_models()
                models_cache["at"] = time.monotonic()
            models = models_cache["models"]
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

    @app.get("/receipts/{run_id}")
    def get_receipt(run_id: str) -> JSONResponse:
        """Read a receipt back. Ids are checked against a strict pattern so a request can
        never reach a file outside the receipts folder."""
        if not _RUN_ID.fullmatch(run_id):
            return JSONResponse(status_code=400, content={"status": "error", "error": "bad_id"})
        path = Path(settings.runs_dir) / f"{run_id}.json"
        if not path.is_file():
            return JSONResponse(status_code=404, content={"status": "error", "error": "not_found"})
        return JSONResponse(content=json.loads(path.read_text(encoding="utf-8")))

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

    def run_meeting(text: str, known: list[KnownEntity], outgoing: list[Outgoing],
                    notes: list[str]) -> dict[str, Any]:
        """Shared by /meeting and /meeting/upload. The protocol model only ever sees the
        masked text; external only after a clean final check. Raises ClaususError."""
        vault, spans, masked = mask(text, known, outgoing)
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

        result = assemble(json.dumps(protocol, ensure_ascii=False), vault)
        run_id = f"meeting-{uuid.uuid4().hex[:12]}"
        write_receipt(Receipt(run_id=run_id, model=settings.llm_name, destination=destination,
                              placeholders=_placeholders(vault, spans), outgoing=outgoing,
                              notes=notes), settings.runs_dir)
        log.info("meeting ok: %d spans, destination %s, run %s", len(spans), destination, run_id)
        return {
            "protocol": _restore_deep(protocol, vault),
            "destination": destination,
            "seen_by_model": masked,
            "warnings": {"invented": result.invented, "missing": result.missing},
            "notes": notes,
            "receipt_id": run_id,
        }

    @app.post("/meeting")
    def meeting(req: MeetingRequest) -> JSONResponse:
        """Transcript in, protocol with real names out."""
        try:
            return JSONResponse(content=run_meeting(req.transcript, req.known_entities, [], []))
        except ClaususError as e:
            return failure(e)

    @app.post("/meeting/upload")
    async def meeting_upload(
        audio: Annotated[UploadFile | None, File()] = None,
        images: Annotated[list[UploadFile] | None, File()] = None,
        transcript: Annotated[str, Form()] = "",
        known_entities: Annotated[str, Form()] = "[]",
    ) -> JSONResponse:
        """Audio and/or slides and/or text in one call. Hearing runs first and locally:
        if it fails, no slide is sent anywhere either."""
        images = images or []
        if audio is None and not images and not transcript.strip():
            return JSONResponse(status_code=400, content={"status": "error", "error": "empty",
                                                          "detail": "Send audio, images or a transcript."})
        try:
            known = [KnownEntity(**k) for k in json.loads(known_entities or "[]")]
        except (ValueError, TypeError) as e:
            return JSONResponse(status_code=400, content={"status": "error", "error": "bad_known_entities",
                                                          "detail": str(e)})
        outgoing: list[Outgoing] = []
        notes: list[str] = []
        try:
            for img in images:  # validate every image before anything is sent
                if (img.content_type or "") not in IMAGE_TYPES:
                    raise ClaususError("unsupported_image", f"Unsupported image type: {img.content_type}.")
            spoken = ""
            if audio is not None:
                heard = hear.transcribe(await audio.read(), audio.filename or "audio")
                spoken = heard.text
                if heard.local:
                    notes.append(f"audio transcribed locally ({heard.duration_s:.0f} s on {heard.device})")
                else:  # audio carries originals: listed like any other call that leaves Clausus
                    outgoing.append(Outgoing(step="hear", destination=heard.destination, kind="original",
                                             model=heard.model))
            slides = [read_image(await img.read(), img.content_type or "",
                                 Recorded(apertus, outgoing, "original")) for img in images]
            parts = [p for p in (transcript.strip(), spoken) if p]
            if slides:
                parts.append("Folien:\n" + "\n\n".join(slides))
            result = run_meeting("\n\n".join(parts), known, outgoing, notes)
        except ClaususError as e:
            return failure(e)
        return JSONResponse(content={"transcript": spoken, "slides": slides, **result})

    @app.get("/", response_class=HTMLResponse)
    def playground_page() -> HTMLResponse:
        return HTMLResponse(_PAGE.read_text(encoding="utf-8"))

    @app.post("/playground/run")
    def playground_run(req: PlaygroundRequest) -> JSONResponse:
        """The three stages in one call: what you said, what the model sees, what comes back.
        The vault table goes to the local page only, never to a model."""
        outgoing: list[Outgoing] = []
        try:
            vault, spans, masked = mask(req.text, [], outgoing)
            answer_masked = Recorded(apertus, outgoing, "masked").chat(
                [{"role": "system", "content": req.instruction.strip() or _DEFAULT_INSTRUCTION},
                 {"role": "user", "content": masked}], step="answer")
        except ClaususError as e:
            return failure(e)
        invented = sorted({t for t in _ANY_PLACEHOLDER.findall(answer_masked) if not vault.knows(t)})
        placeholders = _placeholders(vault, spans)
        sources = {p.token: p.sources for p in placeholders}
        run_id = f"redact-{uuid.uuid4().hex[:12]}"
        write_receipt(Receipt(run_id=run_id, model=settings.llm_name, placeholders=placeholders,
                              outgoing=outgoing, notes=["playground"]), settings.runs_dir)
        log.info("playground ok: %d spans, run %s", len(spans), run_id)
        return JSONResponse(content={
            "original_segments": split_original(req.text, masked),
            "masked": masked,
            "answer_masked": answer_masked,
            "answer": vault.restore(answer_masked),
            "invented": invented,
            "vault": [{"token": t, "type": ty, "original": o, "sources": sources.get(t, [])}
                      for t, ty, o in vault.mapping()],
            "outgoing": [o.model_dump() for o in outgoing],
            "receipt_id": run_id,
        })

    @app.post("/beta/litellm_basic_guardrail_api")
    def litellm_guardrail(req: GuardrailRequest,
                          x_api_key: Annotated[str | None, Header()] = None,
                          authorization: Annotated[str | None, Header()] = None) -> JSONResponse:
        """LiteLLM calls this before (request) and after (response) every LLM call.
        Request: mask, keep the vault in memory under litellm_call_id. Response: restore,
        then forget the vault. If checking fails, the call is blocked, never let through."""
        # LiteLLM sends `api_key` from its guardrail config as the x-api-key header
        # (generic_guardrail_api.py, read 02.10.2026). Bearer is accepted for other callers.
        given = x_api_key or (authorization or "").removeprefix("Bearer ")
        if settings.guardrail_key and not hmac.compare_digest(given.encode(), settings.guardrail_key.encode()):
            return JSONResponse(status_code=401, content={"action": "BLOCKED", "blocked_reason": "unauthorized"})
        call_id = req.litellm_call_id or ""
        if req.input_type == "response":
            vault = store.pop(call_id) if call_id else None
            if vault is None:
                return JSONResponse(content={"action": "NONE"})
            return JSONResponse(content={"action": "GUARDRAIL_INTERVENED",
                                         "texts": [vault.restore(t) for t in req.texts]})
        if req.images:
            return JSONResponse(content={"action": "BLOCKED",
                                         "blocked_reason": "Clausus does not check images through LiteLLM yet."})
        outgoing: list[Outgoing] = []
        vault = Vault()
        spans: list[Span] = []
        masked: list[str] = []
        try:
            for text in req.texts:
                found = detect(text, Recorded(apertus, outgoing, "original"), person_model=settings.fast_model)
                spans += found
                masked.append(vault.pseudonymize(text, found))
        except ClaususError as e:
            log.warning("guardrail failed closed: %s", e.code)
            return JSONResponse(content={"action": "BLOCKED",
                                         "blocked_reason": f"Clausus could not check the content ({e.code})."})
        if masked == req.texts:
            return JSONResponse(content={"action": "NONE"})
        if call_id:
            store.put(call_id, vault)
        write_receipt(Receipt(run_id=f"guardrail-{uuid.uuid4().hex[:12]}", model=settings.llm_name,
                              destination="litellm", placeholders=_placeholders(vault, spans),
                              outgoing=outgoing, notes=[f"litellm_call_id {call_id or 'none'}"]),
                      settings.runs_dir)
        return JSONResponse(content={"action": "GUARDRAIL_INTERVENED", "texts": masked})

    return app
