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
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, File, Form, Header, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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
from clausus.sessions import SessionStore, load_or_create_key
from clausus.tasks.protocol import make_protocol
from clausus.vault import Vault

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("clausus.app")

_CLIENT_ERRORS = {"unsupported_image": 415, "hear_bad_audio": 422}
_RUN_ID = re.compile(r"(?:redact|meeting|guardrail|chat)-[0-9a-f]{12}")


class KnownEntity(BaseModel):
    text: str
    type: SpanType = "PERSON"


class RedactRequest(BaseModel):
    text: str = Field(min_length=1)
    known_entities: list[KnownEntity] = Field(default_factory=list)


class AnswerAgainRequest(BaseModel):
    run_id: str
    instruction: str = ""
    answer_model: str = ""


class ChatMessageRequest(BaseModel):
    text: str = Field(min_length=1)
    answer_model: str = ""
    known_entities: list[KnownEntity] = Field(default_factory=list)


class PlaygroundRequest(BaseModel):
    text: str = Field(min_length=1)
    instruction: str = ""
    answer_model: str = ""
    # Names the user already knows are confidential (attendees, clients). Masked even when
    # detection misses them, and every later mention with them (the vault's sweep).
    known_entities: list[KnownEntity] = Field(default_factory=list)


_TOKEN = re.compile(r"\[[A-Z]+_\d+\]")
# Anything a model might write as a placeholder, also "[Dein Name]": starts with a letter,
# at least two characters, so Markdown checkboxes "[x]" and footnotes "[1]" do not count.
_ANY_PLACEHOLDER = re.compile(r"\[[A-Za-zÄÖÜäöü][A-Za-zÄÖÜäöü_ .\-]{1,40}(?:_\d+)?\]")
_PAGE = Path(__file__).parent / "static" / "playground.html"
_SAMPLE_SLIDE = Path(__file__).parent / "static" / "sample-slide.png"
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
    # Recent playground runs: vault + masked text, in memory only, so the answer can be asked
    # again with another model without detecting again. Lost on restart, expires after 30 min.
    recent: dict[str, tuple[Vault, str, dict[str, list[str]], float]] = {}
    vault_dir = settings.vault_dir or str(Path(settings.runs_dir) / "vault")
    store_box: list[SessionStore] = []

    def sessions() -> SessionStore:
        """Created on first use, so starting Clausus leaves no files behind."""
        if not store_box:
            store_box.append(SessionStore(vault_dir, load_or_create_key(settings.vault_key, vault_dir)))
        return store_box[0]
    recent_ttl = 1800.0
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

    @app.get("/receipts")
    def list_receipts() -> JSONResponse:
        """Newest first, at most 200. Receipts never hold original text, so listing them is safe."""
        items = []
        for path in Path(settings.runs_dir).glob("*.json"):
            if not _RUN_ID.fullmatch(path.stem):
                continue
            try:
                r = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            calls = r.get("outgoing", [])
            items.append({
                "run_id": r.get("run_id", path.stem), "created_at": r.get("created_at", ""),
                "model": r.get("model", ""), "destination": r.get("destination", ""),
                "placeholders": len(r.get("placeholders", [])), "calls": len(calls),
                "original_calls": sum(1 for c in calls if c.get("kind") == "original"),
                "masked_calls": sum(1 for c in calls if c.get("kind") == "masked"),
                "ms": sum(c.get("ms") or 0 for c in calls), "notes": r.get("notes", []),
            })
        items.sort(key=lambda i: i["created_at"], reverse=True)
        return JSONResponse(content={"receipts": items[:200], "total": len(items)})

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

    # Detection stays as measured (benchmark); only the answering model is a choice.
    answer_models = list(dict.fromkeys([settings.llm_name, settings.fast_model]))

    @app.get("/playground/config")
    def playground_config() -> JSONResponse:
        host = lambda url: urlparse(url).hostname or url
        return JSONResponse(content={
            "answer_models": answer_models,
            "detection": list(dict.fromkeys([settings.llm_name, settings.fast_model])),
            "hear": "endpoint" if settings.stt_base_url else "local",
            # What is configured, never a key: only whether one is set.
            "endpoints": {
                "detection": {"host": host(settings.llm_base_url), "models": list(dict.fromkeys(
                    [settings.llm_name, settings.fast_model])), "key_set": bool(settings.llm_api_key)},
                "answer": {"host": host(settings.llm_base_url), "models": answer_models},
                "hear": ({"mode": "endpoint", "host": host(settings.stt_base_url), "model": settings.stt_model,
                          "key_set": bool(settings.stt_api_key)} if settings.stt_base_url
                         else {"mode": "local", "host": host(settings.hear_url)}),
                "external": {"configured": settings.external_configured,
                             "host": host(settings.external_base_url) if settings.external_base_url else "",
                             "model": settings.external_model},
                "guardrail": {"path": "/beta/litellm_basic_guardrail_api", "key_set": bool(settings.guardrail_key)},
                "receipts_dir": settings.runs_dir,
            },
        })

    @app.get("/playground/sample-slide.png")
    def sample_slide() -> FileResponse:
        return FileResponse(_SAMPLE_SLIDE, media_type="image/png")

    def bad_model(name: str) -> JSONResponse | None:
        if name and name not in answer_models:
            return JSONResponse(status_code=400, content={
                "status": "error", "error": "unknown_model",
                "detail": f"Answer model must be one of {', '.join(answer_models)}."})
        return None

    def playground(text: str, instruction: str, answer_model: str, outgoing: list[Outgoing],
                   extra: dict[str, Any], started: float | None = None,
                   known: list[KnownEntity] | None = None) -> JSONResponse:
        """The three stages in one call: what you said, what the answering model sees, what comes
        back. The vault table goes to the local page only, never to a model. Raises ClaususError."""
        started = started if started is not None else time.perf_counter()
        t0 = time.perf_counter()
        vault, spans, masked = mask(text, known or [], outgoing)
        detect_ms = round((time.perf_counter() - t0) * 1000)
        answered = answer(vault, masked, instruction, answer_model, outgoing)
        placeholders = _placeholders(vault, spans)
        sources = {pl.token: pl.sources for pl in placeholders}
        run_id = f"redact-{uuid.uuid4().hex[:12]}"
        write_receipt(Receipt(run_id=run_id, model=settings.llm_name, placeholders=placeholders,
                              outgoing=outgoing, notes=["playground", *extra.pop("notes", [])]),
                      settings.runs_dir)
        now = time.monotonic()
        for k in [k for k, v in recent.items() if now - v[3] > recent_ttl]:
            del recent[k]
        recent[run_id] = (vault, masked, sources, now)
        log.info("playground ok: %d spans, run %s", len(spans), run_id)
        timings = {"hear_ms": extra.pop("hear_ms", 0), "see_ms": extra.pop("see_ms", 0),
                   "detect_ms": detect_ms,
                   "answer_ms": next((o.ms or 0 for o in reversed(outgoing) if o.step == "answer"), 0),
                   "total_ms": round((time.perf_counter() - started) * 1000)}
        return JSONResponse(content={
            **extra,
            "timings": timings,
            "original_segments": split_original(text, masked),
            "masked": masked,
            **answered,
            "vault": [{"token": t, "type": ty, "original": o, "sources": sources.get(t, [])}
                      for t, ty, o in vault.mapping()],
            "outgoing": [o.model_dump() for o in outgoing],
            "receipt_id": run_id,
        })

    def answer(vault: Vault, masked: str, instruction: str, answer_model: str,
               outgoing: list[Outgoing]) -> dict[str, Any]:
        """The answering model gets the masked text only. Raises ClaususError."""
        answer_masked = Recorded(apertus, outgoing, "masked").chat(
            [{"role": "system", "content": instruction.strip() or _DEFAULT_INSTRUCTION},
             {"role": "user", "content": masked}], model=answer_model or None, step="answer")
        return {
            "answer_model": answer_model or settings.llm_name,
            "answer_masked": answer_masked,
            "answer": vault.restore(answer_masked),
            "invented": sorted({t for t in _ANY_PLACEHOLDER.findall(answer_masked) if not vault.knows(t)}),
            # Sent to the model but absent from its answer: the model dropped that person or fact.
            "missing": [t for t in dict.fromkeys(_TOKEN.findall(masked)) if t not in answer_masked],
        }

    @app.post("/playground/answer")
    def playground_answer(req: AnswerAgainRequest) -> JSONResponse:
        """Ask again, maybe with another model, on the stored masked text. No detection."""
        if (refused := bad_model(req.answer_model)) is not None:
            return refused
        item = recent.get(req.run_id)
        if item is None or time.monotonic() - item[3] > recent_ttl:
            recent.pop(req.run_id, None)
            return JSONResponse(status_code=404, content={
                "status": "error", "error": "run_expired",
                "detail": "This run is no longer in memory (30 min or a restart). Press Run again."})
        vault, masked, sources, _ = item
        outgoing: list[Outgoing] = []
        started = time.perf_counter()
        try:
            answered = answer(vault, masked, req.instruction, req.answer_model, outgoing)
        except ClaususError as e:
            return failure(e)
        run_id = f"redact-{uuid.uuid4().hex[:12]}"
        write_receipt(Receipt(run_id=run_id, model=settings.llm_name,
                              placeholders=_placeholders(vault, []), outgoing=outgoing,
                              notes=["playground answer again", f"masked text from {req.run_id}"]),
                      settings.runs_dir)
        recent[run_id] = (vault, masked, sources, time.monotonic())
        total = round((time.perf_counter() - started) * 1000)
        return JSONResponse(content={
            **answered,
            "timings": {"answer_ms": outgoing[-1].ms or 0, "total_ms": total},
            "outgoing": [o.model_dump() for o in outgoing],
            "receipt_id": run_id,
        })

    @app.post("/playground/run")
    def playground_run(req: PlaygroundRequest) -> JSONResponse:
        if (refused := bad_model(req.answer_model)) is not None:
            return refused
        try:
            return playground(req.text, req.instruction, req.answer_model, [], {},
                              known=req.known_entities)
        except ClaususError as e:
            return failure(e)

    @app.post("/playground/upload")
    async def playground_upload(
        audio: Annotated[UploadFile | None, File()] = None,
        images: Annotated[list[UploadFile] | None, File()] = None,
        text: Annotated[str, Form()] = "",
        instruction: Annotated[str, Form()] = "",
        answer_model: Annotated[str, Form()] = "",
        known_entities: Annotated[str, Form()] = "[]",
    ) -> JSONResponse:
        """Typed text, a recording and slides together. Hearing runs first; if it fails,
        no slide and no text is sent anywhere."""
        images = images or []
        if audio is None and not images and not text.strip():
            return JSONResponse(status_code=400, content={"status": "error", "error": "empty",
                                                          "detail": "Type text, record or add a file."})
        if (refused := bad_model(answer_model)) is not None:
            return refused
        try:
            known = [KnownEntity(**k) for k in json.loads(known_entities or "[]")]
        except (ValueError, TypeError) as e:
            return JSONResponse(status_code=400, content={"status": "error", "error": "bad_known_entities",
                                                          "detail": str(e)})
        outgoing: list[Outgoing] = []
        notes: list[str] = []
        started = time.perf_counter()
        hear_ms = see_ms = 0
        try:
            for img in images:  # validate every image before anything is sent
                if (img.content_type or "") not in IMAGE_TYPES:
                    raise ClaususError("unsupported_image", f"Unsupported image type: {img.content_type}.")
            spoken = ""
            if audio is not None:
                t0 = time.perf_counter()
                heard = hear.transcribe(await audio.read(), audio.filename or "audio")
                hear_ms = round((time.perf_counter() - t0) * 1000)
                spoken = heard.text
                if heard.local:
                    notes.append(f"audio transcribed locally ({heard.duration_s:.0f} s on {heard.device})")
                else:
                    outgoing.append(Outgoing(step="hear", destination=heard.destination, kind="original",
                                             model=heard.model))
            t0 = time.perf_counter()
            slides = [read_image(await img.read(), img.content_type or "",
                                 Recorded(apertus, outgoing, "original")) for img in images]
            see_ms = round((time.perf_counter() - t0) * 1000) if images else 0
            parts = [x for x in (text.strip(), spoken) if x]
            if slides:
                parts.append("Folien:\n" + "\n\n".join(slides))
            return playground("\n\n".join(parts), instruction, answer_model, outgoing,
                              {"transcript": spoken, "slides": slides, "notes": notes,
                               "hear_ms": hear_ms, "see_ms": see_ms}, started, known=known)
        except ClaususError as e:
            return failure(e)

    # --- chat: one encrypted vault per session ------------------------------------------------
    _CHAT_SYSTEM = ("Du bist ein hilfreicher Assistent. Platzhalter in eckigen Klammern wie [PERSON_1] stehen "
                    "für vertrauliche Angaben: übernimm sie wörtlich und erfinde keine neuen.")

    def session_or_404(session_id: str):
        try:
            return sessions().load(session_id), None
        except ClaususError as e:
            status = 404 if e.code == "session_not_found" else 409
            return None, JSONResponse(status_code=status, content={"status": "error", "error": e.code,
                                                                   "detail": e.message})

    def vault_rows(vault: Vault) -> list[dict[str, str]]:
        return [{"token": t, "type": ty, "original": o} for t, ty, o in vault.mapping()]

    @app.post("/chat/sessions")
    def chat_new() -> JSONResponse:
        s = sessions().create()
        sessions().save(s)
        return JSONResponse(content={"session_id": s.id})

    @app.get("/chat/sessions")
    def chat_list() -> JSONResponse:
        return JSONResponse(content={"sessions": sessions().list(), "key_from_env": bool(settings.vault_key)})

    @app.get("/chat/sessions/{session_id}")
    def chat_get(session_id: str) -> JSONResponse:
        s, err = session_or_404(session_id)
        if err:
            return err
        return JSONResponse(content={"session_id": s.id, "messages": s.messages, "vault": vault_rows(s.vault)})

    @app.delete("/chat/sessions/{session_id}")
    def chat_delete(session_id: str) -> JSONResponse:
        s, err = session_or_404(session_id)
        if err:
            return err
        sessions().delete(s.id)
        return JSONResponse(content={"deleted": s.id})

    @app.post("/chat/sessions/{session_id}/messages")
    def chat_message(session_id: str, req: ChatMessageRequest) -> JSONResponse:
        """Mask the new message with the session's vault (same person, same placeholder), send the
        masked history to the answering model, restore locally, save the session encrypted.
        Nothing is saved if a step fails."""
        if (refused := bad_model(req.answer_model)) is not None:
            return refused
        s, err = session_or_404(session_id)
        if err:
            return err
        outgoing: list[Outgoing] = []
        started = time.perf_counter()
        try:
            s.vault.preload([(k.text, k.type) for k in req.known_entities])
            spans = detect(req.text, Recorded(apertus, outgoing, "original"), person_model=settings.fast_model)
            masked = s.vault.pseudonymize(req.text, spans)
            history = [{"role": "system", "content": _CHAT_SYSTEM}]
            history += [{"role": m["role"], "content": m["masked"]} for m in s.messages]
            history.append({"role": "user", "content": masked})
            answer_masked = Recorded(apertus, outgoing, "masked").chat(
                history, model=req.answer_model or None, step="chat")
        except ClaususError as e:
            return failure(e)
        invented = sorted({t for t in _ANY_PLACEHOLDER.findall(answer_masked) if not s.vault.knows(t)})
        answer = s.vault.restore(answer_masked)
        s.messages.append({"role": "user", "original": req.text, "masked": masked,
                           "segments": split_original(req.text, masked)})
        s.messages.append({"role": "assistant", "original": answer, "masked": answer_masked,
                           "model": req.answer_model or settings.llm_name, "invented": invented})
        sessions().save(s)
        run_id = f"chat-{uuid.uuid4().hex[:12]}"
        write_receipt(Receipt(run_id=run_id, model=settings.llm_name,
                              placeholders=_placeholders(s.vault, spans), outgoing=outgoing,
                              notes=["chat", f"session {s.id}"]), settings.runs_dir)
        return JSONResponse(content={
            "masked": masked, "segments": split_original(req.text, masked),
            "answer_masked": answer_masked, "answer": answer, "invented": invented,
            "answer_model": req.answer_model or settings.llm_name,
            "vault": vault_rows(s.vault), "outgoing": [o.model_dump() for o in outgoing],
            "timings": {"total_ms": round((time.perf_counter() - started) * 1000)},
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
