"""HTTP face of the hear service. Audio never leaves this container."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from hear_service.audio import SAMPLE_RATE, decode
from hear_service.engine import FlixTranscriber, Transcriber

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("hear.app")


def create_app(transcriber: Transcriber | None = None) -> FastAPI:
    if transcriber is None:
        flix = FlixTranscriber()
        flix.load_in_background()
        transcriber = flix
    app = FastAPI(title="Clausus Hear", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        if not transcriber.ready:
            error = getattr(transcriber, "error", None)
            return {"status": "failed" if error else "loading", "device": transcriber.device,
                    "model": transcriber.model, **({"error": error} if error else {})}
        return {"status": "ok", "device": transcriber.device, "model": transcriber.model}

    async def run(file: UploadFile) -> tuple[list, float]:
        if not transcriber.ready:
            raise HTTPException(503, "model is still loading")
        data = await file.read()
        if not data:
            raise HTTPException(400, "empty upload")
        try:
            audio = decode(data)
        except ValueError as e:
            raise HTTPException(422, "audio could not be decoded") from e
        segments = transcriber.transcribe(audio)
        duration = len(audio) / SAMPLE_RATE
        log.info("transcribed %.1f s of audio on %s", duration, transcriber.device)  # no content
        return segments, duration

    def joined(segments: list) -> str:
        return " ".join(s.text.strip() for s in segments if s.text.strip())

    @app.post("/transcribe")
    async def transcribe(file: Annotated[UploadFile, File()]) -> dict:
        segments, duration = await run(file)
        return {
            "text": joined(segments),
            "segments": [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments],
            "duration_s": round(duration, 2),
            "device": transcriber.device,
        }

    # OpenAI-compatible face, so any tool that speaks /v1/audio/transcriptions can use the
    # Swiss German model, Clausus' own STT_BASE_URL mode included. The model field is accepted
    # and ignored: this service runs one model.
    @app.post("/v1/audio/transcriptions", response_model=None)
    async def openai_transcriptions(file: Annotated[UploadFile, File()],
                                    model: Annotated[str, Form()] = "",
                                    response_format: Annotated[str, Form()] = "json") -> dict | PlainTextResponse:
        segments, duration = await run(file)
        text = joined(segments)
        if response_format == "text":
            return PlainTextResponse(text)
        if response_format == "verbose_json":
            return {"text": text, "language": "de", "duration": round(duration, 2),
                    "segments": [{"id": i, "start": s.start, "end": s.end, "text": s.text.strip()}
                                 for i, s in enumerate(segments)]}
        return {"text": text}

    @app.get("/v1/models")
    def openai_models() -> dict:
        return {"object": "list", "data": [{"id": transcriber.model, "object": "model", "owned_by": "local"}]}

    return app
