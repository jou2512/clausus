"""HTTP face of the hear service. Audio never leaves this container."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile

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

    @app.post("/transcribe")
    async def transcribe(file: Annotated[UploadFile, File()]) -> dict:
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
        return {
            "text": " ".join(s.text.strip() for s in segments if s.text.strip()),
            "segments": [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments],
            "duration_s": round(duration, 2),
            "device": transcriber.device,
        }

    return app
