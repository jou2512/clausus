"""The speech model. Heavy imports (torch, transformers) happen only when the model loads,
so the service code stays testable without them."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from hear_service.audio import SAMPLE_RATE

log = logging.getLogger("hear.engine")

DEFAULT_MODEL = "Flix-AI/flix-swissgerman-full"


@dataclass
class Segment:
    start: float
    end: float
    text: str


class Transcriber(Protocol):
    device: str
    model: str
    ready: bool

    def transcribe(self, audio: np.ndarray) -> list[Segment]: ...


class FlixTranscriber:
    """Swiss German Whisper fine-tune (Apache-2.0). Outputs High German text.

    Measured 01.10.2026 on an RTX 3070 (fp16): 2.8x real time, 3.1 GB model, peak 7.1 GB
    VRAM at batch 8. On CPU a 29-minute meeting did not finish in 90 minutes, so CPU is
    for short clips only. Batch size defaults to 2 to stay well below 8 GB.
    """

    def __init__(self, model: str | None = None, batch_size: int | None = None) -> None:
        self.model = model or os.environ.get("HEAR_MODEL", DEFAULT_MODEL)
        self.batch_size = batch_size or int(os.environ.get("HEAR_BATCH_SIZE", "2"))
        self.device = "unknown"
        self.ready = False
        self.error: str | None = None
        self._pipe = None

    def load_in_background(self) -> None:
        threading.Thread(target=self._load, name="hear-load", daemon=True).start()

    def _load(self) -> None:
        try:
            import torch
            from transformers import pipeline

            cuda = torch.cuda.is_available()
            self.device = "cuda" if cuda else "cpu"
            log.info("loading %s on %s", self.model, self.device)
            self._pipe = pipeline("automatic-speech-recognition", model=self.model,
                                  chunk_length_s=30, batch_size=self.batch_size,
                                  device="cuda:0" if cuda else "cpu",
                                  dtype=torch.float16 if cuda else torch.float32)
            self.ready = True
            log.info("model ready on %s", self.device)
        except Exception as e:  # noqa: BLE001 — any load failure must show up in /health
            self.error = f"{type(e).__name__}: {e}"
            log.error("model failed to load: %s", self.error)

    def transcribe(self, audio: np.ndarray) -> list[Segment]:
        if not self.ready or self._pipe is None:
            raise RuntimeError("model not ready")
        out = self._pipe({"raw": audio, "sampling_rate": SAMPLE_RATE}, return_timestamps=True)
        chunks = out.get("chunks") or [{"timestamp": (0.0, len(audio) / SAMPLE_RATE), "text": out["text"]}]
        segments = []
        for c in chunks:
            start, end = c.get("timestamp") or (None, None)
            segments.append(Segment(start=float(start or 0.0), end=float(end or start or 0.0),
                                    text=str(c.get("text", ""))))
        return segments
