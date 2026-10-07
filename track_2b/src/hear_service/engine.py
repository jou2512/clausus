"""The speech model. Heavy imports (torch, transformers) happen only when the model loads,
so the service code stays testable without them."""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
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
    for short clips only.

    Long recordings use Whisper's own long-form decoding, not the pipeline's chunked batches.
    Measured 07.10.2026 on the same GPU with a 50 s recording: chunk_length_s=30 with
    batch_size=2 hung in 4 of 4 runs (also with the 05.10 image, so not a library update);
    long-form finished in about 14 s in 3 of 4 runs, the fourth hit a 120 s limit, cause unknown.

    An idle model is dropped from memory after HEAR_IDLE_UNLOAD_S seconds (default 600, 0 keeps
    it). Measured 07.10.2026: the loaded model held 6 GB VRAM on a desktop that was doing
    nothing with it. The next request loads it again from the disk cache.
    """

    def __init__(self, model: str | None = None, idle_unload_s: float | None = None) -> None:
        self.model = model or os.environ.get("HEAR_MODEL", DEFAULT_MODEL)
        self.device = "unknown"
        self.ready = False
        self.error: str | None = None
        self.idle_unload_s = (idle_unload_s if idle_unload_s is not None
                              else float(os.environ.get("HEAR_IDLE_UNLOAD_S", "600")))
        self._pipe = None
        self._lock = threading.Lock()  # one inference, load or unload at a time
        self._last_used = time.monotonic()

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def load_in_background(self) -> None:
        threading.Thread(target=self._load, name="hear-load", daemon=True).start()
        if self.idle_unload_s > 0:
            threading.Thread(target=self._watch_idle, name="hear-idle", daemon=True).start()

    def _load(self) -> None:
        try:
            with self._lock:
                self._ensure_loaded()
            self.ready = True
            log.info("model ready on %s", self.device)
        except Exception as e:  # noqa: BLE001 — any load failure must show up in /health
            self.error = f"{type(e).__name__}: {e}"
            log.error("model failed to load: %s", self.error)

    def _ensure_loaded(self) -> None:
        if self._pipe is None:
            log.info("loading %s", self.model)
            self._pipe = self._build_pipe()
            self._last_used = time.monotonic()

    def _build_pipe(self):  # returns a transformers pipeline, imported lazily
        import torch
        from transformers import pipeline

        cuda = torch.cuda.is_available()
        self.device = "cuda" if cuda else "cpu"
        return pipeline("automatic-speech-recognition", model=self.model,
                        device="cuda:0" if cuda else "cpu",
                        dtype=torch.float16 if cuda else torch.float32)

    def _release(self) -> None:
        gc.collect()
        if self.device == "cuda":
            import torch

            torch.cuda.empty_cache()  # hands the freed blocks back to the driver

    def unload_if_idle(self, now: float | None = None) -> bool:
        """Drop the model if it has been idle long enough. Never waits on a running request."""
        if self.idle_unload_s <= 0 or not self._lock.acquire(blocking=False):
            return False
        try:
            idle = (now if now is not None else time.monotonic()) - self._last_used
            if self._pipe is None or idle < self.idle_unload_s:
                return False
            self._pipe = None
            self._release()
            log.info("model unloaded after %.0f s idle", idle)
            return True
        finally:
            self._lock.release()

    def _watch_idle(self) -> None:
        while True:
            time.sleep(min(30.0, self.idle_unload_s))
            self.unload_if_idle()

    def transcribe(self, audio: np.ndarray) -> list[Segment]:
        if not self.ready:
            raise RuntimeError("model not ready")
        with self._lock:
            try:
                self._ensure_loaded()
            except Exception as e:  # e.g. VRAM taken by something else meanwhile
                raise RuntimeError(f"model could not be reloaded: {type(e).__name__}") from e
            out = self._pipe({"raw": audio, "sampling_rate": SAMPLE_RATE}, return_timestamps=True)
            self._last_used = time.monotonic()
        chunks = out.get("chunks") or [{"timestamp": (0.0, len(audio) / SAMPLE_RATE), "text": out["text"]}]
        segments = []
        for c in chunks:
            start, end = c.get("timestamp") or (None, None)
            segments.append(Segment(start=float(start or 0.0), end=float(end or start or 0.0),
                                    text=str(c.get("text", ""))))
        return segments
