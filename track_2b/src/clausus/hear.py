"""Speech to text. Default: the local hear container, audio stays in the house.

Optional: any OpenAI-compatible transcription endpoint (STT_BASE_URL), e.g. an in-house
Whisper server. Audio carries originals, so a run that uses it lists the call in the receipt
as original content leaving Clausus.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from clausus.errors import ClaususError


@dataclass
class Transcript:
    text: str
    duration_s: float
    device: str
    local: bool = True
    destination: str = "local"
    model: str = ""


class HearClient:
    def __init__(self, base_url: str, http: httpx.Client | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = http or httpx.Client()

    def transcribe(self, audio: bytes, filename: str) -> Transcript:
        try:
            r = self._http.post(f"{self._base_url}/transcribe",
                                files={"file": (filename, audio, "application/octet-stream")},
                                timeout=3600.0)
        except httpx.HTTPError as e:
            raise ClaususError("hear_unreachable", "The hear service is not reachable.") from e
        if r.status_code == 503:
            raise ClaususError("hear_loading", "The speech model is still loading.")
        if r.status_code in (400, 422):
            raise ClaususError("hear_bad_audio", "The audio file could not be read.")
        if r.status_code >= 400:
            raise ClaususError("hear_failed", f"The hear service answered HTTP {r.status_code}.")
        body = r.json()
        return Transcript(text=str(body.get("text", "")), duration_s=float(body.get("duration_s", 0)),
                          device=str(body.get("device", "")))

    def health(self) -> str:
        try:
            r = self._http.get(f"{self._base_url}/health", timeout=5.0)
            return str(r.json().get("status", "unknown"))
        except (httpx.HTTPError, ValueError):
            return "unreachable"


class EndpointHearClient:
    """OpenAI-compatible POST {base}/audio/transcriptions (multipart: file, model)."""

    def __init__(self, base_url: str, api_key: str, model: str, http: httpx.Client | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._model = model
        self._http = http or httpx.Client()
        self.destination = urlparse(self._base_url).hostname or self._base_url

    def transcribe(self, audio: bytes, filename: str) -> Transcript:
        try:
            r = self._http.post(f"{self._base_url}/audio/transcriptions", headers=self._headers,
                                files={"file": (filename, audio, "application/octet-stream")},
                                data={"model": self._model}, timeout=3600.0)
        except httpx.HTTPError as e:
            raise ClaususError("hear_unreachable", "The transcription endpoint is not reachable.") from e
        if r.status_code == 429:
            raise ClaususError("rate_limited", "The transcription endpoint is rate limiting.")
        if r.status_code in (400, 415, 422):
            raise ClaususError("hear_bad_audio", "The audio file could not be read.")
        if r.status_code >= 400:
            raise ClaususError("hear_failed", f"The transcription endpoint answered HTTP {r.status_code}.")
        try:
            text = str(r.json().get("text", ""))
        except ValueError as e:
            raise ClaususError("hear_failed", "The transcription endpoint did not answer with JSON.") from e
        return Transcript(text=text, duration_s=0.0, device=self._model, local=False,
                          destination=self.destination, model=self._model)

    def health(self) -> str:
        try:
            r = self._http.get(f"{self._base_url}/models", headers=self._headers, timeout=5.0)
        except httpx.HTTPError:
            return "unreachable"
        return "ok" if r.status_code < 400 else "unreachable"
