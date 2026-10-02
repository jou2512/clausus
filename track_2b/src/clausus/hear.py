"""Client for the local hear service. Audio goes from Clausus to that container only."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from clausus.errors import ClaususError


@dataclass
class Transcript:
    text: str
    duration_s: float
    device: str


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
