from __future__ import annotations

import logging
from typing import Any

import httpx

from clausus.config import Settings
from clausus.errors import ClaususError

log = logging.getLogger("clausus.llm")


class ApertusClient:
    """The single place that talks to the Apertus endpoint (OpenAI-compatible)."""

    def __init__(self, settings: Settings, http: httpx.Client | None = None) -> None:
        self._settings = settings
        self._http = http or httpx.Client(timeout=30.0)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._settings.llm_api_key}"}

    def chat(self, messages: list[dict[str, Any]], *, model: str | None = None,
             max_tokens: int = 1500) -> str:
        """One chat completion, temperature 0. Content is never logged."""
        url = f"{self._settings.llm_base_url}/chat/completions"
        body = {"model": model or self._settings.llm_name, "messages": messages,
                "temperature": 0, "max_tokens": max_tokens}
        try:
            r = self._http.post(url, headers=self._headers(), json=body, timeout=180.0)
        except httpx.HTTPError as e:
            log.warning("apertus chat unreachable: %s", type(e).__name__)
            raise ClaususError("unreachable", "Apertus endpoint is not reachable.") from e
        if r.status_code in (401, 403):
            raise ClaususError("auth_failed", "Apertus rejected the API key.")
        if r.status_code >= 400:
            raise ClaususError("bad_response", f"Apertus answered HTTP {r.status_code}.")
        try:
            return r.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise ClaususError("bad_response", "Apertus answered with an unexpected payload.") from e

    def list_models(self) -> list[str]:
        url = f"{self._settings.llm_base_url}/models"
        try:
            r = self._http.get(url, headers=self._headers())
        except httpx.HTTPError as e:
            log.warning("apertus unreachable: %s", type(e).__name__)
            raise ClaususError("unreachable", "Apertus endpoint is not reachable.") from e
        if r.status_code in (401, 403):
            raise ClaususError("auth_failed", "Apertus rejected the API key.")
        if r.status_code >= 400:
            raise ClaususError("bad_response", f"Apertus answered HTTP {r.status_code}.")
        try:
            return [m["id"] for m in r.json()["data"]]
        except (ValueError, KeyError, TypeError) as e:
            raise ClaususError("bad_response", "Apertus answered with an unexpected payload.") from e
