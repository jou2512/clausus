from __future__ import annotations

import logging
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from clausus.config import Settings
from clausus.errors import ClaususError
from clausus.receipt import Outgoing

log = logging.getLogger("clausus.llm")


class ChatClient:
    """Talks to one OpenAI-compatible endpoint. Content is never logged."""

    def __init__(self, base_url: str, api_key: str, model: str, label: str,
                 http: httpx.Client | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.model = model
        self.label = label
        self._http = http or httpx.Client(timeout=30.0)

    @property
    def destination(self) -> str:
        return f"{self.label} @ {urlparse(self._base_url).hostname}"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _check(self, r: httpx.Response) -> None:
        if r.status_code in (401, 403):
            raise ClaususError("auth_failed", f"{self.label} rejected the API key.")
        if r.status_code == 429:
            raise ClaususError("rate_limited", f"{self.label} is rate-limiting us (HTTP 429). Try again shortly.")
        if r.status_code >= 400:
            raise ClaususError("bad_response", f"{self.label} answered HTTP {r.status_code}.")

    def chat(self, messages: list[dict[str, Any]], *, model: str | None = None,
             max_tokens: int = 1500, step: str | None = None) -> str:
        """One chat completion, temperature 0. `step` is only used by the recorder."""
        body = {"model": model or self.model, "messages": messages,
                "temperature": 0, "max_tokens": max_tokens}
        try:
            r = self._http.post(f"{self._base_url}/chat/completions", headers=self._headers(),
                                json=body, timeout=180.0)
        except httpx.HTTPError as e:
            log.warning("%s chat unreachable: %s", self.label, type(e).__name__)
            raise ClaususError("unreachable", f"{self.label} endpoint is not reachable.") from e
        self._check(r)
        try:
            return r.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise ClaususError("bad_response", f"{self.label} answered with an unexpected payload.") from e

    def list_models(self) -> list[str]:
        try:
            r = self._http.get(f"{self._base_url}/models", headers=self._headers())
        except httpx.HTTPError as e:
            log.warning("%s unreachable: %s", self.label, type(e).__name__)
            raise ClaususError("unreachable", f"{self.label} endpoint is not reachable.") from e
        self._check(r)
        try:
            return [m["id"] for m in r.json()["data"]]
        except (ValueError, KeyError, TypeError) as e:
            raise ClaususError("bad_response", f"{self.label} answered with an unexpected payload.") from e


class ApertusClient(ChatClient):
    def __init__(self, settings: Settings, http: httpx.Client | None = None) -> None:
        super().__init__(settings.llm_base_url, settings.llm_api_key, settings.llm_name,
                         "apertus", http)


class Recorded:
    """Wraps a client for one request and logs every call that leaves the house:
    where it went, and whether it carried original or masked text. Original text
    is never stored, masked text is stored verbatim."""

    def __init__(self, inner: ChatClient, outgoing: list[Outgoing], kind: str) -> None:
        self._inner = inner
        self._outgoing = outgoing
        self._kind = kind

    def chat(self, messages: list[dict[str, Any]], *, model: str | None = None,
             step: str | None = None, **_: Any) -> str:
        content = "\n".join(str(m.get("content", "")) for m in messages)
        entry = Outgoing(
            step=step or "chat", destination=self._inner.destination, kind=self._kind,
            model=model or self._inner.model, text=content if self._kind == "masked" else None,
        )
        self._outgoing.append(entry)  # listed before the call: a failed call still left the house
        start = time.perf_counter()
        try:
            return self._inner.chat(messages, model=model, step=step)
        finally:
            entry.ms = round((time.perf_counter() - start) * 1000)
