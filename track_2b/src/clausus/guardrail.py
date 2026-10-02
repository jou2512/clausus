"""Clausus as a guardrail for LiteLLM (Generic Guardrail API).

Contract read from docs.litellm.ai/docs/adding_provider/generic_guardrail_api on 02.10.2026:
LiteLLM calls POST /beta/litellm_basic_guardrail_api with `texts`, `input_type`
("request" | "response") and `litellm_call_id`, which stays the same for the request
and the response of one LLM call. The answer is {"action": "NONE" | "BLOCKED" |
"GUARDRAIL_INTERVENED", "texts": [...], "blocked_reason": "..."}.

The vault for a call lives only in memory between the request and the response,
is deleted as soon as the response is restored, and expires after a TTL otherwise.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from pydantic import BaseModel, Field

from clausus.vault import Vault


class GuardrailRequest(BaseModel):
    texts: list[str] = Field(default_factory=list)
    images: list[str] | None = None
    input_type: str = "request"
    litellm_call_id: str | None = None


class GuardrailStore:
    """Vaults keyed by litellm_call_id, in memory only, with expiry."""

    def __init__(self, ttl_s: float = 600, clock: Callable[[], float] = time.monotonic) -> None:
        self._ttl = ttl_s
        self._clock = clock
        self._items: dict[str, tuple[Vault, float]] = {}
        self._lock = threading.Lock()

    def _purge(self) -> None:
        now = self._clock()
        for key in [k for k, (_, t) in self._items.items() if now - t > self._ttl]:
            del self._items[key]

    def put(self, call_id: str, vault: Vault) -> None:
        with self._lock:
            self._purge()
            self._items[call_id] = (vault, self._clock())

    def pop(self, call_id: str) -> Vault | None:
        with self._lock:
            self._purge()
            item = self._items.pop(call_id, None)
        return item[0] if item else None

    def __len__(self) -> int:
        with self._lock:
            self._purge()
            return len(self._items)
