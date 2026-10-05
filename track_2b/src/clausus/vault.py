"""The vault: maps confidential text to placeholders and back. It never leaves the house.

One vault per session. The same person gets the same placeholder throughout,
also when only the surname or first name comes up later ("Thomas Brunner" ->
"Brunner"). An ambiguous short form ("Thomas" with two Thomases) is never merged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from clausus.models import Span

_STOP = {"ag", "gmbh", "sa", "sàrl", "sarl", "kg", "de", "vo", "von", "der", "die", "das", "dr",
         "herr", "frau", "&", "und", "and", "the", "inc", "ltd"}
_TOKEN = re.compile(r"\[[A-Z]+_\d+\]")


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"[^\wäöüÄÖÜàéèç&]+", text.casefold()) if w]


def _significant(text: str) -> frozenset[str]:
    words = _words(text)
    sig = frozenset(w for w in words if w not in _STOP)
    return sig or frozenset(words)


@dataclass
class _Entry:
    token: str
    type: str
    variants: list[str] = field(default_factory=list)
    keys: frozenset[str] = frozenset()

    @property
    def canonical(self) -> str:
        return max(self.variants, key=len)


class Vault:
    def __init__(self) -> None:
        self._entries: list[_Entry] = []
        self._counters: dict[str, int] = {}

    # --- registration -----------------------------------------------------------------------
    def token_for(self, text: str, span_type: str) -> str:
        text = text.strip()
        keys = _significant(text)
        same_type = [e for e in self._entries if e.type == span_type]
        for e in same_type:
            if any(v.casefold() == text.casefold() for v in e.variants):
                return e.token
        related = [e for e in same_type if keys and (keys <= e.keys or e.keys <= keys)]
        if len(related) == 1:
            e = related[0]
            e.variants.append(text)
            e.keys = e.keys | keys
            return e.token
        n = self._counters.get(span_type, 0) + 1
        self._counters[span_type] = n
        entry = _Entry(token=f"[{span_type}_{n}]", type=span_type, variants=[text], keys=keys)
        self._entries.append(entry)
        return entry.token

    def preload(self, known: list[tuple[str, str]]) -> None:
        """Names the user already knows are confidential (attendee or client list)."""
        for text, span_type in known:
            if text.strip():
                self.token_for(text, span_type)

    @property
    def tokens(self) -> list[tuple[str, str]]:
        return [(e.token, e.type) for e in self._entries]

    # --- masking ----------------------------------------------------------------------------
    def pseudonymize(self, text: str, spans: list[Span]) -> str:
        """Number placeholders in reading order, then replace from the end so the
        offsets of earlier spans stay valid."""
        ordered = sorted(spans, key=lambda s: s.start)
        tokens = [self.token_for(s.text, s.type) for s in ordered]
        out = text
        for s, token in reversed(list(zip(ordered, tokens, strict=True))):
            out = out[:s.start] + token + out[s.end:]
        return self._sweep(out)

    def _sweep(self, text: str) -> str:
        """Mask every later mention of something already known (SPEC: once found, always found)."""
        needles: dict[str, set[str]] = {}
        for e in self._entries:
            for v in e.variants:
                needles.setdefault(v.casefold(), set()).add(e.token)
            if e.type == "PERSON":
                for w in e.keys:
                    if len(w) >= 3:
                        needles.setdefault(w, set()).add(e.token)
        for needle in sorted(needles, key=len, reverse=True):
            tokens = needles[needle]
            pattern = re.compile(rf"(?<![\w\[]){re.escape(needle)}(?![\w\]])", re.IGNORECASE)
            if len(tokens) == 1:
                text = pattern.sub(next(iter(tokens)), text)
            else:
                # Ambiguous short form ("Marco" with two Marcos): never guess which person,
                # but never leave it open either. It gets a placeholder of its own.
                # Benchmark 02.10.2026: leaving it open leaked it in 3 of 3 runs.
                text = pattern.sub(lambda m: self._ambiguous_token(m.group(0)), text)
        return text

    def _ambiguous_token(self, word: str) -> str:
        for e in self._entries:
            if e.type == "PERSON" and any(v.casefold() == word.casefold() for v in e.variants):
                return e.token
        n = self._counters.get("PERSON", 0) + 1
        self._counters["PERSON"] = n
        entry = _Entry(token=f"[PERSON_{n}]", type="PERSON", variants=[word], keys=frozenset())
        self._entries.append(entry)
        return entry.token

    def restore(self, text: str) -> str:
        """Put originals back. If the answer already wrote the first word of the original
        right before the placeholder ("Projekt [PROJECT_1]" for "Projekt Nordlicht"), that
        word is not doubled."""
        lookup = {e.token: e.canonical for e in self._entries}

        def put_back(m: re.Match[str]) -> str:
            before, token = m.group(1) or "", m.group(2)
            original = lookup.get(token)
            if original is None:
                return m.group(0)
            first = original.split(" ", 1)[0]
            if before and before.strip().casefold() == first.casefold():
                return original
            return before + original

        return re.sub(r"(\b[\wäöüÄÖÜ]+\s+)?(\[[A-Z]+_\d+\])", put_back, text)

    def mapping(self) -> list[tuple[str, str, str]]:
        """(token, type, original) for the local playground. Never sent to a model."""
        return [(e.token, e.type, e.canonical) for e in self._entries]

    def knows(self, token: str) -> bool:
        return any(e.token == token for e in self._entries)
