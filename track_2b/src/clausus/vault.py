"""The vault: maps confidential text to placeholders and back. It never leaves the house.

One vault per session. The same person gets the same placeholder throughout,
also when only the surname or first name comes up later ("Thomas Brunner" ->
"Brunner"). An ambiguous short form ("Thomas" with two Thomases) is never merged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from clausus.detect.llm import is_ordinary_small
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


# Trap set 08.10.2026: "Projekt Kleinhünige" and "Alpenblick Consulting GmbH" were masked, the later
# "Kleinhünige zahlt" and "die Offerte von Alpenblick" leaked. The bare name is the same thing.
# A company only gets a short form when everything after its first word is a legal form or a trade
# word, so "Schweizerische Post" does not spread to "schweizerische" and "Gemeinde Uster" not to
# "Gemeinde".
_PROJECT_LABEL = re.compile(r"^(?:Projekt|Projekts|Project)\s+(\S.*)$")
_TRADE_WORDS = {
    "ag", "gmbh", "sa", "sàrl", "sarl", "kg", "ltd", "inc", "llc", "&", "co", "und", "partner", "consulting",
    "informatik", "treuhand", "bau", "holding", "group", "gruppe", "solutions", "services", "technik",
    "logistik", "digital", "engineering", "immobilien", "beratung", "software", "systems", "it", "media",
    "design", "architekten", "electronics", "elektro", "transport", "management",
}


def _short_form(variant: str, span_type: str) -> str | None:
    if span_type == "PROJECT":
        m = _PROJECT_LABEL.match(variant.strip())
        return m.group(1) if m else None
    if span_type == "ORG":
        words = variant.split()
        if (len(words) >= 2 and words[0][:1].isupper() and len(words[0]) >= 4
                and all(w.casefold().strip(".,") in _TRADE_WORDS for w in words[1:])):
            return words[0]
    return None


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
        # The exact same text is the same thing, whatever type a later pass gives it ("brunnag" came
        # back as a project after "Brunnag" was an organisation: two placeholders for one company).
        for e in self._entries:
            if any(v.casefold() == text.casefold() for v in e.variants):
                return e.token
        same_type = [e for e in self._entries if e.type == span_type]
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
    def pseudonymize(self, text: str, spans: list[Span], keep_open: set[str] | None = None) -> str:
        """Number placeholders in reading order, then replace from the end so the
        offsets of earlier spans stay valid. `keep_open` (casefolded) are words the organisation
        released: the sweep leaves them even if this vault knows them, persons excepted."""
        ordered = sorted(spans, key=lambda s: s.start)
        tokens = [self.token_for(s.text, s.type) for s in ordered]
        out = text
        for s, token in reversed(list(zip(ordered, tokens, strict=True))):
            out = out[:s.start] + token + out[s.end:]
        return self._sweep(out, keep_open or set())

    def _sweep(self, text: str, keep_open: set[str] | None = None) -> str:
        """Mask every later mention of something already known (SPEC: once found, always found)."""
        needles: dict[str, set[str]] = {}
        spelled: dict[str, str] = {}  # how the name was found, to tell "Wil" from "wil" (= weil)
        for e in self._entries:
            for v in e.variants:
                needles.setdefault(v.casefold(), set()).add(e.token)
                spelled.setdefault(v.casefold(), v)
            if e.type == "PERSON":
                for w in e.keys:
                    if len(w) >= 3:
                        needles.setdefault(w, set()).add(e.token)
        for e in self._entries:  # short forms last: they never take a needle another entry owns
            for v in e.variants:
                short = _short_form(v, e.type)
                if short and short.casefold() not in needles:
                    needles[short.casefold()] = {e.token}
                    spelled[short.casefold()] = short
        # Released in the profile ("Don't mask"), also in a running chat: the word itself and every
        # form of the same entry ("Zellweger" for "Zellweger Logistik"). Never a person.
        released = {e.token for e in self._entries if e.type != "PERSON"
                    and any(v.casefold() in (keep_open or set()) for v in e.variants)}
        for needle in sorted(needles, key=len, reverse=True):
            tokens = needles[needle]
            if tokens <= released:
                continue
            name = spelled.get(needle, needle)
            pattern = re.compile(rf"(?<![\w\[]){re.escape(needle)}(?![\w\]])", re.IGNORECASE)
            if len(tokens) == 1:
                token = next(iter(tokens))
                text = pattern.sub(lambda m, name=name, token=token: m.group(0)
                                   if is_ordinary_small(name, m.group(0)) else token, text)
            else:
                # Ambiguous short form ("Marco" with two Marcos): never guess which person,
                # but never leave it open either. It gets a placeholder of its own.
                # Benchmark 02.10.2026: leaving it open leaked it in 3 of 3 runs.
                text = pattern.sub(lambda m, name=name: m.group(0) if is_ordinary_small(name, m.group(0))
                                   else self._ambiguous_token(m.group(0)), text)
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

    # --- persistence (encrypted by clausus.sessions, never written in the clear) -------------
    def to_dict(self) -> dict:
        return {"entries": [{"token": e.token, "type": e.type, "variants": e.variants, "keys": sorted(e.keys)}
                            for e in self._entries],
                "counters": dict(self._counters)}

    @classmethod
    def from_dict(cls, data: dict) -> Vault:
        v = cls()
        v._entries = [_Entry(token=e["token"], type=e["type"], variants=list(e["variants"]),
                             keys=frozenset(e["keys"])) for e in data.get("entries", [])]
        v._counters = {k: int(n) for k, n in data.get("counters", {}).items()}
        return v

    def mapping(self) -> list[tuple[str, str, str]]:
        """(token, type, original) for the local playground. Never sent to a model."""
        return [(e.token, e.type, e.canonical) for e in self._entries]

    def variants(self) -> dict[str, list[str]]:
        """Every spelling found per placeholder, to align the local display. Never sent to a model."""
        return {e.token: list(e.variants) for e in self._entries}

    def knows(self, token: str) -> bool:
        return any(e.token == token for e in self._entries)
