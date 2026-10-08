"""The organisation's confidentiality profile, kept next to the vault and encrypted like it.

Joel, 08.10.2026: a company documents its own privacy rules, and Clausus has to make that cheap.
So nobody writes a profile from scratch. It starts from the default rules, Apertus can turn the
organisation's written policy into a proposal, and every "do not mask" or "always mask" click while
working adds to it. What it holds:
  - keep:        rules for what may stay open; the judge reads them (detect/judge.py)
  - always_mask: names the organisation knows are confidential (clients, projects)
  - never_mask:  exact words that never get a placeholder (a tool, a term); never applied to persons
It holds client names, so it is never written in the clear and never sent to a model as a whole.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from clausus.detect.judge import DEFAULT_KEEP
from clausus.detect.llm import TYPE_MAP, ChatModel, extract_json
from clausus.errors import ClaususError
from clausus.models import SPAN_TYPES, Span

_FILE = "profile.bin"
_AD = b"clausus-profile"


@dataclass
class Profile:
    name: str = "Standard"
    keep: list[str] = field(default_factory=lambda: list(DEFAULT_KEEP))
    always_mask: list[dict[str, str]] = field(default_factory=list)
    never_mask: list[str] = field(default_factory=list)

    def keep_open(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.always_mask = [a for a in self.always_mask if a["text"].casefold() != text.casefold()]
        if text.casefold() not in {n.casefold() for n in self.never_mask}:
            self.never_mask.append(text)

    def always(self, text: str, span_type: str) -> None:
        text = text.strip()
        if not text:
            return
        self.never_mask = [n for n in self.never_mask if n.casefold() != text.casefold()]
        if text.casefold() not in {a["text"].casefold() for a in self.always_mask}:
            self.always_mask.append({"text": text, "type": span_type if span_type in SPAN_TYPES else "OTHER"})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Profile:
        return cls(name=str(data.get("name") or "Standard"),
                   keep=[str(k) for k in data.get("keep", DEFAULT_KEEP) if str(k).strip()],
                   always_mask=[{"text": str(a["text"]), "type": str(a.get("type", "OTHER"))}
                                for a in data.get("always_mask", []) if isinstance(a, dict) and a.get("text")],
                   never_mask=[str(n) for n in data.get("never_mask", []) if str(n).strip()])


def apply_never_mask(spans: list[Span], profile: Profile) -> list[Span]:
    """Drop finds whose exact text the organisation never masks. Persons stay masked regardless."""
    never = {n.casefold() for n in profile.never_mask}
    return [s for s in spans if s.type == "PERSON" or s.text.casefold() not in never]


class ProfileStore:
    """One profile per installation, AES-256-GCM like the chat sessions (same key)."""

    def __init__(self, directory: str | Path, key: bytes) -> None:
        self._path = Path(directory) / _FILE
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._aead = AESGCM(key)

    def load(self) -> Profile:
        if not self._path.is_file():
            return Profile()
        blob = self._path.read_bytes()
        try:
            plain = self._aead.decrypt(blob[:12], blob[12:], _AD)
        except InvalidTag as e:
            raise ClaususError("profile_unreadable",
                               "The profile cannot be decrypted (other key, or the file was changed).") from e
        return Profile.from_dict(json.loads(plain))

    def save(self, profile: Profile) -> None:
        nonce = os.urandom(12)
        blob = nonce + self._aead.encrypt(nonce, json.dumps(profile.to_dict(), ensure_ascii=False).encode(), _AD)
        tmp = self._path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
        tmp.replace(self._path)


POLICY_PROMPT = (
    "Hier ist die Datenschutz- oder Vertraulichkeitsrichtlinie einer Organisation. Mache daraus Regeln "
    "für einen Filter, der Meeting-Texte maskiert, bevor ein Sprachmodell sie sieht. Antworte NUR mit "
    'JSON: {"offen":["<Art von Angaben, die offen bleiben dürfen>"],'
    '"immer":[{"text":"<konkreter Name, der immer vertraulich ist>","typ":"PERSON|FIRMA|PROJEKT|ORT"}],'
    '"nie":["<konkretes Wort, das nie maskiert werden soll>"]}. Übernimm nur, was in der Richtlinie '
    "steht; erfinde nichts dazu.\n\nRICHTLINIE:\n"
)


def profile_from_policy(policy: str, llm: ChatModel, model: str | None = None) -> Profile:
    """A proposal from the organisation's own policy text. Not saved: a person confirms it first."""
    data = extract_json(llm.chat([{"role": "user", "content": POLICY_PROMPT + policy}], model=model,
                                 step="policy"))
    if not isinstance(data, dict):
        raise ClaususError("bad_model_output", "The policy answer is not a JSON object.")
    p = Profile(name="Aus Richtlinie",
                keep=[str(k) for k in data.get("offen", []) if str(k).strip()] or list(DEFAULT_KEEP))
    for item in data.get("immer", []):
        if isinstance(item, dict) and item.get("text"):
            p.always(str(item["text"]), TYPE_MAP.get(str(item.get("typ", "")).upper(), "OTHER"))
    for word in data.get("nie", []):
        p.keep_open(str(word))
    return p
