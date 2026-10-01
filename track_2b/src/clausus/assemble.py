"""Puts real names back into an answer and reports what does not add up."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from clausus.vault import Vault

_TOKEN = re.compile(r"\[[A-Z]+_\d+\]")


@dataclass
class AssembleResult:
    text: str
    invented: list[str] = field(default_factory=list)  # placeholders the vault never issued
    missing: list[str] = field(default_factory=list)   # issued placeholders absent from the answer


def assemble(answer: str, vault: Vault) -> AssembleResult:
    seen: list[str] = []
    for m in _TOKEN.finditer(answer):
        if m.group(0) not in seen:
            seen.append(m.group(0))
    invented = [t for t in seen if not vault.knows(t)]
    missing = [t for t, _ in vault.tokens if t not in seen]
    return AssembleResult(text=vault.restore(answer), invented=invented, missing=missing)
