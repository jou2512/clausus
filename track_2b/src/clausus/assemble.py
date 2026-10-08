"""Puts real names back into an answer and reports what does not add up."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from clausus.detect.rules import find_rule_spans
from clausus.vault import Vault

_TOKEN = re.compile(r"\[[A-Z]+_\d+\]")
# Placeholder-like brackets ("[PERSON_1]", "[Dein Name]"), not JSON lists: a protocol arrives as JSON.
_BRACKETS = re.compile(r"\[[A-Za-zÄÖÜäöü][A-Za-zÄÖÜäöü_ .\-]{1,40}(?:_\d+)?\]")
_NUMBER = re.compile(r"(?<![\w\[])\d(?:[\d'’]*\d)?(?:[.,]\d+)*")


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


def _norm(number: str) -> str:
    return number.replace("'", "").replace("’", "")


def _known_numbers(texts: list[str]) -> set[str]:
    known: set[str] = set()
    for t in texts:
        for m in _NUMBER.finditer(t):
            n = _norm(m.group(0))
            known.add(n)
            known.update(re.split(r"[.,]", n))
    return known


def unsourced_numbers(answer: str, sent: list[str]) -> list[str]:
    """Figures in the model's answer that were in nothing it was sent (masked text, history,
    instruction). The model never saw a real amount or date, so such a figure is invented or
    computed; restore leaves it untouched and it would read like a fact from the meeting.
    Integers below 10 stay out: list numbering and counts ("3 Punkte") are no claim."""
    blank = _BRACKETS.sub(lambda m: " " * len(m.group(0)), answer)
    known = _known_numbers(sent)
    sent_text = "\n".join(sent).casefold()

    def sourced(text: str) -> bool:
        numbers = [_norm(m.group(0)) for m in _NUMBER.finditer(text)]
        return text.casefold() in sent_text or (bool(numbers) and all(
            n in known or all(p in known for p in re.split(r"[.,]", n)) for n in numbers))

    found: list[tuple[int, str]] = []
    covered: list[tuple[int, int]] = []
    # Longest first, so "50'000" inside "CHF 50'000" is not reported a second time.
    for span in sorted(find_rule_spans(blank), key=lambda s: (s.start, -(s.end - s.start))):
        if any(a <= span.start < b for a, b in covered):
            continue
        covered.append((span.start, span.end))
        if not sourced(span.text):
            found.append((span.start, span.text))
    for m in _NUMBER.finditer(blank):
        if any(a <= m.start() < b for a, b in covered):
            continue
        n = _norm(m.group(0))
        if n.isdigit() and int(n) < 10:
            continue
        if not sourced(m.group(0)):
            found.append((m.start(), m.group(0)))
    return list(dict.fromkeys(text for _, text in sorted(found)))
