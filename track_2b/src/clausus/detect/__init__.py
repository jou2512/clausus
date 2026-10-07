"""Detection in four layers, overlaps resolved:
1. fixed rules (formats with a clear pattern)
2. broad model pass (all categories)
3. narrow person pass (fast model)
4. narrow names pass (organisations, projects, places)
Each model pass has blind spots the others do not share (measured, see detect/llm.py)."""

from __future__ import annotations

import re

from clausus.detect.llm import ChatModel, detect_names, detect_persons, detect_with_model
from clausus.detect.rules import find_rule_spans
from clausus.models import Span

# Swiss German articles a model often includes: "Dr Zenhäusern" (= der), "de Kellerei", "d'Sandra",
# "em Peter". "Dr" only without a dot: "Dr. Lea Zimmermann" is a title and stays. Capitalised
# "Der/Die/Das" are left alone, because "Die Post" is a name.
_ARTICLE = re.compile(r"^(?:Dr\s+|(?:de|der|die|das|em|im|am|vo|vom)\s+|d['’])")
_NAMED = {"PERSON", "ORG", "PLACE", "PROJECT", "OTHER"}
_ORG_SUFFIX = re.compile(r"(?:\b(?:AG|GmbH|SA|Sàrl|KG|Ltd|Inc|Genossenschaft|Stiftung)|\bS\.A\.)$")


def strip_article(s: Span) -> Span:
    if s.type not in _NAMED:
        return s
    m = _ARTICLE.match(s.text)
    if not m or m.end() >= len(s.text):
        return s
    return s.model_copy(update={"text": s.text[m.end():], "start": s.start + m.end()})


def type_untyped(spans: list[Span]) -> list[Span]:
    """The names pass returns a plain list (as measured), so its finds are OTHER. Such a span
    takes the type another layer found for the same place, or ORG for a company suffix.
    Only the type changes, never what is masked."""
    out = []
    for s in spans:
        if s.type == "OTHER":
            typed = [o for o in spans if o.type != "OTHER" and o.start < s.end and s.start < o.end]
            if typed:
                s = s.model_copy(update={"type": max(typed, key=lambda o: o.length).type})
            elif _ORG_SUFFIX.search(s.text):
                s = s.model_copy(update={"type": "ORG"})
        out.append(s)
    return out


def resolve_overlaps(spans: list[Span]) -> list[Span]:
    """Longer spans win; on equal length a rule beats the model. Result sorted by position."""
    ranked = sorted(spans, key=lambda s: (-s.length, s.source != "rule", s.start))
    kept: list[Span] = []
    for s in ranked:
        if all(s.end <= k.start or s.start >= k.end for k in kept):
            kept.append(s)
    return sorted(kept, key=lambda s: s.start)


def detect(text: str, llm: ChatModel, person_model: str | None = None) -> list[Span]:
    """All layers must run. If any model call fails, the error propagates: a half
    detection must never be treated as a complete one (fail closed)."""
    spans = find_rule_spans(text) + detect_with_model(text, llm)
    spans += detect_persons(text, llm, model=person_model)
    spans += detect_names(text, llm)
    spans = [strip_article(s) for s in spans]
    return resolve_overlaps(type_untyped(spans))
