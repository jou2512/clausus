"""Detection = layer 1 (rules) + layer 2 (Apertus), overlaps resolved."""

from __future__ import annotations

from clausus.detect.llm import ChatModel, detect_persons, detect_with_model
from clausus.detect.rules import find_rule_spans
from clausus.models import Span


def resolve_overlaps(spans: list[Span]) -> list[Span]:
    """Longer spans win; on equal length a rule beats the model. Result sorted by position."""
    ranked = sorted(spans, key=lambda s: (-s.length, s.source != "rule", s.start))
    kept: list[Span] = []
    for s in ranked:
        if all(s.end <= k.start or s.start >= k.end for k in kept):
            kept.append(s)
    return sorted(kept, key=lambda s: s.start)


def detect(text: str, llm: ChatModel, person_model: str | None = None) -> list[Span]:
    """Rules, a broad model pass and a narrow person pass must all run. If any model
    call fails, the error propagates: a half detection must never be treated as a
    complete one (fail closed)."""
    spans = find_rule_spans(text) + detect_with_model(text, llm)
    spans += detect_persons(text, llm, model=person_model)
    return resolve_overlaps(spans)
