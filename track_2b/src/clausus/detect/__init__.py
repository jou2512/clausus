"""Detection in four layers, overlaps resolved:
1. fixed rules (formats with a clear pattern)
2. broad model pass (all categories)
3. narrow person pass (fast model)
4. narrow names pass (organisations, projects, places)
Each model pass has blind spots the others do not share (measured, see detect/llm.py)."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from clausus.detect.judge import judge as judge_finds
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


# Joel's chat 07.10.2026: on a 2755-character planning text the 70B pass returned three names and
# the person pass missed "Gisigers"; paragraph by paragraph both were found. Measured on the
# held-out set joined into long texts (191 spans): one call 9.4 % leaked, 700-char sections 4.7 %,
# 400-char sections 2.1 %. Every short benchmark case is under 400 characters, so it stays one
# section and its number is unaffected.
SECTION_CHARS = 400


def sections(text: str, size: int = SECTION_CHARS) -> list[tuple[int, str]]:
    """Consecutive pieces of at most about `size` characters, cut only at line ends, with their
    offset in the text. Together they are exactly the text."""
    out: list[tuple[int, str]] = []
    start = pos = 0
    for line in text.splitlines(keepends=True):
        if pos > start and pos + len(line) - start > size:
            out.append((start, text[start:pos]))
            start = pos
        pos += len(line)
    out.append((start, text[start:]))
    return out


def _model_spans(part: str, offset: int, llm: ChatModel,
                 person_model: str | None) -> list[tuple[str, Span]]:
    """The three model passes on one section, each find tagged with its pass."""
    found = [("broad", s) for s in detect_with_model(part, llm)]
    found += [("person", s) for s in detect_persons(part, llm, model=person_model)]
    found += [("names", s) for s in detect_names(part, llm)]
    return [(layer, s.model_copy(update={"start": s.start + offset, "end": s.end + offset}))
            for layer, s in found]


def detect(text: str, llm: ChatModel, person_model: str | None = None, judge: bool = False,
           keep: list[str] | None = None) -> list[Span]:
    """All layers must run, on every section. If any model call fails, the error propagates: a
    half detection must never be treated as a complete one (fail closed). With `judge`, Apertus
    then rules on finds only one model pass made (detect/judge.py); `keep` are the organisation's
    rules for what may stay open."""
    parts = sections(text)
    tagged: list[tuple[str, Span]] = [("rule", s) for s in find_rule_spans(text)]
    if len(parts) == 1:
        tagged += _model_spans(text, 0, llm, person_model)
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for found in pool.map(lambda p: _model_spans(p[1], p[0], llm, person_model), parts):
                tagged += found
    tagged = [(layer, strip_article(s)) for layer, s in tagged]
    if judge:
        opened = judge_finds(text, tagged, llm, keep)
        tagged = [(layer, s) for layer, s in tagged
                  if layer == "rule" or s.type == "PERSON" or s.text.casefold() not in opened]
    return resolve_overlaps(type_untyped([s for _, s in tagged]))
