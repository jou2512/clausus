"""Spike: a second look at capitalised words the three model passes left open.

The passes miss single names in a long list ("Baar", "Nordlicht"), but a model answers a
yes/no question about one word in its sentence much better than it lists every name in a text.
So rules pick candidates, the fast model only decides. Off unless CLAUSUS_CANDIDATE_CHECK=1.

Candidates are capitalised words that are not frequent German words (wordfreq Zipf below
RARE_ZIPF), plus runs of two or more capitalised words inside a sentence: German capitalises
nouns but not adjectives, so "Kantonalen Bank" mid-sentence is almost always a name.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass

from clausus.detect.llm import (
    FUNCTION_WORDS,
    GENERIC,
    OUTLINE_LABEL,
    TYPE_MAP,
    ChatModel,
    _locate,
    extract_json,
)
from clausus.models import Span

# Zipf 3.5 (about 3 in a million words). Measured 08.10.2026 on a word list: short place and company
# names sit below it (Winterthur 3.48, Olten 3.13, Migros 3.12, Swisscom 2.96), everyday nouns
# above it (Sitzung 4.41, Budget 4.17, Quartal 3.97). Well-known names above it (Müller 4.8,
# Zürich 4.65) are the ones the model passes already find. Words in between (Offerte 3.15) only
# cost a question, the model says no.
RARE_ZIPF = 3.5
_WORD = re.compile(r"[A-ZÄÖÜ][\wäöüÄÖÜß’'-]*")
_SENTENCE_END = re.compile(r"[.!?:;\n]")


@dataclass
class Candidate:
    text: str
    start: int
    end: int
    sentence: str


def _zipf_de(word: str) -> float:
    from wordfreq import zipf_frequency  # only needed when the switch is on

    return zipf_frequency(word, "de")


def _sentence_start(text: str, pos: int) -> bool:
    before = text[:pos].rstrip()
    return not before or bool(_SENTENCE_END.fullmatch(before[-1]))


def _sentence(text: str, start: int, end: int) -> str:
    a = max((m.end() for m in _SENTENCE_END.finditer(text, 0, start)), default=0)
    m = _SENTENCE_END.search(text, end)
    b = m.end() if m else len(text)
    return text[a:b].strip()[:240]


def _runs(text: str) -> list[list[re.Match[str]]]:
    runs: list[list[re.Match[str]]] = []
    for m in _WORD.finditer(text):
        if runs and text[runs[-1][-1].end():m.start()] == " ":
            runs[-1].append(m)
        else:
            runs.append([m])
    return runs


def find_candidates(text: str, covered: list[Span], zipf: Callable[[str], float] | None = None) -> list[Candidate]:
    zipf = zipf or _zipf_de
    out: list[Candidate] = []
    seen: set[str] = set()
    for run in _runs(text):
        if _sentence_start(text, run[0].start()):
            while run and zipf(run[0].group(0)) >= RARE_ZIPF:
                run = run[1:]
        run = [m for m in run if m.group(0).casefold() not in FUNCTION_WORDS]
        if not run or (len(run) == 1 and zipf(run[0].group(0)) >= RARE_ZIPF):
            continue
        start, end = run[0].start(), run[-1].end()
        word = text[start:end]
        if (word.casefold() in GENERIC or word.casefold() in seen or OUTLINE_LABEL.fullmatch(word)
                or any(start < s.end and s.start < end for s in covered)):
            continue
        seen.add(word.casefold())
        out.append(Candidate(word, start, end, _sentence(text, start, end)))
    return out


PROMPT = (
    "Für jedes nummerierte Wort unten: Ist es in seinem Satz der Name einer bestimmten Person, Firma, "
    "Organisation, eines Orts oder eines Projekts? Allgemeine Wörter (Sitzung, Offerte, Bank, Kunde) "
    "sind keine Namen. Die Sätze können Schweizerdeutsch sein. Antworte NUR mit JSON: "
    '{"antworten":[{"nr":1,"name":true,"typ":"PERSON|FIRMA|ORT|PROJEKT"}]}\n\n'
)


@dataclass
class Verdict:
    candidate: Candidate
    name: bool | None  # None: the model gave no readable answer for this one (counted as no)
    type: str = "OTHER"


def ask_candidates(text: str, covered: list[Span], llm: ChatModel, model: str | None = None,
                   zipf: Callable[[str], float] | None = None) -> list[Verdict]:
    """One call for all candidates. A failed call raises (fail closed, like every other pass)."""
    cands = find_candidates(text, covered, zipf)
    if not cands:
        return []
    listing = "\n".join(f'{i}. „{c.text}“ — Satz: {c.sentence}' for i, c in enumerate(cands, 1))
    data = extract_json(llm.chat([{"role": "user", "content": PROMPT + listing}], model=model,
                                 step="detect_candidates"))
    answers = data.get("antworten", []) if isinstance(data, dict) else []
    verdicts = [Verdict(c, None) for c in cands]
    for a in answers:
        if not isinstance(a, dict):
            continue
        try:
            nr = int(a.get("nr", 0))
        except (ValueError, TypeError):
            continue
        if not 1 <= nr <= len(cands):  # nr 0 would silently pick the last candidate
            continue
        verdicts[nr - 1] = Verdict(cands[nr - 1], a.get("name") is True,
                                   TYPE_MAP.get(str(a.get("typ", "")).strip().upper(), "OTHER"))
    return verdicts


def spans_from(text: str, verdicts: list[Verdict]) -> list[Span]:
    return [s for v in verdicts if v.name for s in _locate(text, v.candidate.text, v.type)]


def check_candidates(text: str, covered: list[Span], llm: ChatModel, model: str | None = None,
                     zipf: Callable[[str], float] | None = None) -> list[Span]:
    return spans_from(text, ask_candidates(text, covered, llm, model, zipf))


def enabled() -> bool:
    return os.environ.get("CLAUSUS_CANDIDATE_CHECK", "").strip() == "1"
