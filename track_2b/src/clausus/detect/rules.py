"""Layer 1 of detection: fixed rules for everything with a clear pattern.

Rules are cheap and deterministic, so anything that has a format (phone, e-mail,
IBAN, AHV number, amounts, dates) is caught here before the model is asked.
Over-masking is the safe direction (SPEC section 4), so patterns lean wide.
"""

from __future__ import annotations

import re

from clausus.models import Span

_MONTHS = (
    r"Januar|Jänner|Februar|März|Maerz|April|Mai|Juni|Juli|August|September|"
    r"Oktober|Okt\.|November|Novämber|Dezember|Dezämber"
)
_THOUSANDS = r"\d{1,3}(?:['’]\d{3})+|\d+"

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("IBAN", re.compile(r"\bCH\d{2}(?:\s?[0-9A-Z]{4}){4}\s?[0-9A-Z]\b")),
    ("AHV", re.compile(r"\b756\.\d{4}\.\d{4}\.\d{2}\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+41\s?\d{2}|0\d{2})\s?\d{3}\s?\d{2}\s?\d{2}(?!\d)")),
    ("AMOUNT", re.compile(rf"\b(?:CHF|SFr\.?|Fr\.)\s?(?:{_THOUSANDS})(?:[.,]\d{{2}}|\.[-–])?")),
    ("AMOUNT", re.compile(
        r"\b\d{1,3}(?:['’]\d{3})+(?:[.,]\d{2})?\s?(?:CHF|Franken|Franke|Fr\.)",
        re.IGNORECASE)),
    ("DATE", re.compile(rf"\b\d{{1,2}}\.\s?(?:{_MONTHS})(?:\s\d{{4}})?", re.IGNORECASE)),
    ("DATE", re.compile(r"\b(?:0?[1-9]|[12]\d|3[01])\.(?:0?[1-9]|1[0-2])\.(?:\d{4}|\d{2})\b")),
    # Periods tied to a month: "Ende November", "Änd Oktober", "anfangs Dezember", "Mitte Januar".
    # Benchmark 02.10.2026 (dev set): three of these leaked in every run.
    ("DATE", re.compile(
        rf"\b(?:Anfang|anfangs|Afang|Mitte|Mitti|Ende|Änd|Ändi|Endi)\s+(?:{_MONTHS})\b", re.IGNORECASE)),
    # Amounts with millions or billions: "1,2 Millionen", "14,5 Mio. CHF", "0.8 Millionen Franken".
    ("AMOUNT", re.compile(
        r"\b\d+(?:[.,]\d+)?\s?(?:Mio\.?|Millionen|Millione|Mrd\.?|Milliarden)(?:\s?(?:CHF|Franken|Franke|Fr\.))?",
        re.IGNORECASE)),
    # Spoken thousands in Swiss German or German words: "füfedriissgtuusig Franke", "achtzgtuusig Stutz".
    ("AMOUNT", re.compile(r"\b[\wäöü]+(?:tuusig|tausend)\s+(?:Franken|Franke|Stutz|CHF)\b", re.IGNORECASE)),
    # The capitalised name right after "Projekt"/"Projekts"/"Project". Only the name is masked:
    # the word itself reveals nothing and keeps the masked text grammatical ("des Projekts [P]").
    ("PROJECT", re.compile(r"(?<=\bProjekt )[A-ZÄÖÜ][\wäöüÄÖÜ-]*")),
    ("PROJECT", re.compile(r"(?<=\bProjekts )[A-ZÄÖÜ][\wäöüÄÖÜ-]*")),
    ("PROJECT", re.compile(r"(?<=\bProject )[A-ZÄÖÜ][\wäöüÄÖÜ-]*")),
]


def find_rule_spans(text: str) -> list[Span]:
    spans: list[Span] = []
    for span_type, pattern in PATTERNS:
        for m in pattern.finditer(text):
            spans.append(Span(text=m.group(0), type=span_type, start=m.start(), end=m.end(),
                              source="rule"))
    return spans
