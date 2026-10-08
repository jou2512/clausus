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
    r"Oktober|Okt\.|November|Novämber|Dezember|Dezämber|Merz|Auguscht"
)
_THOUSANDS = r"\d{1,3}(?:['’]\d{3})+|\d+"
_CURRENCY = r"Franken|Franke|Franku|Stutz|CHF"
# "tausend" and its dialect forms (Zürich "tuusig", Wallis "tüüsig").
_THOUSAND_WORD = r"(?:tausend|tuusig|tüüsig|tüsig|tausig)"
_SMALL_NUMBER_WORD = (
    r"ei|eis|eine|einer|zwei|zwöi|zwo|drei|drü|drüü|vier|füf|fünf|sechs|sächs|sibe|sieben|"
    r"acht|nün|neun|zäh|zehn|anderthalb|eineinhalb|zweieinhalb"
)
_ORDINAL = r"(?:erst|ersch|zweit|dritt|viert|letzt|letscht|lescht)\w*"

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
    # Day and month without a year: "am 2.12.", "bis 31.1." (Joel's live test 05.10.2026: only the
    # model found "2.12."). Not followed by a digit, so full dates stay with the rule above.
    ("DATE", re.compile(r"\b(?:0?[1-9]|[12]\d|3[01])\.(?:0?[1-9]|1[0-2])\.(?!\d)")),
    # Periods tied to a month: "Ende November", "Änd Oktober", "anfangs Dezember", "Mitte Januar".
    # Benchmark 02.10.2026 (dev set): three of these leaked in every run.
    ("DATE", re.compile(
        rf"\b(?:Anfang|anfangs|Afang|Mitte|Mitti|Ende|Änd|Ändi|Endi)\s+(?:{_MONTHS})\b", re.IGNORECASE)),
    # Amounts with millions or billions: "1,2 Millionen", "14,5 Mio. CHF", "0.8 Millionen Franken".
    ("AMOUNT", re.compile(
        r"\b\d+(?:[.,]\d+)?\s?(?:Mio\.?|Millionen|Millione|Mrd\.?|Milliarden)(?:\s?(?:CHF|Franken|Franke|Fr\.))?",
        re.IGNORECASE)),
    # Spoken thousands in Swiss German or German words: "füfedriissgtuusig Franke", "achtzgtuusig Stutz".
    # A compound ending in a thousand-word is an amount even without a currency: a transcript
    # repeats a figure in words ("CHF 184'000. Also hundert...tausend"). Held-out set 07.10.2026:
    # both spoken forms without a rule leaked. "Tausend" alone and "Tausende" stay.
    ("AMOUNT", re.compile(
        rf"\b[\wäöü]+{_THOUSAND_WORD}\b(?:\s+(?:{_CURRENCY})\b)?", re.IGNORECASE)),
    # Spoken hundreds only with a currency: "achthundert Franken", but not "hundert Prozent".
    ("AMOUNT", re.compile(rf"\b[\wäöü]*hundert[\wäöü]*\s+(?:{_CURRENCY})\b", re.IGNORECASE)),
    # A number word before millions: "zwöi Millione Franke", "eine Million CHF".
    ("AMOUNT", re.compile(
        rf"\b(?:{_SMALL_NUMBER_WORD})\s+(?:Millionen|Millione|Million|Milliarden|Milliarde)"
        rf"(?:\s+(?:{_CURRENCY})\b)?", re.IGNORECASE)),
    # A numbered week in a month: "di zweit Wuche im Merz", "letzten Woche im Oktober". The article
    # stays in clear, as with "Projekt", so the masked text keeps its grammar.
    ("DATE", re.compile(
        rf"\b{_ORDINAL}\s+(?:Woch|Wuch)[\wäöü]*\s+(?:im|in|vom|vo|em)\s+(?:{_MONTHS})\b", re.IGNORECASE)),
    # A number with Swiss thousands separators is an amount even without a currency: the second half
    # of "CHF 4'000 bis 7'000", "rund 5'800". Joel's chat 07.10.2026: the 70B pass left all of these
    # in clear on a long text. Inside "CHF 4'000" the longer currency span wins (resolve_overlaps).
    ("AMOUNT", re.compile(r"(?<![\d'’])\d{1,3}(?:['’]\d{3})+(?:[.,]\d{2})?(?![\d'’])")),
    # Quarters, halves and year periods: "Q4 2026", "H1 2027", "Ende 2027", "Mitte 2028".
    ("DATE", re.compile(r"\b(?:Q[1-4]|H[12])\s?(?:20)?\d{2}\b")),
    ("DATE", re.compile(
        r"\b(?:Anfang|anfangs|Afang|Mitte|Mitti|Ende|Änd|Ändi|Endi)\s+(?:19|20)\d{2}\b", re.IGNORECASE)),
    # The capitalised name right after "Projekt"/"Projekts"/"Project". Only the name is masked:
    # the word itself reveals nothing and keeps the masked text grammatical ("des Projekts [P]").
    ("PROJECT", re.compile(r"(?<=\bProjekt )[A-ZÄÖÜ][\wäöüÄÖÜ-]*")),
    ("PROJECT", re.compile(r"(?<=\bProjekts )[A-ZÄÖÜ][\wäöüÄÖÜ-]*")),
    ("PROJECT", re.compile(r"(?<=\bProject )[A-ZÄÖÜ][\wäöüÄÖÜ-]*")),
    # A capitalised name after Frau/Herr/Herrn is a person; the title stays in clear, like "Projekt".
    # Trap set 08.10.2026: "d'Frau Lüthi" was found by none of the three model passes.
    ("PERSON", re.compile(r"(?<=\bFrau )[A-ZÄÖÜ][\wäöüÄÖÜ-]+")),
    ("PERSON", re.compile(r"(?<=\bHerr )[A-ZÄÖÜ][\wäöüÄÖÜ-]+")),
    ("PERSON", re.compile(r"(?<=\bHerrn )[A-ZÄÖÜ][\wäöüÄÖÜ-]+")),
]


def find_rule_spans(text: str) -> list[Span]:
    spans: list[Span] = []
    for span_type, pattern in PATTERNS:
        for m in pattern.finditer(text):
            spans.append(Span(text=m.group(0), type=span_type, start=m.start(), end=m.end(),
                              source="rule"))
    return spans
