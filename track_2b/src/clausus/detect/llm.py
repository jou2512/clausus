"""Layer 2 of detection: Apertus finds what needs understanding.

Names, companies, projects, places and dialect forms have no fixed pattern.
The model only says WHAT is confidential; WHERE it sits is computed here from
the original text. Anything the model names that is not literally in the text
is dropped, never "replaced" (it would be an invention).
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from clausus.errors import ClaususError
from clausus.models import Span

PROMPT = (
    "Du bist ein Datenschutzfilter. Finde im folgenden Text ALLE vertraulichen Stellen: "
    "Personen (auch nur Vor- oder Nachname), Firmen und Organisationen, Projektnamen, Orte, "
    "Beträge, Daten, Telefonnummern, E-Mail-Adressen. Der Text kann Schweizerdeutsch, "
    "Hochdeutsch oder Englisch sein. Achtung Schweizerdeutsch: Namen stehen oft mit Artikel "
    "(d'Anna, de Marco, em Peter, vom Müller) — ein Vorname allein ist auch eine Person; gib "
    "dann nur den Namen ohne Artikel zurück. Gib jede Stelle GENAU so zurück, wie sie im Text steht, "
    "ohne sie zu übersetzen oder zu korrigieren. Antworte NUR mit JSON in dieser Form: "
    '{"stellen":[{"text":"<wörtlich>","typ":"PERSON|FIRMA|PROJEKT|ORT|BETRAG|DATUM|TELEFON|'
    'EMAIL|SONSTIGES"}]}\n\nTEXT:\n'
)

TYPE_MAP = {
    "PERSON": "PERSON", "FIRMA": "ORG", "ORGANISATION": "ORG", "ORG": "ORG",
    "PROJEKT": "PROJECT", "ORT": "PLACE", "BETRAG": "AMOUNT", "DATUM": "DATE",
    "TELEFON": "PHONE", "EMAIL": "EMAIL", "E-MAIL": "EMAIL", "IBAN": "IBAN", "AHV": "AHV",
}


class ChatModel(Protocol):
    def chat(self, messages: list[dict[str, Any]], **kw: Any) -> str: ...


def extract_json(reply: str) -> Any:
    """Models wrap JSON in ``` fences or add prose around it (seen with Apertus 70B)."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", reply, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    start, end = reply.find("{"), reply.rfind("}")
    if start != -1 and end > start:
        candidates.append(reply[start:end + 1])
    for c in candidates:
        try:
            return json.loads(c)
        except ValueError:
            continue
    raise ClaususError("bad_model_output", "The model did not return readable JSON.")


PERSON_PROMPT = (
    "Nenne ALLE Personen im folgenden Text, auch wenn nur ein Vorname dasteht. Im "
    "Schweizerdeutschen steht vor Namen oft ein Artikel (d'Anna, de Marco) — gib dann nur den "
    "Namen zurück, genau so geschrieben wie im Text. Antworte NUR mit JSON: "
    '{"personen":["<Name wie im Text>"]}\n\nTEXT:\n'
)


# Pronouns and articles never name anyone, yet a model returns them (07.10.2026, chat: "sie" from
# the 70B broad pass and the 8B person pass). Masked once, the vault masks every later "Sie" too.
FUNCTION_WORDS = {
    "ich", "du", "er", "sie", "es", "wir", "ihr", "man", "mir", "mich", "dir", "dich", "uns", "euch",
    "ihm", "ihn", "ihnen", "ihre", "ihrer", "ihren", "sein", "seine", "seiner", "der", "die", "das",
    "dem", "den", "des", "ein", "eine", "einer", "einem", "einen", "dieser", "diese", "dieses",
    "si", "är", "mer", "öich", "ihne", "en", "e", "i", "he", "she", "it", "we", "they", "you", "me",
    "him", "her", "us", "them", "his", "their", "the", "a", "an",
}


# "Stufe 2", "Paket 3b": a section of a long planning text without names came back with its
# outline labels as confidential (07.10.2026). They name a place in the document, not a party.
OUTLINE_LABEL = re.compile(
    r"(?:Stufe|Paket|Phase|Modul|Schritt|Teil|Kapitel|Punkt|Abschnitt|Step|Stage|Package|Version)"
    r"\s*\d+[a-z]?", re.IGNORECASE)


# Words a model sometimes returns as "names" although they name nothing specific: generic nouns,
# and bodies, roles and public buildings without a name. Long Seerose text 08.10.2026: "Gmeind",
# "Geschäftsleitung", "Stadthuus" and "Rechenzentrum" came back from all three passes, so the list
# applies to every pass. "Gmeind Uster" or "Stadthaus Uster" still count, they carry a name.
GENERIC = {"kunde", "kunden", "team", "bank", "projekt", "firma", "filiale", "büro", "buero",
           "firmen", "unternehmen", "gemeinde", "stadt", "kanton", "abteilung", "verwaltung",
           "gmeind", "gmeinde", "gemeinderat", "gmeindrat", "stadtrat", "geschäftsleitung",
           "gschäftsleitig", "geschäftsführung", "verwaltungsrat", "vorstand", "stadthaus",
           "stadthuus", "gemeindehaus", "rathaus", "rechenzentrum"}

# Places and first names that are an ordinary word when written small: "wil" (= weil), "ins", "au"
# (= auch), "baden", "will". Long Seerose text 08.10.2026: the place "Wil" masked "wil üsi Datä".
# Other names still match in lower case, because transcripts are not always capitalised.
ORDINARY_WHEN_SMALL = {"wil", "ins", "au", "baden", "rain", "bad", "will", "rosa"}


def is_ordinary_small(name: str, found: str) -> bool:
    """A capitalised name found written small, where the small form is an ordinary word."""
    return name[:1].isupper() and found.islower() and found.casefold() in ORDINARY_WHEN_SMALL


def _locate(text: str, needle: str, span_type: str) -> list[Span]:
    """Every whole-word occurrence of the model's answer in the original; nothing if it is not
    there. A trailing genitive s still counts ("Otts Büro"), a longer word does not ("Otto")."""
    needle = needle.strip()
    if (len(needle) < 2 or needle.casefold() in FUNCTION_WORDS or needle.casefold() in GENERIC
            or OUTLINE_LABEL.fullmatch(needle)):
        return []
    pattern = re.compile(rf"(?<!\w){re.escape(needle)}(?=s?(?!\w))", re.IGNORECASE)
    return [Span(text=m.group(0), type=span_type, start=m.start(), end=m.end(), source="model")
            for m in pattern.finditer(text) if not is_ordinary_small(needle, m.group(0))]


def detect_with_model(text: str, llm: ChatModel) -> list[Span]:
    """Broad pass over all confidential categories."""
    data = extract_json(llm.chat([{"role": "user", "content": PROMPT + text}], step="detect"))
    items = data.get("stellen", []) if isinstance(data, dict) else []
    spans: dict[tuple[int, int], Span] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        span_type = TYPE_MAP.get(str(item.get("typ", "")).strip().upper(), "OTHER")
        for s in _locate(text, str(item.get("text", "")), span_type):
            spans[(s.start, s.end)] = s
    return list(spans.values())


def detect_persons(text: str, llm: ChatModel, model: str | None = None) -> list[Span]:
    """Narrow pass for people only. Measured 02.10.2026: the broad pass on Apertus 70B
    missed a lone Swiss German first name ("D'Nadine") every time at temperature 0;
    this narrow pass found it on both 70B and 8B."""
    data = extract_json(llm.chat([{"role": "user", "content": PERSON_PROMPT + text}], model=model,
                                 step="detect_persons"))
    names = data.get("personen", []) if isinstance(data, dict) else []
    spans: list[Span] = []
    for name in names:
        if isinstance(name, str):
            spans.extend(_locate(text, name, "PERSON"))
    return spans


# Wording exactly as measured on 02.10.2026. A variant that also asked for the type of each
# name ({"text","typ"}) shifted the blind spots on the same slide (found Swisscom, missed
# Nordlicht and Winterthur), so the plain list is kept and found names are typed OTHER.
# Changes to this prompt need a fresh measurement on the test set (make eval), not one slide.
NAMES_PROMPT = (
    "Nenne im folgenden Text ALLE Namen von Firmen, Organisationen, Projekten und Orten "
    "(Städte, Gemeinden, Quartiere, Filialen). Gib sie genau so zurück, wie sie im Text stehen, "
    "ohne Artikel. Allgemeine Wörter wie Kunde, Team, Bank oder Projekt allein sind keine Namen. "
    'Antworte NUR mit JSON: {"namen":["<wie im Text>"]}\n\nTEXT:\n'
)

def detect_names(text: str, llm: ChatModel, model: str | None = None) -> list[Span]:
    """Narrow pass for organisations, projects and places. Measured 02.10.2026 on a slide:
    the broad pass on 70B left out "Nordlicht" and "Winterthur" (2 of 2 runs); this pass
    on 70B found both (3 of 3) but missed "Swisscom", which the broad pass caught. The
    layers have different blind spots, which is why both run."""
    data = extract_json(llm.chat([{"role": "user", "content": NAMES_PROMPT + text}], model=model,
                                 step="detect_names"))
    items = data.get("namen", []) if isinstance(data, dict) else []
    spans: list[Span] = []
    for item in items:
        if isinstance(item, str):
            needle, span_type = item, "OTHER"
        elif isinstance(item, dict):
            needle = str(item.get("text", ""))
            span_type = TYPE_MAP.get(str(item.get("typ", "")).strip().upper(), "OTHER")
        else:
            continue
        spans.extend(_locate(text, needle, span_type))
    return spans
