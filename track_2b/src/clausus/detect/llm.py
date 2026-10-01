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


def _locate(text: str, needle: str, span_type: str) -> list[Span]:
    """Every occurrence of the model's answer in the original; nothing if it is not there."""
    needle = needle.strip()
    if len(needle) < 2:
        return []
    return [Span(text=m.group(0), type=span_type, start=m.start(), end=m.end(), source="model")
            for m in re.finditer(re.escape(needle), text, re.IGNORECASE)]


def detect_with_model(text: str, llm: ChatModel) -> list[Span]:
    """Broad pass over all confidential categories."""
    data = extract_json(llm.chat([{"role": "user", "content": PROMPT + text}]))
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
    data = extract_json(llm.chat([{"role": "user", "content": PERSON_PROMPT + text}], model=model))
    names = data.get("personen", []) if isinstance(data, dict) else []
    spans: list[Span] = []
    for name in names:
        if isinstance(name, str):
            spans.extend(_locate(text, name, "PERSON"))
    return spans
