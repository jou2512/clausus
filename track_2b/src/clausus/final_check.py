"""Last check before anything goes to an external model.

Rules run again on the masked text; if they hit, the model is not even asked.
Then Apertus reads the masked text with one question: is anything still
identifiable? Any hit, or an unreadable answer, means: not clean, stay local.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from clausus.detect.llm import ChatModel, extract_json
from clausus.detect.rules import find_rule_spans
from clausus.errors import ClaususError

_TOKEN = re.compile(r"\[[A-Z]+_\d+\]")

PROMPT = (
    "Im folgenden Text sind vertrauliche Angaben bereits durch Platzhalter in eckigen Klammern "
    "ersetzt, zum Beispiel [PERSON_1]. Die Platzhalter zählen nicht. Kann man im übrigen Text "
    "noch eine konkrete Person, Firma, Organisation, ein Projekt, einen Ort, einen Betrag, ein "
    "Datum oder Kontaktangaben erkennen? Antworte NUR mit JSON: "
    '{"treffer":["<wörtlich wie im Text>"]} — eine leere Liste, wenn nichts mehr erkennbar ist.'
    "\n\nTEXT:\n"
)


@dataclass
class FinalCheck:
    clean: bool
    leftovers: list[tuple[str, str]] = field(default_factory=list)


def final_check(masked: str, llm: ChatModel, model: str | None = None) -> FinalCheck:
    rule_hits = [("rule", s.type) for s in find_rule_spans(masked)]
    if rule_hits:
        return FinalCheck(clean=False, leftovers=rule_hits)

    try:
        data = extract_json(llm.chat([{"role": "user", "content": PROMPT + masked}],
                                     model=model, step="final_check"))
    except ClaususError as e:
        if e.code != "bad_model_output":
            raise
        return FinalCheck(clean=False, leftovers=[("model", "unreadable answer")])

    outside_placeholders = _TOKEN.sub(" ", masked).casefold()
    hits = data.get("treffer", []) if isinstance(data, dict) else []
    leftovers = [("model", h) for h in hits
                 if isinstance(h, str) and len(h.strip()) >= 2
                 and h.strip().casefold() in outside_placeholders]
    return FinalCheck(clean=not leftovers, leftovers=leftovers)
