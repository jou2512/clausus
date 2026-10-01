"""Meeting protocol from a masked transcript. Placeholders must survive verbatim."""

from __future__ import annotations

from typing import Any

from clausus.detect.llm import ChatModel, extract_json
from clausus.errors import ClaususError

PROMPT = (
    "Erstelle aus dem folgenden Meeting-Transkript ein kurzes Protokoll auf Deutsch. "
    "Platzhalter in eckigen Klammern wie [PERSON_1] oder [ORG_2] stehen für vertrauliche "
    "Angaben: übernimm sie WÖRTLICH, erfinde keine neuen und löse sie nicht auf. "
    "Antworte NUR mit JSON in dieser Form: "
    '{"zusammenfassung":"<2-4 Sätze>","entscheide":["<Entscheid>"],'
    '"pendenzen":[{"was":"<Aufgabe>","wer":"<Person oder Platzhalter>","bis":"<Termin oder leer>"}]}'
    "\n\nTRANSKRIPT:\n"
)


def make_protocol(masked_transcript: str, llm: ChatModel, model: str | None = None) -> dict[str, Any]:
    data = extract_json(llm.chat([{"role": "user", "content": PROMPT + masked_transcript}],
                                 model=model, step="protocol"))
    if not isinstance(data, dict):
        raise ClaususError("bad_model_output", "The protocol is not a JSON object.")
    pendenzen = [
        {"was": str(p.get("was", "")), "wer": str(p.get("wer", "")), "bis": str(p.get("bis", ""))}
        for p in data.get("pendenzen", []) if isinstance(p, dict)
    ]
    return {
        "zusammenfassung": str(data.get("zusammenfassung", "")),
        "entscheide": [str(e) for e in data.get("entscheide", []) if isinstance(e, str | int)],
        "pendenzen": pendenzen,
    }
