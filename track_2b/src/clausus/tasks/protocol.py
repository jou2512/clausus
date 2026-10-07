"""Meeting protocol from masked meeting material. Placeholders must survive verbatim."""

from __future__ import annotations

from typing import Any

from clausus.detect.llm import ChatModel, extract_json
from clausus.errors import ClaususError

# Cold start 07.10.2026 (transcript plus one slide): with "Meeting-Transkript" and no field for
# slide content, 12 of 12 slide placeholders were missing from the protocol. Naming the slides and
# giving them a "fakten" field: 4 of 12 missing (one run).
PROMPT = (
    "Erstelle aus dem folgenden Meeting-Material ein kurzes Protokoll auf Deutsch. Das Material kann "
    "aus dem gesprochenen Transkript, Folien und Notizen bestehen: berücksichtige ALLE Teile, auch "
    "die Folien. Platzhalter in eckigen Klammern wie [PERSON_1] oder [ORG_2] stehen für vertrauliche "
    "Angaben: übernimm sie WÖRTLICH, erfinde keine neuen und löse sie nicht auf. "
    "Antworte NUR mit JSON in dieser Form: "
    '{"zusammenfassung":"<2-4 Sätze>",'
    '"fakten":["<wichtige Zahl, Termin, Kunde oder Risiko aus Folien und Notizen>"],'
    '"entscheide":["<Entscheid>"],'
    '"pendenzen":[{"was":"<Aufgabe>","wer":"<Person oder Platzhalter>","bis":"<Termin oder leer>"}]}'
    "\n\nMATERIAL:\n"
)


def _strings(items: Any) -> list[str]:
    return [str(x) for x in items if isinstance(x, str | int)] if isinstance(items, list) else []


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
        "fakten": _strings(data.get("fakten", [])),
        "entscheide": _strings(data.get("entscheide", [])),
        "pendenzen": pendenzen,
    }
