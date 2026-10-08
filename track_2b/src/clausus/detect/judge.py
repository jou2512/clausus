"""Layer 5 of detection: Apertus as judge over what only one model pass found.

Analysis 08.10.2026 (long Seerose text, each placeholder traced to its pass, two runs): about two
thirds of the unneeded placeholders (NDA, "nüüni", "Fritig", "Rahmeabkomme") came from a single
pass. So did one real person ("Brändli", only the person pass). A majority vote would leak him,
so a single-pass find is not dropped; it is judged. Persons are never judged: a missed person is
the worst leak there is. The judge can only take a placeholder away when it says "offen"; an
unsure or unreadable verdict keeps everything masked (over-masking is the safe direction).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from clausus.detect.llm import ChatModel, extract_json
from clausus.errors import ClaususError
from clausus.models import Span

log = logging.getLogger("clausus.judge")

# What an organisation usually lets stay in clear. A profile can replace this list.
DEFAULT_KEEP = [
    "Rollen, Gremien und Abteilungen ohne Namen (Geschäftsleitung, Einkauf, Vorstand)",
    "Wochentage, Uhrzeiten und ungefähre Zeitangaben (Fritig, um nüüni, nächste Woche)",
    "Fachbegriffe, Dokument- und Sitzungsbezeichnungen (Offerte, Rollout, NDA, Protokoll, Steering)",
    "Allgemein bekannte Software und Dienste, wenn sie nur als Werkzeug genannt werden (Excel, Teams)",
]

JUDGE_PROMPT = (
    "Du prüfst für einen Datenschutzfilter einzelne Wörter aus einem Meeting-Text, bevor er an ein "
    "Sprachmodell geht. Vertraulich ist, was Personen, Kunden, Partner, Projekte, Orte, Beträge oder "
    "konkrete Termine erkennbar macht. Nach den Regeln dieser Organisation dürfen offen bleiben:\n"
    "{keep}\n\n"
    "Entscheide für jedes Wort im Kontext seines Satzes: \"offen\" nur, wenn es klar unter diese Regeln "
    "fällt oder nichts und niemanden erkennbar macht; \"maskieren\", wenn es vertraulich sein könnte; "
    "sonst \"unsicher\". Im Zweifel nicht \"offen\". Antworte NUR mit JSON: "
    '{{"urteile":[{{"text":"<Wort wie angegeben>","entscheid":"offen|maskieren|unsicher"}}]}}'
    "\n\nWÖRTER:\n"
)

_SENTENCE_END = re.compile(r"[.!?\n]")


def sentence_around(text: str, start: int, end: int) -> str:
    left = max((m.end() for m in _SENTENCE_END.finditer(text, 0, start)), default=0)
    right = _SENTENCE_END.search(text, end)
    return text[left:right.end() if right else len(text)].strip()


def candidates(tagged: list[tuple[str, Span]]) -> dict[str, Span]:
    """Model finds (not persons) that no other layer backs: neither the same text found by another
    layer anywhere nor an overlapping span of another layer. Keyed by the casefolded text."""
    by_text: dict[str, set[str]] = {}
    for layer, s in tagged:
        by_text.setdefault(s.text.casefold(), set()).add(layer)
    out: dict[str, Span] = {}
    for layer, s in tagged:
        if layer == "rule" or s.type == "PERSON":
            continue
        overlapping = {other for other, t in tagged if other != layer and t.start < s.end and s.start < t.end}
        if overlapping or by_text[s.text.casefold()] != {layer}:
            continue
        out.setdefault(s.text.casefold(), s)
    return out


def judge(text: str, tagged: list[tuple[str, Span]], llm: ChatModel,
          keep: list[str] | None = None, model: str | None = None) -> set[str]:
    """Casefolded texts the judge says may stay open. Asks only when there are candidates."""
    found = candidates(tagged)
    if not found:
        return set()
    rules = "\n".join(f"- {k}" for k in (keep or DEFAULT_KEEP))
    lines = [f'{i}. "{s.text}" im Satz: "{sentence_around(text, s.start, s.end)}"'
             for i, s in enumerate(found.values(), 1)]
    prompt = JUDGE_PROMPT.format(keep=rules) + "\n".join(lines)
    try:
        data: Any = extract_json(llm.chat([{"role": "user", "content": prompt}], model=model, step="judge"))
    except ClaususError as e:
        # A judge that cannot answer takes nothing away: everything found stays masked. Unlike a
        # failed detection pass, this cannot leak, so the request does not fail.
        log.warning("judge unavailable (%s): every find stays masked", e.code)
        return set()
    verdicts = data.get("urteile", []) if isinstance(data, dict) else []
    return {str(v.get("text", "")).strip().casefold() for v in verdicts
            if isinstance(v, dict) and str(v.get("entscheid", "")).strip().lower() == "offen"
            and str(v.get("text", "")).strip().casefold() in found}
