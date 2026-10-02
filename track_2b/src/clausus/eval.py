"""Leak-rate benchmark on the invented test set (data/eval/cases.json).

What is measured: the text that would leave the house after masking (detection +
vault, exactly as /redact does it). For every marked secret:
  - full leak:    the secret still appears verbatim (case-insensitive, whole word)
  - partial leak: a significant part of it still appears ("Brigitte" of "Brigitte Aebischer")
The headline number counts both. Generic words ("AG", "Projekt", "CHF", month names)
left standing are not leaks: they identify nothing.

Also measured, so that masking everything cannot win: spans masked that cover no secret.

Usage:  python -m clausus.eval --cases data/eval/cases.json [--runs 3] [--out runs]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

GENERIC = {
    "ag", "gmbh", "sa", "sàrl", "kg", "ltd", "inc", "&", "und", "and", "de", "der", "die", "das", "vo",
    "von", "vom", "em", "dr", "frau", "herr", "projekt", "projekts", "project", "chf", "fr", "franken",
    "franke", "millionen", "millione", "tuusig", "komma", "filiale", "kunde", "januar", "februar",
    "märz", "april", "mai", "juni", "juli", "august", "september", "oktober", "november", "dezember",
    "novämber", "dezämber", "ende", "änd", "ändi",
}


@dataclass(frozen=True)
class Secret:
    text: str
    type: str


@dataclass
class Case:
    id: str
    style: str
    text: str
    secrets: list[Secret]


@dataclass
class CaseResult:
    case: Case
    masked_text: str
    leaks: list[Secret] = field(default_factory=list)
    partial: list[Secret] = field(default_factory=list)
    extra_masked: int = 0
    placeholders: int = 0


def load_cases(path: str | Path) -> list[Case]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = []
    for c in data["cases"]:
        secrets = [Secret(text=s["text"], type=s["type"]) for s in c["secrets"]]
        for s in secrets:
            if s.text not in c["text"]:
                raise ValueError(f"case {c['id']}: secret {s.text!r} is not in the text")
        cases.append(Case(id=c["id"], style=c["style"], text=c["text"], secrets=secrets))
    return cases


def _present(needle: str, text: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", text, re.IGNORECASE) is not None


def _significant_parts(secret: str) -> list[str]:
    parts = [p.strip(".,;:()") for p in re.findall(r"[\w'’.@+-]+", secret)]
    return [p for p in parts if len(p) >= 3 and p.casefold() not in GENERIC]


def _occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(re.escape(needle), text, re.IGNORECASE)]


def score_case(case: Case, masked_text: str, n_placeholder_spans: int,
               masked_spans: list[tuple[int, int]]) -> CaseResult:
    r = CaseResult(case=case, masked_text=masked_text, placeholders=n_placeholder_spans)
    for s in case.secrets:
        if _present(s.text, masked_text):
            r.leaks.append(s)
        elif any(_present(p, masked_text) for p in _significant_parts(s.text)):
            r.partial.append(s)
    secret_ranges = [o for s in case.secrets for o in _occurrences(case.text, s.text)]
    r.extra_masked = sum(
        1 for a, b in masked_spans
        if not any(a < sb and b > sa for sa, sb in secret_ranges)
    )
    return r


def _rate(n: int, d: int) -> float:
    return round(n / d, 4) if d else 0.0


def aggregate(results: list[CaseResult]) -> dict:
    total = sum(len(r.case.secrets) for r in results)
    full = sum(len(r.leaks) for r in results)
    part = sum(len(r.partial) for r in results)
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"secrets": 0, "leaked": 0})
    by_style: dict[str, dict[str, int]] = defaultdict(lambda: {"secrets": 0, "leaked": 0})
    for r in results:
        leaked = {id(s) for s in r.leaks + r.partial}
        for s in r.case.secrets:
            for bucket in (by_type[s.type], by_style[r.case.style]):
                bucket["secrets"] += 1
                bucket["leaked"] += id(s) in leaked
    finish = lambda d: {k: {**v, "rate": _rate(v["leaked"], v["secrets"])} for k, v in sorted(d.items())}
    return {
        "cases": len(results),
        "secrets": total,
        "leaked_full": full,
        "leaked_partial": part,
        "leaked_any": full + part,
        "leak_rate": _rate(full + part, total),
        "full_leak_rate": _rate(full, total),
        "extra_masked": sum(r.extra_masked for r in results),
        "placeholders": sum(r.placeholders for r in results),
        "by_type": finish(by_type),
        "by_style": finish(by_style),
    }


def run_once(cases: list[Case], llm, person_model: str | None) -> list[CaseResult]:
    from clausus.detect import detect
    from clausus.vault import Vault

    results = []
    for c in cases:
        vault = Vault()
        spans = detect(c.text, llm, person_model=person_model)
        masked = vault.pseudonymize(c.text, spans)
        results.append(score_case(c, masked, len(spans), [(s.start, s.end) for s in spans]))
    return results


def main(argv: list[str] | None = None) -> int:
    from clausus.config import Settings
    from clausus.llm import ApertusClient

    ap = argparse.ArgumentParser(description="Clausus leak-rate benchmark")
    ap.add_argument("--cases", default="data/eval/cases.json")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    settings = Settings.from_env()
    llm = ApertusClient(settings)
    cases = load_cases(args.cases)
    started = time.monotonic()
    runs = []
    for i in range(args.runs):
        results = run_once(cases, llm, settings.fast_model)
        rep = aggregate(results)
        rep["leaks"] = [{"case": r.case.id, "style": r.case.style, "full": [s.text for s in r.leaks],
                         "partial": [s.text for s in r.partial]} for r in results if r.leaks or r.partial]
        runs.append(rep)
        print(f"run {i + 1}/{args.runs}: leak rate {rep['leak_rate']:.1%} "
              f"({rep['leaked_any']} of {rep['secrets']}, full {rep['leaked_full']}, "
              f"partial {rep['leaked_partial']}), extra masked {rep['extra_masked']}")
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": settings.llm_name, "fast_model": settings.fast_model,
        "cases_file": args.cases, "runs": runs,
        "seconds": round(time.monotonic() - started, 1),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"eval-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    last = runs[-1]
    print("\nby style:")
    for k, v in last["by_style"].items():
        print(f"  {k:<24} {v['leaked']:>3} of {v['secrets']:<3} leaked  ({v['rate']:.1%})")
    print("by type:")
    for k, v in last["by_type"].items():
        print(f"  {k:<24} {v['leaked']:>3} of {v['secrets']:<3} leaked  ({v['rate']:.1%})")
    print(f"\nreport: {path}  ({report['seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
