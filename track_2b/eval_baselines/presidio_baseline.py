"""Baseline: Microsoft Presidio on the same eval sets, scored with Clausus' own score_case/aggregate.

Setup (fair, documented): presidio-analyzer with the largest German spaCy model (de_core_news_lg),
all predefined recognizers loaded for German, default score threshold. Every detected span is
masked. Run with the presidio venv:
    python eval_baselines/presidio_baseline.py data/eval/holdout.json   (from track_2b/)
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from clausus.eval import aggregate, load_cases, score_case  # noqa: E402
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry  # noqa: E402
from presidio_analyzer.nlp_engine import NlpEngineProvider  # noqa: E402

conf = {"nlp_engine_name": "spacy", "models": [{"lang_code": "de", "model_name": "de_core_news_lg"}]}
nlp = NlpEngineProvider(nlp_configuration=conf).create_engine()
registry = RecognizerRegistry(supported_languages=["de"])
registry.load_predefined_recognizers(languages=["de"], nlp_engine=nlp)
analyzer = AnalyzerEngine(nlp_engine=nlp, registry=registry, supported_languages=["de"])

active = sorted({(r.name, tuple(r.supported_entities)) for r in registry.get_recognizers("de", all_fields=True)})
print("active recognizers for de:")
for name, ents in active:
    print(f"  {name}: {', '.join(ents)}")


def mask(text: str) -> tuple[str, list[tuple[int, int]], dict[str, int]]:
    results = analyzer.analyze(text=text, language="de")
    spans = sorted({(r.start, r.end) for r in results})
    merged: list[list[int]] = []
    for a, b in spans:  # merge overlaps so replacement offsets stay valid
        if merged and a < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out = text
    for a, b in reversed(merged):
        out = out[:a] + "[X]" + out[b:]
    types: dict[str, int] = {}
    for r in results:
        types[r.entity_type] = types.get(r.entity_type, 0) + 1
    return out, [tuple(m) for m in merged], types


for path in sys.argv[1:]:
    cases = load_cases(path)
    t0 = time.perf_counter()
    results, found = [], {}
    for c in cases:
        masked, spans, types = mask(c.text)
        for k, v in types.items():
            found[k] = found.get(k, 0) + v
        results.append(score_case(c, masked, len(spans), spans))
    rep = aggregate(results)
    secs = time.perf_counter() - t0
    print(f"\n== {Path(path).name}: leak rate {rep['leak_rate']*100:.1f}% "
          f"({rep['leaked_any']} of {rep['secrets']}, full {rep['leaked_full']}, partial {rep['leaked_partial']}), "
          f"extra masked {rep['extra_masked']}, {secs:.1f} s")
    print("  entity types Presidio reported:", dict(sorted(found.items())))
    for k, v in rep["by_style"].items():
        print(f"  style {k:24s} {v['leaked']:3d} of {v['secrets']:3d} leaked ({v['rate']*100:.1f}%)")
    for k, v in rep["by_type"].items():
        print(f"  type  {k:24s} {v['leaked']:3d} of {v['secrets']:3d} leaked ({v['rate']*100:.1f}%)")
    out = Path.cwd() / f"presidio-{Path(path).stem}.json"
    out.write_text(json.dumps({"tool": "presidio-analyzer 2.2.364 + de_core_news_lg 3.8.0",
                               "active_recognizers": [a for a, _ in active], "report": rep,
                               "examples": [{"case": r.case.id, "masked": r.masked_text,
                                             "leaks": [s.text for s in r.leaks],
                                             "partial": [s.text for s in r.partial]} for r in results]},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print("  report:", out)
