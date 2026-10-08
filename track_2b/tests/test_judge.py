"""Apertus as judge over finds only one model pass made (analysis 08.10.2026, Seerose text: about two
thirds of the unneeded placeholders came from a single pass, but so did the person "Brändli")."""

import json

from clausus.detect import detect
from clausus.detect.judge import DEFAULT_KEEP


class Calls:
    """Answers by step: the three detection passes, then the judge. Records the judge prompt."""

    def __init__(self, broad, persons, names, judge=None):
        self.replies = {"detect": broad, "detect_persons": persons, "detect_names": names, "judge": judge}
        self.steps, self.judge_prompt = [], ""

    def chat(self, messages, model=None, step=None, **kw):
        self.steps.append(step)
        if step == "judge":
            self.judge_prompt = messages[0]["content"]
        return self.replies[step]


TEXT = "Bis Fritig unterschreibt d'Huber Treuhand AG s'NDA. Dä Brändli bringt's am Mäntig."
BROAD = json.dumps({"stellen": [{"text": "Huber Treuhand AG", "typ": "FIRMA"},
                                {"text": "NDA", "typ": "SONSTIGES"}, {"text": "Fritig", "typ": "DATUM"}]})
PERSONS = json.dumps({"personen": ["Brändli"]})
NAMES = json.dumps({"namen": ["Huber Treuhand AG"]})


def texts(spans):
    return [s.text for s in spans]


def test_a_single_pass_find_the_judge_calls_open_is_not_masked():
    judge = json.dumps({"urteile": [{"text": "NDA", "entscheid": "offen"},
                                    {"text": "Fritig", "entscheid": "offen"}]})
    llm = Calls(BROAD, PERSONS, NAMES, judge)
    assert texts(detect(TEXT, llm, judge=True)) == ["Huber Treuhand AG", "Brändli"]


def test_only_open_unmasks_masked_and_unsure_stay_masked():
    judge = json.dumps({"urteile": [{"text": "NDA", "entscheid": "unsicher"},
                                    {"text": "Fritig", "entscheid": "maskieren"}]})
    spans = detect(TEXT, Calls(BROAD, PERSONS, NAMES, judge), judge=True)
    assert texts(spans) == ["Fritig", "Huber Treuhand AG", "NDA", "Brändli"]


def test_persons_and_finds_of_two_passes_never_go_to_the_judge():
    llm = Calls(BROAD, PERSONS, NAMES, json.dumps({"urteile": []}))
    detect(TEXT, llm, judge=True)
    asked = llm.judge_prompt.split("WÖRTER:")[1]
    assert '"NDA" im Satz' in asked and '"Fritig" im Satz' in asked
    assert '"Brändli" im Satz' not in asked and '"Huber Treuhand AG" im Satz' not in asked


def test_the_judge_sees_the_sentence_and_the_organisations_rules():
    llm = Calls(BROAD, PERSONS, NAMES, json.dumps({"urteile": []}))
    detect(TEXT, llm, judge=True, keep=["Sitzungsbezeichnungen", "Wochentage"])
    assert "Bis Fritig unterschreibt" in llm.judge_prompt
    assert "Sitzungsbezeichnungen" in llm.judge_prompt and "Wochentage" in llm.judge_prompt


def test_an_unreadable_verdict_unmasks_nothing():
    spans = detect(TEXT, Calls(BROAD, PERSONS, NAMES, "kein json"), judge=True)
    assert texts(spans) == ["Fritig", "Huber Treuhand AG", "NDA", "Brändli"]


def test_without_candidates_or_switched_off_the_judge_is_not_asked():
    only_shared = json.dumps({"stellen": [{"text": "Huber Treuhand AG", "typ": "FIRMA"}]})
    llm = Calls(only_shared, PERSONS, NAMES)
    detect(TEXT, llm, judge=True)
    assert "judge" not in llm.steps
    llm = Calls(BROAD, PERSONS, NAMES)
    detect(TEXT, llm)
    assert "judge" not in llm.steps


def test_default_rules_name_what_may_stay_open():
    joined = " ".join(DEFAULT_KEEP)
    assert "Rollen" in joined and "Wochentage" in joined and "Fachbegriffe" in joined
