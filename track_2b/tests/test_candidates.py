"""Second look at capitalised words the three passes left open (spike, switch off by default)."""

import json

from clausus.detect.candidates import check_candidates, find_candidates
from clausus.models import Span

FREQ = {"sitzung": 4.4, "offerte": 3.2, "die": 7.0, "am": 6.0, "wir": 6.5, "treffen": 4.5, "bank": 4.6,
        "kantonalen": 3.8, "gestern": 5.0, "uns": 6.0, "mit": 7.0, "der": 7.0}


def zipf(word):
    return FREQ.get(word.casefold(), 0.0)


def texts(cands):
    return [c.text for c in find_candidates(cands, [], zipf)] if isinstance(cands, str) else cands


def test_rare_capitalised_words_and_multiword_runs_are_candidates_frequent_nouns_are_not():
    text = "Wir treffen uns mit der Kantonalen Bank in Oberwil. Die Sitzung war gestern."
    assert texts(text) == ["Kantonalen Bank", "Oberwil"]


def test_a_frequent_word_at_sentence_start_is_not_part_of_a_run():
    assert texts("Die Offerte ging an Gisler.") == ["Offerte", "Gisler"]


def test_words_already_masked_are_no_candidates():
    text = "Gisler kommt nach Oberwil."
    covered = [Span(text="Gisler", type="PERSON", start=0, end=6, source="model")]
    assert [c.text for c in find_candidates(text, covered, zipf)] == ["Oberwil"]


class Echo:
    def __init__(self, reply):
        self.reply, self.prompt, self.kw = reply, "", {}

    def chat(self, messages, **kw):
        self.prompt, self.kw = messages[0]["content"], kw
        return self.reply


def test_one_call_asks_yes_or_no_per_candidate_and_only_yes_becomes_a_span():
    text = "Die Offerte ging an Gisler in Oberwil. Gisler ruft an."
    llm = Echo(json.dumps({"antworten": [{"nr": 1, "name": False}, {"nr": 2, "name": True, "typ": "PERSON"},
                                         {"nr": 3, "name": True, "typ": "ORT"}]}))
    spans = check_candidates(text, [], llm, model="fast", zipf=zipf)
    assert [(s.text, s.type, s.start) for s in spans] == [
        ("Gisler", "PERSON", 20), ("Gisler", "PERSON", 39), ("Oberwil", "PLACE", 30)]
    assert llm.kw["model"] == "fast" and "Oberwil" in llm.prompt and "1." in llm.prompt


def test_no_candidates_means_no_call():
    class Never:
        def chat(self, *a, **k):
            raise AssertionError("called")
    assert check_candidates("Die Sitzung war gestern.", [], Never(), zipf=zipf) == []


def test_an_answer_number_outside_the_list_is_ignored():
    llm = Echo('{"antworten":[{"nr":0,"name":true,"typ":"ORT"},{"nr":9,"name":true}]}')
    assert check_candidates("Treffen in Oberwil.", [], llm, zipf=zipf) == []


def test_detect_runs_the_second_look_only_when_switched_on(monkeypatch):
    from clausus.detect import detect

    class Model:
        def __init__(self):
            self.steps = []

        def chat(self, messages, **kw):
            self.steps.append(kw.get("step"))
            if kw.get("step") == "detect_candidates":
                return '{"antworten":[{"nr":1,"name":true,"typ":"ORT"}]}'
            return {"detect": '{"stellen":[]}', "detect_persons": '{"personen":[]}'}.get(
                kw.get("step"), '{"namen":[]}')

    monkeypatch.setattr("clausus.detect.candidates._zipf_de", zipf)
    off = Model()
    assert detect("Treffen in Oberwil.", off) == [] and "detect_candidates" not in off.steps
    on = Model()
    spans = detect("Treffen in Oberwil.", on, candidate_check=True)
    assert [(s.text, s.type) for s in spans] == [("Oberwil", "PLACE")]
    monkeypatch.setenv("CLAUSUS_CANDIDATE_CHECK", "1")
    assert [s.text for s in detect("Treffen in Oberwil.", Model())] == ["Oberwil"]
