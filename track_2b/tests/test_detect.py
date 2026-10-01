import pytest

from clausus.detect import detect
from clausus.detect.llm import detect_with_model, extract_json
from clausus.errors import ClaususError
from clausus.models import Span


class FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls = []

    def chat(self, messages, **kw):
        self.calls.append(messages)
        return self.reply


def test_extract_json_from_fenced_block():
    assert extract_json('Hier:\n```json\n{"stellen": []}\n```') == {"stellen": []}


def test_extract_json_from_bare_object_with_text_around():
    assert extract_json('Antwort: {"stellen": [{"text": "a"}]} fertig') == {"stellen": [{"text": "a"}]}


def test_extract_json_garbage_raises_clausus_error():
    with pytest.raises(ClaususError) as e:
        extract_json("keine Ahnung")
    assert e.value.code == "bad_model_output"


def test_model_spans_get_offsets_from_the_original_and_unknown_text_is_dropped():
    text = "De Reto Ammann vo de Gisler Holzbau GmbH. Später nomol de Reto Ammann."
    llm = FakeLLM('{"stellen":[{"text":"Reto Ammann","typ":"PERSON"},'
                  '{"text":"Gisler Holzbau GmbH","typ":"FIRMA"},'
                  '{"text":"Erfundene AG","typ":"FIRMA"}]}')
    spans = detect_with_model(text, llm)
    got = sorted((s.text, s.type, s.start) for s in spans)
    assert got == [
        ("Gisler Holzbau GmbH", "ORG", text.index("Gisler")),
        ("Reto Ammann", "PERSON", text.index("Reto")),
        ("Reto Ammann", "PERSON", text.rindex("Reto")),
    ]
    assert all(text[s.start:s.end] == s.text for s in spans)
    assert all(s.source == "model" for s in spans)


def test_model_match_is_case_insensitive_but_keeps_original_spelling():
    text = "d'Kundeliste vo de raiffeisen willisau"
    spans = detect_with_model(text, FakeLLM('{"stellen":[{"text":"Raiffeisen Willisau","typ":"FIRMA"}]}'))
    assert [s.text for s in spans] == ["raiffeisen willisau"]


def test_detect_unions_rules_and_model_and_resolves_overlap_by_longer_span():
    text = "Ruf de Reto uf 079 412 33 87 a."
    llm = FakeLLM('{"stellen":[{"text":"Reto","typ":"PERSON"},{"text":"412 33","typ":"TELEFON"}]}')
    spans = detect(text, llm)
    assert [(s.text, s.type) for s in spans] == [("Reto", "PERSON"), ("079 412 33 87", "PHONE")]


class ScriptedLLM:
    """Answers each call from a list, records which model was asked."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.models = []

    def chat(self, messages, model=None, **kw):
        self.models.append(model)
        return self.replies.pop(0)


def test_person_pass_catches_first_name_the_main_pass_missed():
    text = "D'Nadine macht de Termin mit em Reto Ammann."
    llm = ScriptedLLM('{"stellen":[{"text":"Reto Ammann","typ":"PERSON"}]}',
                      '{"personen":["Reto Ammann","Nadine"]}')
    spans = detect(text, llm, person_model="fast-model")
    assert [(s.text, s.type) for s in spans] == [("Nadine", "PERSON"), ("Reto Ammann", "PERSON")]
    assert llm.models == [None, "fast-model"]


def test_person_pass_ignores_names_not_in_text():
    text = "D'Nadine chunnt."
    llm = ScriptedLLM('{"stellen":[]}', '{"personen":["Nadine","Erfundene Person"]}')
    assert [s.text for s in detect(text, llm)] == ["Nadine"]


def test_detect_without_model_raises_rather_than_silently_skipping():
    class Down:
        def chat(self, messages, **kw):
            raise ClaususError("unreachable", "down")

    with pytest.raises(ClaususError):
        detect("De Reto", Down())


def test_span_model_type_validation():
    with pytest.raises(ValueError):
        Span(text="x", type="NONSENSE", start=0, end=1, source="rule")
