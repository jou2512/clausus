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


NO_NAMES = '{"namen":[]}'


def test_person_pass_catches_first_name_the_main_pass_missed():
    text = "D'Nadine macht de Termin mit em Reto Ammann."
    llm = ScriptedLLM('{"stellen":[{"text":"Reto Ammann","typ":"PERSON"}]}',
                      '{"personen":["Reto Ammann","Nadine"]}', NO_NAMES)
    spans = detect(text, llm, person_model="fast-model")
    assert [(s.text, s.type) for s in spans] == [("Nadine", "PERSON"), ("Reto Ammann", "PERSON")]
    assert llm.models == [None, "fast-model", None]


def test_person_pass_ignores_names_not_in_text():
    text = "D'Nadine chunnt."
    llm = ScriptedLLM('{"stellen":[]}', '{"personen":["Nadine","Erfundene Person"]}', NO_NAMES)
    assert [s.text for s in detect(text, llm)] == ["Nadine"]


def test_names_pass_catches_place_and_project_the_broad_pass_missed():
    """Measured 02.10.2026 on a slide: the broad pass on 70B left out "Nordlicht" and
    "Winterthur" (2 of 2 runs); a narrow names pass on 70B found both (3 of 3)."""
    text = "Kunde: Meier Bau AG, Winterthur • Pilot Nordlicht"
    llm = ScriptedLLM('{"stellen":[{"text":"Meier Bau AG","typ":"FIRMA"}]}', '{"personen":[]}',
                      '{"namen":[{"text":"Meier Bau AG","typ":"FIRMA"},{"text":"Winterthur","typ":"ORT"},'
                      '"Nordlicht"]}')
    spans = detect(text, llm)
    assert [(s.text, s.type) for s in spans] == [
        ("Meier Bau AG", "ORG"), ("Winterthur", "PLACE"), ("Nordlicht", "OTHER")]


def test_names_pass_drops_generic_words_and_unknown_text():
    text = "Das Team und die Bank treffen sich."
    llm = ScriptedLLM('{"stellen":[]}', '{"personen":[]}', '{"namen":["Team","Bank","Erfunden AG"]}')
    assert detect(text, llm) == []


def test_detect_without_model_raises_rather_than_silently_skipping():
    class Down:
        def chat(self, messages, **kw):
            raise ClaususError("unreachable", "down")

    with pytest.raises(ClaususError):
        detect("De Reto", Down())


def test_span_model_type_validation():
    with pytest.raises(ValueError):
        Span(text="x", type="NONSENSE", start=0, end=1, source="rule")


# --- 07.10.: type from another layer, Swiss German articles, short dates (Joel's live tests) ---

def test_untyped_name_takes_the_type_another_layer_found():
    """"Projekt Nordlicht" came back as OTHER from the names pass (plain list, as measured),
    although the project rule had typed "Nordlicht" PROJECT."""
    text = "Wir schliessen das Projekt Nordlicht ab."
    llm = ScriptedLLM('{"stellen":[]}', '{"personen":[]}', '{"namen":["Projekt Nordlicht"]}')
    assert [(s.text, s.type) for s in detect(text, llm)] == [("Projekt Nordlicht", "PROJECT")]


def test_untyped_name_with_a_company_suffix_is_an_org():
    text = "Die Meier und Söhne Bau AG liefert."
    llm = ScriptedLLM('{"stellen":[]}', '{"personen":[]}', '{"namen":["Meier und Söhne Bau AG"]}')
    assert [(s.text, s.type) for s in detect(text, llm)] == [("Meier und Söhne Bau AG", "ORG")]


def test_swiss_german_article_is_not_part_of_the_name():
    """"Dr Zenhäusern" in Swiss German is "der Zenhäusern"; "de Kellerei" is "die Kellerei"."""
    text = "Dr Zenhäusern vo de Kellerei het aagrüeft."
    llm = ScriptedLLM('{"stellen":[{"text":"Dr Zenhäusern","typ":"PERSON"},{"text":"de Kellerei","typ":"FIRMA"}]}',
                      '{"personen":[]}', '{"namen":[]}')
    assert [(s.text, s.type) for s in detect(text, llm)] == [("Zenhäusern", "PERSON"), ("Kellerei", "ORG")]


def test_a_title_with_a_dot_stays():
    text = "Dr. Lea Zimmermann kommt."
    llm = ScriptedLLM('{"stellen":[{"text":"Dr. Lea Zimmermann","typ":"PERSON"}]}', '{"personen":[]}', '{"namen":[]}')
    assert [s.text for s in detect(text, llm)] == ["Dr. Lea Zimmermann"]


@pytest.mark.parametrize("text,date", [("Vorführung am 2.12. in Bern", "2.12."), ("bis 31.1. fertig", "31.1.")])
def test_short_date_without_year_is_a_rule(text, date):
    llm = ScriptedLLM('{"stellen":[]}', '{"personen":[]}', '{"namen":[]}')
    assert (date, "DATE") in [(s.text, s.type) for s in detect(text, llm)]


def test_a_pronoun_from_any_model_pass_is_not_masked():
    """Seen 07.10.2026 in the chat: the broad pass (70B) returned "sie" as SONSTIGES and the
    person pass (8B) returned "sie" as a person. The vault then masked every later "sie"/"Sie",
    and the answering model asked who [PERSON_1] was."""
    text = "ich will aber, dass sie mehr unterteilt ist, nicht so grob. Er und ich sehen es."
    llm = ScriptedLLM('{"stellen":[{"text":"sie","typ":"SONSTIGES"},{"text":"Er","typ":"PERSON"}]}',
                      '{"personen":["sie","ich"]}', '{"namen":["es"]}')
    assert detect(text, llm) == []


def test_a_model_find_is_matched_as_a_whole_word_only():
    """"Ott" must not mask the start of "Otto"; a genitive s still counts ("Otts Büro")."""
    text = "Otto und Ott treffen sich in Otts Büro."
    spans = detect_with_model(text, FakeLLM('{"stellen":[{"text":"Ott","typ":"PERSON"}]}'))
    assert [(s.text, s.start) for s in spans] == [("Ott", text.index("Ott ")), ("Ott", text.index("Otts"))]
