import json

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


class ByPromptLLM:
    """Answers by what the prompt asks and which names its text contains, in any call order
    (long texts are checked in parallel sections)."""

    def __init__(self, persons=(), names=()):
        self.persons, self.names, self.calls = persons, names, 0

    def chat(self, messages, **kw):
        self.calls += 1
        prompt = messages[0]["content"]
        text = prompt.split("TEXT:\n", 1)[1]
        if prompt.startswith("Nenne ALLE Personen"):
            return json.dumps({"personen": [p for p in self.persons if p in text]})
        if prompt.startswith("Nenne im folgenden Text ALLE Namen"):
            return json.dumps({"namen": [n for n in self.names if n in text]})
        return '{"stellen":[]}'


def test_sections_split_on_line_breaks_and_cover_the_whole_text():
    from clausus.detect import sections
    text = "\n".join(f"Zeile {i:02d} " + "x" * 60 for i in range(30))
    parts = sections(text, size=300)
    assert len(parts) > 1
    assert "".join(p for _, p in parts) == text
    assert all(text[o:o + len(p)] == p for o, p in parts)
    assert all(p.endswith("\n") for _, p in parts[:-1])  # cut only at line ends


def test_a_long_text_is_checked_section_by_section_and_offsets_point_into_the_original():
    """Joel's chat 07.10.2026: on 2755 characters the person pass missed "Gisigers"; paragraph by
    paragraph it found it."""
    filler = "Dies ist eine lange Zeile ohne Namen, nur Planung und Pakete.\n" * 12
    text = filler + "Wunsch von Gisiger bei der Planung.\n" + filler + "Rückfrage an Fässler.\n"
    llm = ByPromptLLM(persons=["Gisiger", "Fässler"])
    spans = detect(text, llm)
    assert [(s.text, s.type) for s in spans] == [("Gisiger", "PERSON"), ("Fässler", "PERSON")]
    assert all(text[s.start:s.end] == s.text for s in spans)
    assert llm.calls > 3  # more than one section was asked


def test_a_short_text_is_still_one_section():
    llm = ByPromptLLM(persons=["Nadine"])
    assert [s.text for s in detect("D'Nadine chunnt.", llm)] == ["Nadine"]
    assert llm.calls == 3


def test_a_failing_section_fails_the_whole_detection():
    class FailsOnSecond(ByPromptLLM):
        def chat(self, messages, **kw):
            if "ZWEITER" in messages[0]["content"]:
                raise ClaususError("bad_response", "500")
            return super().chat(messages, **kw)

    text = "erster Teil\n" * 80 + "ZWEITER Teil\n" * 80
    with pytest.raises(ClaususError):
        detect(text, FailsOnSecond())


def test_outline_labels_are_not_names():
    """07.10.2026, long planning text: a section without names came back with "Stufe 2",
    "Paket 3" as confidential, which would hide the structure the answer is about."""
    text = "Stufe 2 · Steuern\nPaket 3 · Kompass bei Treuhand Albis, Phase 1, Modul 4b"
    llm = FakeLLM('{"stellen":[{"text":"Stufe 2","typ":"SONSTIGES"},{"text":"Paket 3","typ":"SONSTIGES"},'
                  '{"text":"Phase 1","typ":"PROJEKT"},{"text":"Modul 4b","typ":"SONSTIGES"},'
                  '{"text":"Treuhand Albis","typ":"FIRMA"}]}')
    assert [s.text for s in detect_with_model(text, llm)] == ["Treuhand Albis"]


# Long Seerose text 08.10.2026: the place "Wil" also masked "wil" (= weil) in "wil üsi Datä nöd suuber
# gsi sind". A lowercase occurrence that is an ordinary word is not the name; other lowercase
# occurrences still are (transcripts are not always capitalised, see the test above).
def test_a_place_name_does_not_mask_the_same_word_written_small_when_that_is_an_ordinary_word():
    text = "Dä Brändli vo Wil zahlt nöd, wil d'Rechnig fehlt. Er gaht ins Büro vo Ins."
    llm = FakeLLM('{"stellen":[{"text":"Wil","typ":"ORT"},{"text":"Ins","typ":"ORT"}]}')
    spans = detect_with_model(text, llm)
    assert [(s.text, text[s.start - 3:s.start]) for s in spans] == [("Wil", "vo "), ("Ins", "vo ")]


def test_bodies_and_roles_without_a_name_are_no_names_in_any_pass():
    """Same text: "Gmeind", "Geschäftsleitung", "Stadthuus" and "Rechenzentrum" came back from the
    broad and the person pass too, not only from the names pass that had a generic list."""
    text = "D'Gmeind wott's im Rechenzentrum, d'Geschäftsleitung chunnt is Stadthuus. Gmeind Uster zahlt."
    llm = FakeLLM('{"stellen":[{"text":"Gmeind","typ":"FIRMA"},{"text":"Rechenzentrum","typ":"ORT"},'
                  '{"text":"Geschäftsleitung","typ":"PERSON"},{"text":"Stadthuus","typ":"ORT"},'
                  '{"text":"Gmeind Uster","typ":"FIRMA"}]}')
    assert [s.text for s in detect_with_model(text, llm)] == ["Gmeind Uster"]



def test_a_broken_unicode_escape_does_not_lose_the_whole_answer():
    r"""Trap set 08.10.2026: the 8B person pass wrote "R\uegg" for "Rüegg" (\u without four hex
    digits). json.loads refused the whole list, and with it the detection of the text failed."""
    data = extract_json(r'{"personen": ["Gasser", "R\uegg", "M\u00fcller"]}')
    assert data["personen"][0] == "Gasser" and data["personen"][2] == "Müller"
