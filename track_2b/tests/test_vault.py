from clausus.assemble import assemble
from clausus.models import Span
from clausus.vault import Vault


def spans_for(text, *items):
    out = []
    for t, typ in items:
        i = text.index(t)
        out.append(Span(text=t, type=typ, start=i, end=i + len(t), source="model"))
    return out


def test_full_name_and_surname_get_the_same_placeholder():
    v = Vault()
    a = v.token_for("Thomas Brunner", "PERSON")
    b = v.token_for("Brunner", "PERSON")
    assert a == b == "[PERSON_1]"


def test_ambiguous_first_name_is_not_merged_into_either_person():
    v = Vault()
    v.token_for("Thomas Brunner", "PERSON")
    v.token_for("Thomas Meier", "PERSON")
    assert v.token_for("Thomas", "PERSON") == "[PERSON_3]"


def test_company_suffix_variants_merge():
    v = Vault()
    assert v.token_for("Huber Treuhand AG", "ORG") == v.token_for("Huber Treuhand", "ORG")


def test_different_types_never_share_a_counter_or_token():
    v = Vault()
    assert v.token_for("Luzern", "PLACE") == "[PLACE_1]"
    assert v.token_for("Luzern AG", "ORG") == "[ORG_1]"


def test_pseudonymize_replaces_spans_and_sweeps_later_mentions():
    text = "Thomas Brunner rief an. Später sagte Brunner nein, und brunner nochmals."
    v = Vault()
    masked = v.pseudonymize(text, spans_for(text, ("Thomas Brunner", "PERSON")))
    assert masked == "[PERSON_1] rief an. Später sagte [PERSON_1] nein, und [PERSON_1] nochmals."
    assert "runner" not in masked.lower()


def test_placeholders_are_numbered_in_reading_order():
    text = "Huber Treuhand AG zuerst, dann Meier Bau AG."
    v = Vault()
    masked = v.pseudonymize(text, spans_for(text, ("Huber Treuhand AG", "ORG"), ("Meier Bau AG", "ORG")))
    assert masked == "[ORG_1] zuerst, dann [ORG_2]."


def test_preloaded_known_entities_are_masked_even_without_detection():
    v = Vault()
    v.preload([("Sandra Keller", "PERSON")])
    assert v.pseudonymize("D'Sandra macht s'Kickoff.", []) == "D'[PERSON_1] macht s'Kickoff."


def test_restore_round_trip_is_exact():
    text = "Gisler Holzbau GmbH zahlt CHF 9'000."
    v = Vault()
    spans = spans_for(text, ("Gisler Holzbau GmbH", "ORG"), ("CHF 9'000", "AMOUNT"))
    masked = v.pseudonymize(text, spans)
    assert masked == "[ORG_1] zahlt [AMOUNT_1]."
    assert v.restore(masked) == text


def test_restore_does_not_double_a_word_the_answer_already_put_in_front():
    v = Vault()
    v.token_for("Projekt Nordlicht", "PROJECT")
    assert v.restore("Beim Projekt [PROJECT_1] gab es") == "Beim Projekt Nordlicht gab es"
    assert v.restore("Beim projekt [PROJECT_1]") == "Beim Projekt Nordlicht"
    assert v.restore("Das [PROJECT_1] läuft") == "Das Projekt Nordlicht läuft"


def test_assemble_restores_and_flags_invented_placeholder():
    v = Vault()
    v.token_for("Reto Ammann", "PERSON")
    result = assemble("[PERSON_1] ruft [PERSON_7] an.", v)
    assert result.text == "Reto Ammann ruft [PERSON_7] an."
    assert result.invented == ["[PERSON_7]"]


def test_assemble_reports_placeholders_missing_from_answer():
    v = Vault()
    v.token_for("Reto Ammann", "PERSON")
    v.token_for("Sursee", "PLACE")
    result = assemble("Nur [PLACE_1].", v)
    assert result.missing == ["[PERSON_1]"]
