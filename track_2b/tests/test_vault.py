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


def test_ambiguous_short_form_is_masked_with_its_own_placeholder_not_left_open():
    """Benchmark 02.10.2026: with two Marcos, a lone "Marco" stayed in the text 3 of 3 runs.
    Ambiguous means: do not merge it into either person. It must still be masked."""
    text = "Marco Bernasconi und Marco Keller kommen. Marco bringt die Unterlagen, marco auch."
    v = Vault()
    spans = spans_for(text, ("Marco Bernasconi", "PERSON"), ("Marco Keller", "PERSON"))
    masked = v.pseudonymize(text, spans)
    assert "marco" not in masked.lower()
    assert masked.startswith("[PERSON_1] und [PERSON_2] kommen. [PERSON_3] bringt")
    assert masked.endswith("[PERSON_3] auch.")


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


def test_the_same_text_keeps_its_placeholder_even_when_a_later_pass_types_it_differently():
    """Joel's chat 07.10.2026: "Brunnag" was [ORG_1]; a later "brunnag" came back from the model as a
    project and got [PROJECT_2], so one company had two placeholders."""
    from clausus.models import Span
    v = Vault()
    assert v.token_for("Brunnag", "ORG") == "[ORG_1]"
    text = "das brunnag sieht"
    masked = v.pseudonymize(text, [Span(text="brunnag", type="PROJECT", start=4, end=11, source="model")])
    assert masked == "das [ORG_1] sieht"
    assert v.tokens == [("[ORG_1]", "ORG")]


# Numbers the model wrote itself. It only ever sees masked text, so a figure in its answer that was
# not in what it was sent cannot come from the meeting: it is invented or computed. It passes
# restore untouched and would read like a fact, so it is reported.
def test_a_number_the_model_made_up_is_reported():
    from clausus.assemble import unsourced_numbers
    sent = "[PERSON_1] will [AMOUNT_1] für 30 Mitarbeitende, Version 2.0."
    answer = "[PERSON_1] plant CHF 50'000 für 30 Leute, also rund 1'667 pro Kopf, bis 15.11.2026."
    assert unsourced_numbers(answer, [sent]) == ["CHF 50'000", "1'667", "15.11.2026"]


def test_numbers_from_the_input_placeholders_and_small_counts_are_not_reported():
    from clausus.assemble import unsourced_numbers
    sent = "Budget [AMOUNT_1], Team von 12 Personen, 5'800 im Jahr, Version 2.0."
    answer = "1. [PERSON_3] übernimmt [AMOUNT_1]. 2. Die 12 Personen, 5800 im Jahr, Version 2.0, drei Punkte."
    assert unsourced_numbers(answer, [sent]) == []


def test_a_spoken_amount_the_model_made_up_is_reported_too():
    from clausus.assemble import unsourced_numbers
    assert unsourced_numbers("Rechne mit zehntausend Franken.", ["Budget [AMOUNT_1]."]) == ["zehntausend Franken"]


def test_the_instruction_counts_as_a_source():
    from clausus.assemble import unsourced_numbers
    assert unsourced_numbers("Hier 150 Wörter.", ["Text", "Antworte in 150 Wörtern"]) == []


def test_the_sweep_leaves_an_ordinary_small_word_that_looks_like_a_known_place():
    text = "Dä vo Wil zahlt, wil d'Rechnig stimmt. Z'Wil isch's schön."
    v = Vault()
    masked = v.pseudonymize(text, spans_for(text, ("Wil", "PLACE")))
    assert masked == "Dä vo [PLACE_1] zahlt, wil d'Rechnig stimmt. Z'[PLACE_1] isch's schön."


def test_a_project_found_with_its_label_is_masked_later_without_it():
    """Trap set 08.10.2026: "Projekt Kleinhünige" was masked, a later "Kleinhünige zahlt" leaked."""
    text = "Dr Go-live vom Projekt Kleinhünige isch verschobe. Kleinhünige zahlt spöter."
    v = Vault()
    masked = v.pseudonymize(text, spans_for(text, ("Projekt Kleinhünige", "PROJECT")))
    assert "Kleinhünige" not in masked and masked.count("[PROJECT_1]") == 2


def test_a_company_found_with_its_trade_words_is_masked_later_by_its_name():
    """Same set: "Alpenblick Consulting GmbH" was masked, "die Offerte von Alpenblick" leaked."""
    text = "Daniel vom Partner Alpenblick Consulting GmbH meint ja. Die Offerte von Alpenblick kommt."
    v = Vault()
    masked = v.pseudonymize(text, spans_for(text, ("Alpenblick Consulting GmbH", "ORG")))
    assert "Alpenblick" not in masked and masked.count("[ORG_1]") == 2


def test_a_company_name_made_of_ordinary_words_does_not_spread_to_them():
    text = "Die Schweizerische Post liefert. Die schweizerische Lösung ist gut. Gemeinde Uster, gemeinde."
    v = Vault()
    masked = v.pseudonymize(text, spans_for(text, ("Schweizerische Post", "ORG"), ("Gemeinde Uster", "ORG")))
    assert "schweizerische Lösung" in masked and masked.endswith("gemeinde.")


def test_a_word_released_by_the_organisation_is_not_swept_even_if_the_vault_knows_it():
    """UI check 08.10.2026: after "Don't mask" on "Zellweger Logistik", the next chat message still
    masked it, because the session vault knew it from an earlier message."""
    text = "Zellweger Logistik bringt's. Rolf bringt's."
    v = Vault()
    v.token_for("Zellweger Logistik", "ORG")
    v.token_for("Rolf", "PERSON")
    masked = v.pseudonymize(text, [], keep_open={"zellweger logistik", "rolf"})
    assert masked == "Zellweger Logistik bringt's. [PERSON_1] bringt's."



def test_a_company_with_a_generic_word_in_front_is_still_found_by_its_name():
    """Trap set 08.10.2026: "Partner Alpenblick Consulting GmbH" was masked, "von Alpenblick" leaked."""
    text = "Daniel vom Partner Alpenblick Consulting GmbH meint ja. Die Offerte von Alpenblick kommt."
    v = Vault()
    masked = v.pseudonymize(text, spans_for(text, ("Partner Alpenblick Consulting GmbH", "ORG")))
    assert "Alpenblick" not in masked
