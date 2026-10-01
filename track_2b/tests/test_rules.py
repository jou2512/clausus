import pytest

from clausus.detect.rules import find_rule_spans


def found(text):
    return {(s.text, s.type) for s in find_rule_spans(text)}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Ruf an: 079 412 33 87 bitte", ("079 412 33 87", "PHONE")),
        ("Nummer +41 79 412 33 87.", ("+41 79 412 33 87", "PHONE")),
        ("kompakt 0794123387 ok", ("0794123387", "PHONE")),
        ("Mail an lea.zimmermann@meier-bau.ch heute", ("lea.zimmermann@meier-bau.ch", "EMAIL")),
        ("IBAN CH93 0076 2011 6238 5295 7 bitte", ("CH93 0076 2011 6238 5295 7", "IBAN")),
        ("AHV 756.1234.5678.97 liegt vor", ("756.1234.5678.97", "AHV")),
        ("Budget CHF 128'500 genehmigt", ("CHF 128'500", "AMOUNT")),
        ("das kostet 45'000 Franken total", ("45'000 Franken", "AMOUNT")),
        ("Offerte bis am 15. Oktober schicken", ("15. Oktober", "DATE")),
        ("Steering am 07.10.2026 in Bern", ("07.10.2026", "DATE")),
    ],
)
def test_swiss_formats_are_found_with_exact_text_and_type(text, expected):
    assert expected in found(text)


def test_offsets_point_at_the_exact_text():
    text = "Hallo, ruf 079 412 33 87 an."
    (span,) = find_rule_spans(text)
    assert text[span.start:span.end] == "079 412 33 87"
    assert span.source == "rule"


def test_plain_numbers_and_words_are_not_flagged():
    assert found("Wir haben 3 Punkte und 12 Teilnehmer, Version zwei.") == set()
