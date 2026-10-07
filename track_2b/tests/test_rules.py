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
        ("Abgabe bis Ende November, sagt sie", ("Ende November", "DATE")),
        ("mir bruuched das bis Änd Oktober", ("Änd Oktober", "DATE")),
        ("Start anfangs Dezember geplant", ("anfangs Dezember", "DATE")),
        ("Budget von 1,2 Millionen genehmigt", ("1,2 Millionen", "AMOUNT")),
        ("Volumen 14,5 Mio. CHF im Jahr", ("14,5 Mio. CHF", "AMOUNT")),
        ("rund 0.8 Millionen Franken", ("0.8 Millionen Franken", "AMOUNT")),
        ("si bruuched füfedriissgtuusig Franke", ("füfedriissgtuusig Franke", "AMOUNT")),
        ("das sind achtzgtuusig Stutz", ("achtzgtuusig Stutz", "AMOUNT")),
        ("Stand vom Projekt Alpenblick heute", ("Alpenblick", "PROJECT")),
        ("Stand des Projekts Nordlicht-2 heute", ("Nordlicht-2", "PROJECT")),
        # Joel's chat 07.10.2026: the model saw these in clear, because the second half of a
        # range and amounts without a currency had no rule, and the 70B pass left them out.
        ("CHF 4'000 bis 7'000", ("7'000", "AMOUNT")),
        ("ohne Import rund 5'800", ("5'800", "AMOUNT")),
        ("plant selbst 10'000 im Jahr ein", ("10'000", "AMOUNT")),
        ("Stufe 1 · Fundament (Q4 2026)", ("Q4 2026", "DATE")),
        ("Start im H1 2027", ("H1 2027", "DATE")),
        ("laut Workshop etwa Ende 2027", ("Ende 2027", "DATE")),
        ("bis Mitte 2028 fertig", ("Mitte 2028", "DATE")),
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


def test_vague_period_words_alone_and_non_amount_millions_are_not_flagged():
    assert found("Am Ende des Meetings und Mitte der Woche.") == set()
    assert found("Es gibt Millionen von Möglichkeiten.") == set()


def test_a_range_is_masked_on_both_sides_and_small_counts_stay():
    spans = found("Schulung aller rund 30 Mitarbeitenden: CHF 4'000 bis 7'000, Version 2.0, Q5 2026")
    assert {("CHF 4'000", "AMOUNT"), ("7'000", "AMOUNT")} <= spans
    assert not {s for s in spans if s[0] in ("30", "2.0") or "Q5" in s[0]}
