import json

from clausus.eval import Case, Secret, aggregate, load_cases, score_case


def case(text, *secrets, style="mundart"):
    return Case(id="c", style=style, text=text, secrets=[Secret(text=t, type=ty) for t, ty in secrets])


def test_fully_masked_secret_is_not_a_leak():
    c = case("De Reto Ammann chunnt.", ("Reto Ammann", "PERSON"))
    r = score_case(c, "De [PERSON_1] chunnt.", n_placeholder_spans=1, masked_spans=[(3, 14)])
    assert r.leaks == [] and r.partial == []


def test_secret_still_verbatim_is_a_full_leak():
    c = case("De Reto Ammann chunnt.", ("Reto Ammann", "PERSON"))
    r = score_case(c, "De Reto Ammann chunnt.", n_placeholder_spans=0, masked_spans=[])
    assert [s.text for s in r.leaks] == ["Reto Ammann"]


def test_significant_part_left_is_a_partial_leak():
    c = case("Frau Brigitte Aebischer kommt.", ("Brigitte Aebischer", "PERSON"))
    r = score_case(c, "Frau Brigitte [PERSON_1] kommt.", n_placeholder_spans=1, masked_spans=[(14, 23)])
    assert r.leaks == []
    assert [s.text for s in r.partial] == ["Brigitte Aebischer"]


def test_generic_words_left_do_not_count_as_partial_leak():
    c = case("Die Holzbau Rüegsegger AG und das Projekt Seeblick.",
             ("Holzbau Rüegsegger AG", "ORG"), ("Projekt Seeblick", "PROJECT"))
    r = score_case(c, "Die [ORG_1] und das Projekt [PROJECT_1].", n_placeholder_spans=2, masked_spans=[])
    assert r.leaks == [] and r.partial == []


def test_match_is_case_insensitive_and_whole_word():
    c = case("Herr Muri aus Muri-Gries.", ("Muri", "PERSON"))
    assert [s.text for s in score_case(c, "Herr muri aus [PLACE_1].", 1, []).leaks] == ["Muri"]
    # "Murigasse" is not the word "Muri"
    assert score_case(c, "Herr [PERSON_1] an der Murigasse.", 1, []).leaks == []


def test_extra_masking_counts_spans_that_cover_no_secret():
    text = "D'Anna hät gseit, s'Budget isch guet."
    c = case(text, ("Anna", "PERSON"))
    anna = (text.index("Anna"), text.index("Anna") + 4)
    budget = (text.index("Budget"), text.index("Budget") + 6)
    r = score_case(c, "D'[PERSON_1] hät gseit, s'[OTHER_1] isch guet.", 2, [anna, budget])
    assert r.extra_masked == 1


def test_aggregate_reports_rates_overall_per_type_and_per_style():
    a = score_case(case("A Reto B", ("Reto", "PERSON"), style="mundart"), "A [PERSON_1] B", 1, [])
    b = score_case(case("C Bern D 079 412 33 87", ("Bern", "PLACE"), ("079 412 33 87", "PHONE"), style="folie"),
                   "C Bern D [PHONE_1]", 1, [])
    rep = aggregate([a, b])
    assert rep["secrets"] == 3
    assert rep["leaked_full"] == 1 and rep["leaked_any"] == 1
    assert rep["leak_rate"] == round(1 / 3, 4)
    assert rep["by_type"]["PLACE"] == {"secrets": 1, "leaked": 1, "rate": 1.0}
    assert rep["by_style"]["mundart"] == {"secrets": 1, "leaked": 0, "rate": 0.0}
    assert rep["by_style"]["folie"]["leaked"] == 1


def test_load_cases_reads_the_committed_test_set():
    cases = load_cases("data/eval/cases.json")
    assert len(cases) == 24
    assert sum(len(c.secrets) for c in cases) == 167
    for c in cases:
        for s in c.secrets:
            assert s.text in c.text


def test_load_cases_rejects_secret_not_in_text(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"version": 1, "cases": [
        {"id": "x", "style": "folie", "text": "abc", "secrets": [{"text": "zzz", "type": "PERSON"}]}]}),
        encoding="utf-8")
    try:
        load_cases(p)
    except ValueError as e:
        assert "zzz" in str(e)
    else:
        raise AssertionError("bad case accepted")
