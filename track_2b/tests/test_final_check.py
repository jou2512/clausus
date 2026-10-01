from clausus.final_check import final_check


class Reply:
    def __init__(self, text):
        self.text = text
        self.seen = None

    def chat(self, messages, **kw):
        self.seen = messages[0]["content"]
        return self.text


def test_clean_masked_text_passes():
    llm = Reply('{"treffer": []}')
    result = final_check("[PERSON_1] ruft [ORG_1] an.", llm)
    assert result.clean is True
    assert result.leftovers == []


def test_rule_hit_in_masked_text_fails_without_asking_model_first():
    llm = Reply('{"treffer": []}')
    result = final_check("[PERSON_1], Natel 079 412 33 87.", llm)
    assert result.clean is False
    assert ("rule", "PHONE") in result.leftovers


def test_model_hit_that_is_really_in_the_text_fails():
    result = final_check("Grüess vom Bruno AG an [PERSON_1].", Reply('{"treffer": ["Bruno AG"]}'))
    assert result.clean is False
    assert ("model", "Bruno AG") in result.leftovers


def test_model_hit_inside_a_placeholder_or_absent_is_ignored():
    result = final_check("[PERSON_1] kommt.", Reply('{"treffer": ["PERSON_1", "Erfunden"]}'))
    assert result.clean is True


def test_unreadable_model_answer_counts_as_not_clean():
    result = final_check("[PERSON_1] kommt.", Reply("weiss nicht"))
    assert result.clean is False
    assert result.leftovers == [("model", "unreadable answer")]
