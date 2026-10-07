"""The protocol covers the slides too, not only what was said."""

import json

from clausus.tasks.protocol import make_protocol


class Echo:
    def __init__(self, reply):
        self.reply, self.prompt = reply, ""

    def chat(self, messages, **kw):
        self.prompt = messages[0]["content"]
        return self.reply


def test_facts_from_slides_get_their_own_field():
    """Cold start 07.10.2026: transcript plus one slide, and 12 of 12 slide placeholders were missing
    from the protocol; it only had room for summary, decisions and to-dos of the spoken part."""
    llm = Echo(json.dumps({"zusammenfassung": "s", "fakten": ["Budget Phase 2: [AMOUNT_2]", 7],
                           "entscheide": [], "pendenzen": []}))
    p = make_protocol("Gesprochen.\n\nFolien:\nBudget Phase 2: [AMOUNT_2]", llm)
    assert p["fakten"] == ["Budget Phase 2: [AMOUNT_2]", "7"]
    assert "Folien" in llm.prompt and '"fakten"' in llm.prompt


def test_a_protocol_without_facts_still_has_the_field():
    p = make_protocol("x", Echo('{"zusammenfassung":"s","entscheide":[],"pendenzen":[]}'))
    assert p["fakten"] == []
