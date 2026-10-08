"""The organisation's confidentiality profile: what may stay open, what is always masked, what is
never masked. Joel 08.10.2026: a company documents its own privacy rules; Clausus must make that
cheap (a default, its policy read by Apertus, and what users click while working)."""

import json
import os

import pytest

from clausus.detect.judge import DEFAULT_KEEP
from clausus.errors import ClaususError
from clausus.models import Span
from clausus.profile import Profile, ProfileStore, apply_never_mask, profile_from_policy


def span(text, s, typ="OTHER"):
    i = s.index(text)
    return Span(text=text, type=typ, start=i, end=i + len(text), source="model")


def test_a_new_profile_starts_from_the_default_rules():
    p = Profile()
    assert p.keep == DEFAULT_KEEP and p.never_mask == [] and p.always_mask == []


def test_never_mask_drops_exactly_those_finds_in_any_case_and_nothing_else():
    text = "Abacus und das NDA mit Abacus-Partner"
    spans = [span("Abacus", text, "ORG"), span("NDA", text), span("Abacus-Partner", text, "ORG")]
    kept = apply_never_mask(spans, Profile(never_mask=["abacus", "NDA"]))
    assert [s.text for s in kept] == ["Abacus-Partner"]


def test_persons_are_never_unmasked_by_the_list():
    text = "Rolf kommt"
    assert apply_never_mask([span("Rolf", text, "PERSON")], Profile(never_mask=["Rolf"])) != []


def test_learning_from_clicks_adds_once_and_moves_between_lists():
    p = Profile()
    p.keep_open("Rollout")
    p.keep_open("rollout")
    assert p.never_mask == ["Rollout"]
    p.always("Rollout", "PROJECT")
    assert p.never_mask == [] and p.always_mask == [{"text": "Rollout", "type": "PROJECT"}]


def test_the_profile_is_stored_encrypted_and_reads_back(tmp_path):
    store = ProfileStore(tmp_path, os.urandom(32))
    p = Profile(name="Treuhand Huber", always_mask=[{"text": "Brunnag", "type": "ORG"}], never_mask=["Abacus"])
    store.save(p)
    raw = (tmp_path / "profile.bin").read_bytes()
    assert b"Brunnag" not in raw and b"Abacus" not in raw
    assert store.load() == p


def test_without_a_stored_profile_the_default_is_used(tmp_path):
    assert ProfileStore(tmp_path, os.urandom(32)).load() == Profile()


def test_a_profile_under_another_key_is_refused_not_replaced(tmp_path):
    ProfileStore(tmp_path, os.urandom(32)).save(Profile(name="x"))
    with pytest.raises(ClaususError):
        ProfileStore(tmp_path, os.urandom(32)).load()


class Reply:
    def __init__(self, reply):
        self.reply, self.prompt = reply, ""

    def chat(self, messages, **kw):
        self.prompt = messages[0]["content"]
        return self.reply


def test_apertus_turns_a_written_policy_into_a_proposal():
    llm = Reply(json.dumps({"offen": ["Rollen ohne Namen"], "immer": [{"text": "Brunnag", "typ": "FIRMA"}],
                            "nie": ["Abacus"]}))
    p = profile_from_policy("Kundennamen sind vertraulich, z. B. Brunnag. Software wie Abacus nicht.", llm)
    assert p.keep == ["Rollen ohne Namen"]
    assert p.always_mask == [{"text": "Brunnag", "type": "ORG"}] and p.never_mask == ["Abacus"]
    assert "Brunnag" in llm.prompt


def test_an_unreadable_policy_answer_is_an_error_not_an_empty_profile():
    with pytest.raises(ClaususError):
        profile_from_policy("Richtlinie", Reply("weiss nicht"))
