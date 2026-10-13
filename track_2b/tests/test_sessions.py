"""Chat sessions: vault and history, encrypted at rest (AES-256-GCM)."""

import base64
import os

import pytest

from clausus.errors import ClaususError
from clausus.models import Span
from clausus.sessions import SessionStore, load_or_create_key
from clausus.vault import Vault


def key() -> bytes:
    return os.urandom(32)


def test_vault_round_trips_through_dict():
    v = Vault()
    v.pseudonymize("Thomas Brunner und Brunner.", [Span(text="Thomas Brunner", type="PERSON", start=0, end=14, source="model")])
    w = Vault.from_dict(v.to_dict())
    assert w.restore("[PERSON_1]") == "Thomas Brunner"
    assert w.token_for("Anna Meier", "PERSON") == "[PERSON_2]"  # numbering continues


def test_session_file_holds_no_plain_text(tmp_path):
    store = SessionStore(tmp_path, key())
    s = store.create()
    s.vault.token_for("Anna Keller", "PERSON")
    s.messages.append({"role": "user", "original": "Ruf Anna Keller an", "masked": "Ruf [PERSON_1] an"})
    store.save(s)
    raw = b"".join(p.read_bytes() for p in tmp_path.rglob("*") if p.is_file())
    assert b"Anna" not in raw and b"Keller" not in raw and b"PERSON_1" not in raw


def test_session_loads_back_with_the_same_key(tmp_path):
    k = key()
    store = SessionStore(tmp_path, k)
    s = store.create()
    s.vault.token_for("Anna Keller", "PERSON")
    s.messages.append({"role": "user", "original": "x", "masked": "y"})
    store.save(s)
    t = SessionStore(tmp_path, k).load(s.id)
    assert t.vault.restore("[PERSON_1]") == "Anna Keller"
    assert t.messages == s.messages


def test_wrong_key_cannot_read(tmp_path):
    store = SessionStore(tmp_path, key())
    s = store.create()
    store.save(s)
    with pytest.raises(ClaususError) as e:
        SessionStore(tmp_path, key()).load(s.id)
    assert e.value.code == "session_unreadable"


def test_tampered_file_is_refused(tmp_path):
    k = key()
    store = SessionStore(tmp_path, k)
    s = store.create()
    store.save(s)
    path = next(p for p in tmp_path.rglob("*.bin"))
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(ClaususError):
        store.load(s.id)


def test_list_and_delete(tmp_path):
    store = SessionStore(tmp_path, key())
    a, b = store.create(), store.create()
    store.save(a)
    store.save(b)
    assert {x["id"] for x in store.list()} == {a.id, b.id}
    store.delete(a.id)
    assert {x["id"] for x in store.list()} == {b.id}
    with pytest.raises(ClaususError) as e:
        store.load(a.id)
    assert e.value.code == "session_not_found"


@pytest.mark.parametrize("bad", ["../etc/passwd", "x", "s-ZZZ"])
def test_session_ids_are_checked(tmp_path, bad):
    with pytest.raises(ClaususError) as e:
        SessionStore(tmp_path, key()).load(bad)
    assert e.value.code == "session_not_found"


def test_key_from_env_or_created_once(tmp_path):
    k = base64.b64encode(os.urandom(32)).decode()
    assert load_or_create_key(k, tmp_path) == base64.b64decode(k)
    created = load_or_create_key("", tmp_path)
    assert len(created) == 32
    assert load_or_create_key("", tmp_path) == created  # same key on the next start
    with pytest.raises(ClaususError):
        load_or_create_key("too-short", tmp_path)
