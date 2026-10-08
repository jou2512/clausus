"""The profile through the app: clicks change what gets masked next time, the policy becomes a proposal."""

import base64
import json
import os

import httpx
from fastapi.testclient import TestClient

from clausus.app import create_app
from clausus.config import Settings

TEXT = "D'Brunnag AG unterschribt s'NDA am Fritig."
BROAD = json.dumps({"stellen": [{"text": "NDA", "typ": "SONSTIGES"}, {"text": "Fritig", "typ": "DATUM"}]})
EMPTY_P, EMPTY_N = '{"personen":[]}', '{"namen":[]}'


def make(tmp_path, replies):
    def apertus(req):
        return httpx.Response(200, json={"choices": [{"message": {"content": replies.pop(0)}}]})

    env = {"LLM_BASE_URL": "https://a.test/v1", "LLM_API_KEY": "k", "CLAUSUS_RUNS_DIR": str(tmp_path / "runs"),
           "CLAUSUS_VAULT_DIR": str(tmp_path / "vault"),
           "CLAUSUS_VAULT_KEY": base64.b64encode(os.urandom(32)).decode()}
    app = create_app(settings=Settings.from_env(env),
                     http_client=httpx.Client(transport=httpx.MockTransport(apertus)))
    return TestClient(app)


def test_masking_without_a_profile_uses_the_defaults_and_writes_no_file(tmp_path):
    client = make(tmp_path, [BROAD, EMPTY_P, EMPTY_N])
    assert client.get("/profile").json()["never_mask"] == []
    r = client.post("/redact", json={"text": TEXT})
    assert r.status_code == 200 and "[OTHER_1]" in r.json()["masked_text"]
    assert not (tmp_path / "vault" / "profile.bin").exists()


def test_a_do_not_mask_click_and_an_always_mask_click_change_the_next_run(tmp_path):
    client = make(tmp_path, [BROAD, EMPTY_P, EMPTY_N])
    client.post("/profile/keep", json={"text": "NDA"})
    p = client.post("/profile/always", json={"text": "Brunnag AG", "type": "ORG"}).json()
    assert p["never_mask"] == ["NDA"] and p["always_mask"] == [{"text": "Brunnag AG", "type": "ORG"}]
    masked = client.post("/redact", json={"text": TEXT}).json()["masked_text"]
    assert "NDA" in masked and "Brunnag" not in masked and "Fritig" not in masked
    raw = (tmp_path / "vault" / "profile.bin").read_bytes()
    assert b"Brunnag" not in raw


def test_a_policy_becomes_a_proposal_that_is_not_saved_until_confirmed(tmp_path):
    reply = json.dumps({"offen": ["Rollen ohne Namen"], "immer": [{"text": "Brunnag", "typ": "FIRMA"}], "nie": []})
    client = make(tmp_path, [reply])
    r = client.post("/profile/from-policy", json={"policy": "Kunden wie Brunnag sind vertraulich."}).json()
    assert r["proposal"]["always_mask"] == [{"text": "Brunnag", "type": "ORG"}]
    assert client.get("/profile").json()["always_mask"] == []
    client.put("/profile", json=r["proposal"])
    assert client.get("/profile").json()["keep"] == ["Rollen ohne Namen"]


def test_the_vault_table_lists_only_what_this_text_masked(tmp_path):
    """UI check 08.10.2026: always-mask names that do not occur showed up in the playground table."""
    client = make(tmp_path, [BROAD, EMPTY_P, EMPTY_N, "ok"])
    client.post("/profile/always", json={"text": "Muster Bau GmbH", "type": "ORG"})
    b = client.post("/playground/run", json={"text": TEXT}).json()
    assert "Muster Bau GmbH" not in [r["original"] for r in b["vault"]]


def test_do_not_mask_applies_to_the_next_message_of_a_running_chat(tmp_path):
    client = make(tmp_path, [BROAD, EMPTY_P, EMPTY_N, "ok", BROAD, EMPTY_P, EMPTY_N, "ok"])
    sid = client.post("/chat/sessions").json()["session_id"]
    first = client.post(f"/chat/sessions/{sid}/messages", json={"text": TEXT}).json()
    assert "NDA" not in first["masked"]
    client.post("/profile/keep", json={"text": "NDA"})
    second = client.post(f"/chat/sessions/{sid}/messages", json={"text": TEXT}).json()
    assert "NDA" in second["masked"]
