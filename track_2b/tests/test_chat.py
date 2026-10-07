"""Chat over several messages: one vault per session, the model sees the masked history only."""

import base64
import json
import os

import httpx
from fastapi.testclient import TestClient

from clausus.app import create_app
from clausus.config import Settings

PERSON = '{"stellen":[{"text":"Anna Keller","typ":"PERSON"}]}'
NONE = '{"stellen":[]}'
EMPTY_P, EMPTY_N = '{"personen":[]}', '{"namen":[]}'


def make(tmp_path, replies, down=False):
    seen = []

    def apertus(req):
        if down:
            raise httpx.ConnectError("down")
        body = json.loads(req.content)
        seen.append(body)
        return httpx.Response(200, json={"choices": [{"message": {"content": replies.pop(0)}}]})

    env = {"LLM_BASE_URL": "https://a.test/v1", "LLM_API_KEY": "k", "CLAUSUS_RUNS_DIR": str(tmp_path / "runs"),
           "CLAUSUS_VAULT_DIR": str(tmp_path / "vault"),
           "CLAUSUS_VAULT_KEY": base64.b64encode(os.urandom(32)).decode()}
    app = create_app(settings=Settings.from_env(env),
                     http_client=httpx.Client(transport=httpx.MockTransport(apertus)))
    return TestClient(app), seen


def test_two_messages_share_one_vault_and_the_model_sees_only_masked_history(tmp_path):
    client, seen = make(tmp_path, [PERSON, EMPTY_P, EMPTY_N, "Klar, ich rufe [PERSON_1] an.",
                                   NONE, EMPTY_P, EMPTY_N, "[PERSON_1] erreichst du am Morgen."])
    sid = client.post("/chat/sessions").json()["session_id"]
    a = client.post(f"/chat/sessions/{sid}/messages", json={"text": "Ruf bitte Anna Keller an."}).json()
    assert a["answer"] == "Klar, ich rufe Anna Keller an."
    b = client.post(f"/chat/sessions/{sid}/messages", json={"text": "Wann erreiche ich Keller?"}).json()
    assert b["masked"] == "Wann erreiche ich [PERSON_1]?"  # same person, same placeholder
    assert b["answer"] == "Anna Keller erreichst du am Morgen."
    history = json.dumps(seen[-1]["messages"], ensure_ascii=False)
    assert "Ralph" not in history and "Keller" not in history
    assert "Ruf bitte [PERSON_1] an." in history and "Klar, ich rufe [PERSON_1] an." in history


def test_session_reads_back_and_deletes(tmp_path):
    client, _ = make(tmp_path, [PERSON, EMPTY_P, EMPTY_N, "ok [PERSON_1]"])
    sid = client.post("/chat/sessions").json()["session_id"]
    client.post(f"/chat/sessions/{sid}/messages", json={"text": "Anna Keller ruft an."})
    s = client.get(f"/chat/sessions/{sid}").json()
    assert [m["role"] for m in s["messages"]] == ["user", "assistant"]
    assert s["messages"][0]["original"] == "Anna Keller ruft an."
    assert s["vault"][0]["original"] == "Anna Keller"
    listed = client.get("/chat/sessions").json()["sessions"]
    assert listed[0]["id"] == sid and listed[0]["messages"] == 2
    assert client.delete(f"/chat/sessions/{sid}").status_code == 200
    assert client.get(f"/chat/sessions/{sid}").status_code == 404


def test_nothing_on_disk_in_the_clear(tmp_path):
    client, _ = make(tmp_path, [PERSON, EMPTY_P, EMPTY_N, "ok [PERSON_1]"])
    sid = client.post("/chat/sessions").json()["session_id"]
    client.post(f"/chat/sessions/{sid}/messages", json={"text": "Anna Keller ruft an."})
    raw = b"".join(p.read_bytes() for p in (tmp_path / "vault").rglob("*") if p.is_file())
    assert b"Keller" not in raw
    receipts = b"".join(p.read_bytes() for p in (tmp_path / "runs").glob("*.json"))
    assert b"Keller" not in receipts and b"[PERSON_1]" in receipts


def test_failed_detection_stores_nothing(tmp_path):
    client, _ = make(tmp_path, [], down=True)
    sid = client.post("/chat/sessions").json()["session_id"]
    r = client.post(f"/chat/sessions/{sid}/messages", json={"text": "Anna Keller ruft an."})
    assert r.status_code == 503 and "Ralph" not in r.text
    assert client.get(f"/chat/sessions/{sid}").json()["messages"] == []


def test_unknown_session_is_404(tmp_path):
    client, seen = make(tmp_path, [])
    assert client.post("/chat/sessions/s-0000000000000000/messages", json={"text": "x"}).status_code == 404
    assert client.get("/chat/sessions/../../etc").status_code == 404
    assert seen == []
