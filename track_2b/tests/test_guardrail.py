"""LiteLLM Generic Guardrail API contract (docs.litellm.ai/docs/adding_provider/generic_guardrail_api,
read 02.10.2026): POST /beta/litellm_basic_guardrail_api with texts, input_type request|response,
litellm_call_id; answer {"action": NONE|BLOCKED|GUARDRAIL_INTERVENED, "texts", "blocked_reason"}."""

import json

import httpx
from fastapi.testclient import TestClient

from clausus.app import create_app
from clausus.config import Settings
from clausus.guardrail import GuardrailStore

URL = "/beta/litellm_basic_guardrail_api"
DETECT = '{"stellen":[{"text":"Thomas Brunner","typ":"PERSON"},{"text":"Huber Treuhand AG","typ":"FIRMA"}]}'


def make(tmp_path, replies=None, env_extra=None, down=False, store=None):
    calls = []

    def apertus(req):
        if down:
            raise httpx.ConnectError("down")
        calls.append(json.loads(req.content)["messages"][0]["content"])
        reply = replies.pop(0) if replies else '{"stellen":[],"personen":[],"namen":[]}'
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    env = {"LLM_BASE_URL": "https://a.test/v1", "LLM_API_KEY": "k", "CLAUSUS_RUNS_DIR": str(tmp_path),
           **(env_extra or {})}
    app = create_app(settings=Settings.from_env(env),
                     http_client=httpx.Client(transport=httpx.MockTransport(apertus)),
                     guardrail_store=store)
    return TestClient(app), calls


def body(texts, input_type, call_id="call-1", **extra):
    return {"texts": texts, "input_type": input_type, "litellm_call_id": call_id, **extra}


def test_request_masks_all_texts_with_consistent_placeholders(tmp_path):
    client, _ = make(tmp_path, [DETECT, '{"personen":[]}', '{"namen":[]}',
                                '{"stellen":[]}', '{"personen":[]}', '{"namen":[]}'])
    r = client.post(URL, json=body(["Thomas Brunner von der Huber Treuhand AG ruft an.",
                                    "Bitte Brunner zurückrufen."], "request"))
    assert r.status_code == 200
    out = r.json()
    assert out["action"] == "GUARDRAIL_INTERVENED"
    assert out["texts"] == ["[PERSON_1] von der [ORG_1] ruft an.", "Bitte [PERSON_1] zurückrufen."]


def test_response_restores_names_and_forgets_the_vault(tmp_path):
    client, _ = make(tmp_path, [DETECT, '{"personen":[]}', '{"namen":[]}'])
    client.post(URL, json=body(["Thomas Brunner von der Huber Treuhand AG ruft an."], "request"))
    r = client.post(URL, json=body(["Antwort an [PERSON_1] von [ORG_1]."], "response"))
    assert r.json() == {"action": "GUARDRAIL_INTERVENED",
                        "texts": ["Antwort an Thomas Brunner von Huber Treuhand AG."]}
    again = client.post(URL, json=body(["[PERSON_1]"], "response"))
    assert again.json()["action"] == "NONE"


def test_unknown_call_id_on_response_changes_nothing(tmp_path):
    client, _ = make(tmp_path)
    assert client.post(URL, json=body(["hallo"], "response", call_id="nope")).json()["action"] == "NONE"


def test_nothing_confidential_is_none(tmp_path):
    client, _ = make(tmp_path)
    assert client.post(URL, json=body(["Wie wird das Wetter?"], "request")).json()["action"] == "NONE"


def test_detection_failure_blocks_and_stores_nothing(tmp_path):
    store = GuardrailStore()
    client, _ = make(tmp_path, down=True, store=store)
    r = client.post(URL, json=body(["Thomas Brunner"], "request"))
    assert r.status_code == 200
    assert r.json()["action"] == "BLOCKED"
    assert "Thomas" not in r.text
    assert len(store) == 0


def test_images_are_blocked_without_calling_the_model(tmp_path):
    client, calls = make(tmp_path)
    r = client.post(URL, json=body(["schau dir das an"], "request", images=["aGVsbG8="]))
    assert r.json()["action"] == "BLOCKED"
    assert calls == []


def test_vault_expires_after_ttl(tmp_path):
    now = [1000.0]
    store = GuardrailStore(ttl_s=600, clock=lambda: now[0])
    client, _ = make(tmp_path, [DETECT, '{"personen":[]}', '{"namen":[]}'], store=store)
    client.post(URL, json=body(["Thomas Brunner ruft an."], "request"))
    now[0] += 601
    assert client.post(URL, json=body(["[PERSON_1]"], "response")).json()["action"] == "NONE"
    assert len(store) == 0


def test_shared_key_is_enforced_when_configured(tmp_path):
    client, _ = make(tmp_path, env_extra={"CLAUSUS_GUARDRAIL_KEY": "s3cret"})
    assert client.post(URL, json=body(["x"], "request")).status_code == 401
    assert client.post(URL, json=body(["x"], "request"), headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post(URL, json=body(["x"], "request"), headers={"x-api-key": "wrong"}).status_code == 401
    ok = client.post(URL, json=body(["x"], "request"), headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200


def test_litellm_sends_the_key_as_x_api_key(tmp_path):
    # LiteLLM's generic_guardrail_api turns `api_key` into this header (source read 02.10.2026).
    client, _ = make(tmp_path, env_extra={"CLAUSUS_GUARDRAIL_KEY": "s3cret"})
    assert client.post(URL, json=body(["x"], "request"), headers={"x-api-key": "s3cret"}).status_code == 200


def test_request_writes_a_receipt_without_originals(tmp_path):
    client, _ = make(tmp_path, [DETECT, '{"personen":[]}', '{"namen":[]}'])
    client.post(URL, json=body(["Thomas Brunner von der Huber Treuhand AG ruft an."], "request"))
    files = list(tmp_path.glob("guardrail-*.json"))
    assert len(files) == 1
    rec = files[0].read_text(encoding="utf-8")
    assert "Brunner" not in rec and "Huber" not in rec
    assert '"detect"' in rec
