import json

import httpx
from fastapi.testclient import TestClient

from clausus.app import create_app, split_original
from clausus.config import Settings

TEXT = "Thomas Brunner von der Huber Treuhand AG ruft an. Brunner will CHF 180'000."
DETECT = '{"stellen":[{"text":"Thomas Brunner","typ":"PERSON"},{"text":"Huber Treuhand AG","typ":"FIRMA"}]}'
ANSWER = "Rückruf an [PERSON_1] ([ORG_1]), Budget [AMOUNT_1]. Zuständig: [NICHT GENANNT]."


def make(tmp_path, replies):
    seen = []

    def apertus(req):
        seen.append(json.loads(req.content)["messages"][-1]["content"])
        return httpx.Response(200, json={"choices": [{"message": {"content": replies.pop(0)}}]})

    env = {"LLM_BASE_URL": "https://a.test/v1", "LLM_API_KEY": "k", "CLAUSUS_RUNS_DIR": str(tmp_path)}
    app = create_app(settings=Settings.from_env(env),
                     http_client=httpx.Client(transport=httpx.MockTransport(apertus)))
    return TestClient(app), seen


def test_playground_masks_asks_and_restores(tmp_path):
    client, seen = make(tmp_path, [DETECT, '{"personen":[]}', '{"namen":[]}', ANSWER])
    r = client.post("/playground/run", json={"text": TEXT})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["masked"] == "[PERSON_1] von der [ORG_1] ruft an. [PERSON_1] will [AMOUNT_1]."
    # the answering model only ever saw the masked text
    assert "Brunner" not in seen[-1] and "[PERSON_1]" in seen[-1]
    assert b["answer_masked"] == ANSWER
    assert b["answer"] == "Rückruf an Thomas Brunner (Huber Treuhand AG), Budget CHF 180'000. Zuständig: [NICHT GENANNT]."
    assert b["invented"] == ["[NICHT GENANNT]"]
    vault = {m["token"]: (m["type"], m["original"]) for m in b["vault"]}
    assert vault["[PERSON_1]"] == ("PERSON", "Thomas Brunner")
    assert vault["[AMOUNT_1]"][1] == "CHF 180'000"
    # what was really said, piece by piece: "Brunner" alone stays "Brunner", not the canonical name
    segs = b["original_segments"]
    assert "".join(s["text"] for s in segs) == TEXT
    assert [(s["text"], s["token"]) for s in segs if s["token"]] == [
        ("Thomas Brunner", "[PERSON_1]"), ("Huber Treuhand AG", "[ORG_1]"),
        ("Brunner", "[PERSON_1]"), ("CHF 180'000", "[AMOUNT_1]")]


def test_playground_receipt_has_no_originals_and_lists_the_answer_call(tmp_path):
    client, _ = make(tmp_path, [DETECT, '{"personen":[]}', '{"namen":[]}', ANSWER])
    b = client.post("/playground/run", json={"text": TEXT}).json()
    rec = json.loads((tmp_path / f"{b['receipt_id']}.json").read_text(encoding="utf-8"))
    assert ("answer", "masked") in [(o["step"], o["kind"]) for o in rec["outgoing"]]
    assert "Brunner" not in json.dumps(rec, ensure_ascii=False)
    assert client.get(f"/receipts/{b['receipt_id']}").status_code == 200


def test_playground_fails_closed_when_detection_fails(tmp_path):
    def down(req):
        raise httpx.ConnectError("no")

    env = {"LLM_BASE_URL": "https://a.test/v1", "LLM_API_KEY": "k", "CLAUSUS_RUNS_DIR": str(tmp_path)}
    client = TestClient(create_app(settings=Settings.from_env(env),
                                   http_client=httpx.Client(transport=httpx.MockTransport(down))))
    r = client.post("/playground/run", json={"text": TEXT})
    assert r.status_code == 503
    assert "Brunner" not in r.text


def test_split_original_aligns_tokens_to_what_was_said():
    segs = split_original("Anna und Ben in Bern.", "[PERSON_1] und [PERSON_2] in [PLACE_1].")
    assert [(s["text"], s["token"]) for s in segs] == [
        ("Anna", "[PERSON_1]"), (" und ", None), ("Ben", "[PERSON_2]"), (" in ", None),
        ("Bern", "[PLACE_1]"), (".", None)]


def test_playground_serves_the_page(tmp_path):
    client, _ = make(tmp_path, [])
    r = client.get("/")
    assert r.status_code == 200 and "Clausus" in r.text and "text/html" in r.headers["content-type"]
