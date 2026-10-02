import json

import httpx
from fastapi.testclient import TestClient

from clausus.app import create_app
from clausus.config import Settings

SPOKEN = "Thomas Brunner von der Huber Treuhand AG will das Budget bis Ende November."
SLIDE = "Projekt Nordlicht – Kunde: Meier Bau AG"
DETECT = ('{"stellen":[{"text":"Thomas Brunner","typ":"PERSON"},{"text":"Huber Treuhand AG","typ":"FIRMA"},'
          '{"text":"Projekt Nordlicht","typ":"PROJEKT"},{"text":"Meier Bau AG","typ":"FIRMA"}]}')
PERSONS = '{"personen":["Thomas Brunner"]}'
NAMES = '{"namen":[]}'
PROTOCOL = ('{"zusammenfassung":"[PERSON_1] von [ORG_1] will Budget für [PROJECT_1].",'
            '"entscheide":[],"pendenzen":[]}')


def make(tmp_path, apertus_replies, hear_handler=None):
    prompts = []

    def apertus(req):
        body = json.loads(req.content)
        prompts.append(body["messages"][0]["content"])
        return httpx.Response(200, json={"choices": [{"message": {"content": apertus_replies.pop(0)}}]})

    def default_hear(req):
        return httpx.Response(200, json={"text": SPOKEN, "segments": [], "duration_s": 4.2, "device": "cpu"})

    env = {"LLM_BASE_URL": "https://apertus.test/v1", "LLM_API_KEY": "k", "CLAUSUS_RUNS_DIR": str(tmp_path)}
    app = create_app(settings=Settings.from_env(env),
                     http_client=httpx.Client(transport=httpx.MockTransport(apertus)),
                     hear_http=httpx.Client(transport=httpx.MockTransport(hear_handler or default_hear)))
    return TestClient(app), prompts


def receipt(tmp_path, body):
    return json.loads((tmp_path / f"{body['receipt_id']}.json").read_text(encoding="utf-8"))


def test_audio_and_slide_flow_into_one_protocol(tmp_path):
    client, prompts = make(tmp_path, [SLIDE, DETECT, PERSONS, NAMES, PROTOCOL])
    r = client.post("/meeting/upload",
                    files=[("audio", ("m.m4a", b"AUDIO", "audio/mp4")),
                           ("images", ("s.png", b"\x89PNG", "image/png"))])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["transcript"] == SPOKEN
    assert body["slides"] == [SLIDE]
    assert body["protocol"]["zusammenfassung"] == ("Thomas Brunner von Huber Treuhand AG will Budget "
                                                   "für Projekt Nordlicht.")
    for secret in ("Brunner", "Huber", "Nordlicht", "Meier"):
        assert secret not in body["seen_by_model"]
        assert secret not in prompts[4]  # the protocol prompt (5th call)
    rec = receipt(tmp_path, body)
    assert [(o["step"], o["kind"]) for o in rec["outgoing"]] == [
        ("see", "original"), ("detect", "original"), ("detect_persons", "original"), ("detect_names", "original"), ("protocol", "masked")]
    assert any("audio transcribed locally" in n and "cpu" in n for n in rec["notes"])
    assert "Brunner" not in json.dumps(rec, ensure_ascii=False)


def test_audio_only(tmp_path):
    client, _ = make(tmp_path, [DETECT, PERSONS, NAMES, PROTOCOL])
    r = client.post("/meeting/upload", files=[("audio", ("m.wav", b"AUDIO", "audio/wav"))])
    assert r.status_code == 200, r.text
    assert r.json()["slides"] == []


def test_text_plus_known_entities_as_form_fields(tmp_path):
    client, _ = make(tmp_path, ['{"stellen":[]}', '{"personen":[]}', '{"namen":[]}', '{"zusammenfassung":"x","entscheide":[],"pendenzen":[]}'])
    r = client.post("/meeting/upload", data={
        "transcript": "D'Sandra macht s'Kickoff.",
        "known_entities": json.dumps([{"text": "Sandra", "type": "PERSON"}]),
    })
    assert r.status_code == 200, r.text
    assert "Sandra" not in r.json()["seen_by_model"]


def test_nothing_uploaded_is_a_client_error(tmp_path):
    client, prompts = make(tmp_path, [])
    r = client.post("/meeting/upload", data={})
    assert r.status_code == 400
    assert prompts == []


def test_hear_failure_sends_nothing_not_even_slides(tmp_path):
    def down(req):
        raise httpx.ConnectError("no")

    client, prompts = make(tmp_path, [SLIDE], hear_handler=down)
    r = client.post("/meeting/upload",
                    files=[("audio", ("m.m4a", b"A", "audio/mp4")), ("images", ("s.png", b"P", "image/png"))])
    assert r.status_code == 503
    assert r.json()["error"] == "hear_unreachable"
    assert prompts == []
    assert list(tmp_path.iterdir()) == []


def test_unsupported_image_is_415_and_nothing_is_sent(tmp_path):
    client, prompts = make(tmp_path, [])
    r = client.post("/meeting/upload", files=[("images", ("x.pdf", b"%PDF", "application/pdf"))])
    assert r.status_code == 415
    assert prompts == []


def test_unreadable_audio_is_422(tmp_path):
    client, _ = make(tmp_path, [], hear_handler=lambda req: httpx.Response(422, json={"detail": "x"}))
    r = client.post("/meeting/upload", files=[("audio", ("m.m4a", b"A", "audio/mp4"))])
    assert r.status_code == 422
    assert r.json()["error"] == "hear_bad_audio"
