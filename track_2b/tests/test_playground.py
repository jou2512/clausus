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
    assert b["missing"] == []
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


def test_playground_reports_placeholders_the_model_dropped(tmp_path):
    client, _ = make(tmp_path, [DETECT, '{"personen":[]}', '{"namen":[]}', "Ruf [PERSON_1] an."])
    b = client.post("/playground/run", json={"text": TEXT}).json()
    assert b["missing"] == ["[ORG_1]", "[AMOUNT_1]"]


# --- audio, slides and the answer model (05.10.) ---

def make_full(tmp_path, replies, hear_text="Anna Keller ruft morgen an."):
    seen = []

    def apertus(req):
        body = json.loads(req.content)
        seen.append(body)
        return httpx.Response(200, json={"choices": [{"message": {"content": replies.pop(0)}}]})

    def hear(req):
        return httpx.Response(200, json={"text": hear_text, "duration_s": 3.0, "device": "cuda"})

    env = {"LLM_BASE_URL": "https://a.test/v1", "LLM_API_KEY": "k", "CLAUSUS_RUNS_DIR": str(tmp_path)}
    app = create_app(settings=Settings.from_env(env),
                     http_client=httpx.Client(transport=httpx.MockTransport(apertus)),
                     hear_http=httpx.Client(transport=httpx.MockTransport(hear)))
    return TestClient(app), seen


PERSON_ONLY = '{"stellen":[{"text":"Anna Keller","typ":"PERSON"}]}'


def test_upload_with_audio_and_slide_joins_everything_before_masking(tmp_path):
    client, seen = make_full(tmp_path, ["Projekt Nordlicht – Kunde: Meier Bau AG", PERSON_ONLY,
                                        '{"personen":[]}', '{"namen":[]}', "Ruf [PERSON_1] an."])
    r = client.post("/playground/upload", data={"text": "Bitte zusammenfassen."},
                    files=[("audio", ("rec.webm", b"AUDIO", "audio/webm")),
                           ("images", ("s.png", b"\x89PNG", "image/png"))])
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["transcript"] == "Anna Keller ruft morgen an."
    assert b["slides"] == ["Projekt Nordlicht – Kunde: Meier Bau AG"]
    assert "[PERSON_1]" in b["masked"] and "Ralph" not in b["masked"]
    assert "Ralph" not in json.dumps(seen[-1]["messages"])  # answer call: placeholders only
    assert b["answer"] == "Ruf Anna Keller an."
    assert [o["step"] for o in b["outgoing"]][:2] == ["see", "detect"]


def test_upload_needs_something(tmp_path):
    client, _ = make_full(tmp_path, [])
    assert client.post("/playground/upload", data={"text": " "}).status_code == 400


def test_unsupported_image_is_rejected_before_anything_is_sent(tmp_path):
    client, seen = make_full(tmp_path, [])
    r = client.post("/playground/upload", data={"text": "x"},
                    files=[("images", ("a.pdf", b"%PDF", "application/pdf"))])
    assert r.status_code == 415 and seen == []


def test_answer_model_is_used_for_the_answer_only(tmp_path):
    client, seen = make_full(tmp_path, [PERSON_ONLY, '{"personen":[]}', '{"namen":[]}', "ok [PERSON_1]"])
    r = client.post("/playground/run", json={"text": "Anna Keller ruft an.", "answer_model": "apertus-v1.5-8b"})
    assert r.status_code == 200, r.text
    assert seen[-1]["model"] == "apertus-v1.5-8b"
    assert seen[0]["model"] == "apertus-v1.5-70b"  # detection stays as measured


def test_unknown_answer_model_is_refused(tmp_path):
    client, seen = make_full(tmp_path, [])
    r = client.post("/playground/run", json={"text": "x", "answer_model": "gpt-4o"})
    assert r.status_code == 400 and seen == []


def test_playground_config_lists_answer_models_and_fixed_detection(tmp_path):
    client, _ = make_full(tmp_path, [])
    c = client.get("/playground/config").json()
    assert c["answer_models"] == ["apertus-v1.5-70b", "apertus-v1.5-8b"]
    assert c["detection"] == ["apertus-v1.5-70b", "apertus-v1.5-8b"]
    assert client.get("/playground/sample-slide.png").headers["content-type"] == "image/png"


def test_invented_placeholders_in_any_case_are_reported_but_not_checkboxes(tmp_path):
    answer = "Gruss, [Dein Name]. - [x] erledigt, siehe [1]. Ruf [PERSON_1] an."
    client, _ = make(tmp_path, [DETECT, '{"personen":[]}', '{"namen":[]}', answer])
    b = client.post("/playground/run", json={"text": TEXT}).json()
    assert b["invented"] == ["[Dein Name]"]


def test_every_call_and_the_whole_run_are_timed(tmp_path):
    client, _ = make_full(tmp_path, ["Folie", PERSON_ONLY, '{"personen":[]}', '{"namen":[]}', "Ruf [PERSON_1] an."])
    r = client.post("/playground/upload", data={"text": "x"},
                    files=[("audio", ("rec.webm", b"AUDIO", "audio/webm")),
                           ("images", ("s.png", b"\x89PNG", "image/png"))])
    b = r.json()
    assert all(isinstance(o["ms"], int) and o["ms"] >= 0 for o in b["outgoing"])
    t = b["timings"]
    assert set(t) >= {"hear_ms", "total_ms"} and t["total_ms"] >= t["hear_ms"] >= 0
    rec = json.loads((tmp_path / f"{b['receipt_id']}.json").read_text(encoding="utf-8"))
    assert all("ms" in o for o in rec["outgoing"])


# --- answer again with another model, without detecting again (05.10.) ---

def test_reanswer_uses_the_stored_masked_text_and_skips_detection(tmp_path):
    client, seen = make_full(tmp_path, [PERSON_ONLY, '{"personen":[]}', '{"namen":[]}', "A: [PERSON_1]",
                                        "B: [PERSON_1] und [ORG_9]"])
    first = client.post("/playground/run", json={"text": "Anna Keller ruft an."}).json()
    calls_before = len(seen)
    r = client.post("/playground/answer", json={"run_id": first["receipt_id"], "answer_model": "apertus-v1.5-8b"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert len(seen) == calls_before + 1  # one answer call, no detection
    assert seen[-1]["model"] == "apertus-v1.5-8b"
    assert "Ralph" not in json.dumps(seen[-1]["messages"]) and "[PERSON_1]" in json.dumps(seen[-1]["messages"])
    assert b["answer"] == "B: Anna Keller und [ORG_9]"
    assert b["invented"] == ["[ORG_9]"]
    assert b["answer_model"] == "apertus-v1.5-8b"
    assert b["receipt_id"] != first["receipt_id"]
    rec = json.loads((tmp_path / f"{b['receipt_id']}.json").read_text(encoding="utf-8"))
    assert [o["step"] for o in rec["outgoing"]] == ["answer"]
    assert "Ralph" not in json.dumps(rec, ensure_ascii=False)


def test_reanswer_unknown_or_expired_run_is_404(tmp_path):
    client, seen = make_full(tmp_path, [])
    r = client.post("/playground/answer", json={"run_id": "redact-000000000000"})
    assert r.status_code == 404 and seen == []


def test_reanswer_refuses_unknown_model(tmp_path):
    client, seen = make_full(tmp_path, [PERSON_ONLY, '{"personen":[]}', '{"namen":[]}', "A"])
    first = client.post("/playground/run", json={"text": "Anna Keller ruft an."}).json()
    n = len(seen)
    assert client.post("/playground/answer", json={"run_id": first["receipt_id"], "answer_model": "gpt-4o"}).status_code == 400
    assert len(seen) == n


# --- known names, given upfront (07.10.) ---

NOTHING = '{"stellen":[]}'


def test_known_names_are_masked_even_when_detection_misses_them(tmp_path):
    client, seen = make_full(tmp_path, [NOTHING, '{"personen":[]}', '{"namen":[]}', "Ruf [PERSON_1] bei [ORG_1] an."])
    r = client.post("/playground/run", json={
        "text": "Dr Zenhäusern vo de Kellerei Ried-Brig het aagrüeft.",
        "known_entities": [{"text": "Silvan Zenhäusern", "type": "PERSON"}, {"text": "Kellerei Ried-Brig", "type": "ORG"}]})
    assert r.status_code == 200, r.text
    b = r.json()
    assert "Zenhäusern" not in b["masked"] and "Kellerei" not in b["masked"]
    assert "Zenhäusern" not in json.dumps(seen[-1]["messages"], ensure_ascii=False)
    assert b["answer"] == "Ruf Silvan Zenhäusern bei Kellerei Ried-Brig an."
    assert {v["original"]: v["sources"] for v in b["vault"]}["Kellerei Ried-Brig"] == ["known"]


def test_known_names_work_with_upload_too(tmp_path):
    client, _ = make_full(tmp_path, [NOTHING, '{"personen":[]}', '{"namen":[]}', "ok"], hear_text="Signer chunnt.")
    r = client.post("/playground/upload", data={"text": "", "known_entities": '[{"text": "Hans Signer", "type": "PERSON"}]'},
                    files=[("audio", ("rec.webm", b"AUDIO", "audio/webm"))])
    assert r.status_code == 200, r.text
    assert "Signer" not in r.json()["masked"]


def test_bad_known_entities_are_a_client_error(tmp_path):
    client, seen = make_full(tmp_path, [])
    r = client.post("/playground/upload", data={"text": "x", "known_entities": "not json"})
    assert r.status_code == 400 and seen == []


def test_page_has_unique_ids_and_the_known_names_field(tmp_path):
    import re as _re
    client, _ = make(tmp_path, [])
    html = client.get("/").text
    ids = _re.findall(r'\bid="([^"]+)"', html)
    assert len(ids) == len(set(ids)), sorted({i for i in ids if ids.count(i) > 1})
    for needed in ("input", "known", "model", "b-run", "b-again", "progress", "atts", "map", "calls"):
        assert f'id="{needed}"' in html, needed


# --- receipts list and endpoints view (07.10.) ---

def test_receipts_list_newest_first_without_originals(tmp_path):
    client, _ = make_full(tmp_path, [PERSON_ONLY, '{"personen":[]}', '{"namen":[]}', "A [PERSON_1]",
                                     PERSON_ONLY, '{"personen":[]}', '{"namen":[]}', "B [PERSON_1]"])
    a = client.post("/playground/run", json={"text": "Anna Keller ruft an."}).json()["receipt_id"]
    b = client.post("/playground/run", json={"text": "Anna Keller ruft an."}).json()["receipt_id"]
    r = client.get("/receipts")
    assert r.status_code == 200
    items = r.json()["receipts"]
    assert [i["run_id"] for i in items][:2] == [b, a]
    first = items[0]
    assert first["placeholders"] == 1 and first["calls"] == 4
    assert first["original_calls"] == 3 and first["masked_calls"] == 1
    assert "Ralph" not in r.text


def test_config_lists_endpoints_but_never_keys(tmp_path):
    env = {"LLM_BASE_URL": "https://a.test/v1", "LLM_API_KEY": "sekret-llm", "CLAUSUS_RUNS_DIR": str(tmp_path),
           "STT_BASE_URL": "https://stt.example.ch/v1", "STT_API_KEY": "sekret-stt",
           "CLAUSUS_GUARDRAIL_KEY": "sekret-guard"}
    client = TestClient(create_app(settings=Settings.from_env(env)))
    c = client.get("/playground/config")
    body = c.json()
    assert "sekret" not in c.text
    e = body["endpoints"]
    assert e["detection"]["host"] == "a.test" and e["detection"]["key_set"] is True
    assert e["hear"] == {"mode": "endpoint", "host": "stt.example.ch", "model": "whisper-1", "key_set": True}
    assert e["guardrail"]["key_set"] is True and e["external"]["configured"] is False
