import json

import httpx
import pytest
from fastapi.testclient import TestClient

from clausus.app import create_app
from clausus.config import Settings
from clausus.receipt import Receipt, write_receipt

SECRET_TEXT = "De Reto Ammann vo de Gisler Holzbau GmbH, Natel 079 412 33 87."
MODEL_REPLY = ('{"stellen":[{"text":"Reto Ammann","typ":"PERSON"},'
               '{"text":"Gisler Holzbau GmbH","typ":"FIRMA"}]}')


def test_receipt_is_written_once_and_never_overwritten(tmp_path):
    r = Receipt(run_id="abc", model="m", placeholders=[], outgoing=[])
    p = write_receipt(r, tmp_path)
    assert json.loads(p.read_text(encoding="utf-8"))["run_id"] == "abc"
    with pytest.raises(FileExistsError):
        write_receipt(r, tmp_path)


def make_client(tmp_path) -> TestClient:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path.endswith("/chat/completions")
        return httpx.Response(200, json={"choices": [{"message": {"content": MODEL_REPLY}}]})

    settings = Settings.from_env({"LLM_BASE_URL": "https://a.test/v1", "LLM_API_KEY": "k",
                                  "CLAUSUS_RUNS_DIR": str(tmp_path)})
    return TestClient(create_app(settings=settings,
                                 http_client=httpx.Client(transport=httpx.MockTransport(handler))))


def test_redact_masks_rules_and_model_spans_and_writes_receipt(tmp_path):
    r = make_client(tmp_path).post("/redact", json={"text": SECRET_TEXT})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["masked_text"] == "De [PERSON_1] vo de [ORG_1], Natel [PHONE_1]."
    assert sorted(p["token"] for p in body["placeholders"]) == ["[ORG_1]", "[PERSON_1]", "[PHONE_1]"]
    receipt = (tmp_path / f"{body['receipt_id']}.json").read_text(encoding="utf-8")
    for secret in ("Reto", "Ammann", "Gisler", "079 412"):
        assert secret not in receipt
        assert secret not in r.text


def test_redact_with_known_entities_masks_them(tmp_path):
    r = make_client(tmp_path).post(
        "/redact", json={"text": "Grüess a d'Nadine.", "known_entities": [{"text": "Nadine", "type": "PERSON"}]}
    )
    assert "Nadine" not in r.json()["masked_text"]


def test_redact_fails_closed_when_model_is_down(tmp_path):
    def handler(req):
        raise httpx.ConnectError("down")

    settings = Settings.from_env({"LLM_BASE_URL": "https://a.test/v1", "LLM_API_KEY": "k",
                                  "CLAUSUS_RUNS_DIR": str(tmp_path)})
    c = TestClient(create_app(settings=settings,
                              http_client=httpx.Client(transport=httpx.MockTransport(handler))))
    r = c.post("/redact", json={"text": SECRET_TEXT})
    assert r.status_code == 503
    assert "Reto" not in r.text
    assert list(tmp_path.iterdir()) == []
