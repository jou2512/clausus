import base64
import json

import httpx
import pytest

from clausus.errors import ClaususError
from clausus.hear import HearClient
from clausus.llm import Recorded
from clausus.see import read_image


def hear(handler) -> HearClient:
    return HearClient("http://hear.test:8000", httpx.Client(transport=httpx.MockTransport(handler)))


def test_hear_client_posts_file_and_returns_text():
    seen = {}

    def handler(req):
        seen["path"] = req.url.path
        seen["body"] = req.content
        return httpx.Response(200, json={"text": "Hallo zusammen.", "segments": [], "duration_s": 2.0,
                                         "device": "cpu"})

    result = hear(handler).transcribe(b"AUDIO-BYTES", "meeting.m4a")
    assert result.text == "Hallo zusammen."
    assert result.duration_s == 2.0
    assert seen["path"] == "/transcribe"
    assert b"AUDIO-BYTES" in seen["body"] and b"meeting.m4a" in seen["body"]


@pytest.mark.parametrize("status,code", [(503, "hear_loading"), (422, "hear_bad_audio"), (500, "hear_failed")])
def test_hear_client_maps_errors(status, code):
    with pytest.raises(ClaususError) as e:
        hear(lambda req: httpx.Response(status, json={"detail": "x"})).transcribe(b"a", "a.wav")
    assert e.value.code == code


def test_hear_client_unreachable():
    def down(req):
        raise httpx.ConnectError("no")

    with pytest.raises(ClaususError) as e:
        hear(down).transcribe(b"a", "a.wav")
    assert e.value.code == "hear_unreachable"


def test_hear_health_states():
    assert hear(lambda r: httpx.Response(200, json={"status": "ok"})).health() == "ok"
    assert hear(lambda r: httpx.Response(200, json={"status": "loading"})).health() == "loading"

    def down(req):
        raise httpx.ConnectError("no")

    assert hear(down).health() == "unreachable"


class Echo:
    destination = "apertus @ a.test"
    model = "m"

    def __init__(self):
        self.messages = None

    def chat(self, messages, model=None, step=None):
        self.messages = messages
        return "Projekt Nordlicht – Steering Committee"


def test_read_image_sends_data_url_and_instruction_and_records_as_original():
    inner, outgoing = Echo(), []
    text = read_image(b"\x89PNG-bytes", "image/png", Recorded(inner, outgoing, "original"))
    assert text == "Projekt Nordlicht – Steering Committee"
    content = inner.messages[0]["content"]
    kinds = [c["type"] for c in content]
    assert kinds == ["text", "image_url"]
    assert content[1]["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(b"\x89PNG-bytes").decode()
    assert [(o.step, o.kind, o.text) for o in outgoing] == [("see", "original", None)]
    assert "PNG-bytes" not in json.dumps([o.model_dump() for o in outgoing])


def test_read_image_rejects_non_image_mime():
    with pytest.raises(ClaususError) as e:
        read_image(b"x", "application/pdf", Echo())
    assert e.value.code == "unsupported_image"


# --- transcription through any OpenAI-compatible endpoint (Vasco, 05.10.: "give them the option") ---

from clausus.hear import EndpointHearClient


def endpoint(handler, key="sk-test") -> EndpointHearClient:
    return EndpointHearClient("https://stt.test/v1", key, "whisper-large-v3",
                              httpx.Client(transport=httpx.MockTransport(handler)))


def test_endpoint_client_speaks_the_openai_transcription_api():
    seen = {}

    def handler(req):
        seen["path"] = req.url.path
        seen["auth"] = req.headers.get("authorization")
        seen["body"] = req.content
        return httpx.Response(200, json={"text": "Grüezi mitenand."})

    result = endpoint(handler).transcribe(b"AUDIO-BYTES", "meeting.m4a")
    assert result.text == "Grüezi mitenand."
    assert seen["path"] == "/v1/audio/transcriptions"
    assert seen["auth"] == "Bearer sk-test"
    assert b"AUDIO-BYTES" in seen["body"] and b"whisper-large-v3" in seen["body"]
    assert result.destination == "stt.test" and not result.local


def test_endpoint_client_without_key_sends_no_auth_header():
    seen = {}

    def handler(req):
        seen["auth"] = req.headers.get("authorization")
        return httpx.Response(200, json={"text": "x"})

    endpoint(handler, key="").transcribe(b"a", "a.wav")
    assert seen["auth"] is None


@pytest.mark.parametrize("status,code", [(400, "hear_bad_audio"), (429, "rate_limited"), (500, "hear_failed")])
def test_endpoint_client_maps_errors(status, code):
    with pytest.raises(ClaususError) as e:
        endpoint(lambda req: httpx.Response(status, json={"error": "x"})).transcribe(b"a", "a.wav")
    assert e.value.code == code


def test_endpoint_client_unreachable_and_health():
    def down(req):
        raise httpx.ConnectError("no")

    with pytest.raises(ClaususError) as e:
        endpoint(down).transcribe(b"a", "a.wav")
    assert e.value.code == "hear_unreachable"
    assert endpoint(down).health() == "unreachable"
    assert endpoint(lambda r: httpx.Response(200, json={"data": []})).health() == "ok"


def test_local_hear_client_reports_local():
    t = hear(lambda req: httpx.Response(200, json={"text": "x", "duration_s": 1, "device": "cpu"})).transcribe(b"a", "a.wav")
    assert t.local
