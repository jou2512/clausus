import httpx
from fastapi.testclient import TestClient

from clausus.app import create_app
from clausus.config import Settings

SETTINGS = Settings.from_env(
    {"LLM_BASE_URL": "https://apertus.test/v1", "LLM_API_KEY": "sk-secret-123",
     "LLM_NAME": "apertus-v1.5-70b"}
)


def client_with(handler, hear_status="ok") -> TestClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    if hear_status == "unreachable":
        def hear_handler(req):
            raise httpx.ConnectError("no")
    else:
        def hear_handler(req):
            return httpx.Response(200, json={"status": hear_status})
    hear = httpx.Client(transport=httpx.MockTransport(hear_handler))
    return TestClient(create_app(settings=SETTINGS, http_client=http, hear_http=hear))


def test_health_is_degraded_while_hear_is_loading_or_unreachable():
    ok = lambda req: httpx.Response(200, json={"data": [{"id": "apertus-v1.5-70b"}]})
    for state in ("loading", "unreachable"):
        body = client_with(ok, hear_status=state).get("/health").json()
        assert body["status"] == "degraded"
        assert body["hear"] == state


def test_health_ok_lists_models_and_confirms_configured_model():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"data": [{"id": "apertus-v1.5-70b"}, {"id": "apertus-v1.5-8b"}]})

    r = client_with(handler).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "status": "ok",
        "apertus": "ok",
        "model": "apertus-v1.5-70b",
        "model_available": True,
        "models": ["apertus-v1.5-70b", "apertus-v1.5-8b"],
        "hear": "ok",
    }
    assert seen["url"] == "https://apertus.test/v1/models"
    assert seen["auth"] == "Bearer sk-secret-123"


def test_health_flags_configured_model_missing_from_endpoint():
    r = client_with(lambda req: httpx.Response(200, json={"data": [{"id": "other"}]})).get("/health")
    assert r.status_code == 200
    assert r.json()["model_available"] is False
    assert r.json()["status"] == "degraded"


def test_health_reports_auth_failure_without_leaking_key():
    r = client_with(lambda req: httpx.Response(401, json={"error": "bad key"})).get("/health")
    assert r.status_code == 503
    assert r.json()["apertus"] == "auth_failed"
    assert "sk-secret-123" not in r.text


def test_health_reports_unreachable_endpoint():
    def handler(request):
        raise httpx.ConnectError("no route")

    r = client_with(handler).get("/health")
    assert r.status_code == 503
    assert r.json()["apertus"] == "unreachable"


def test_health_reports_unexpected_payload():
    r = client_with(lambda req: httpx.Response(200, text="not json")).get("/health")
    assert r.status_code == 503
    assert r.json()["apertus"] == "bad_response"
