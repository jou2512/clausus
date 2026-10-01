import httpx
from fastapi.testclient import TestClient

from clausus.app import create_app
from clausus.config import Settings

SETTINGS = Settings.from_env(
    {"LLM_BASE_URL": "https://apertus.test/v1", "LLM_API_KEY": "sk-secret-123",
     "LLM_NAME": "apertus-v1.5-70b"}
)


def client_with(handler) -> TestClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return TestClient(create_app(settings=SETTINGS, http_client=http))


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
