import os

import pytest
from fastapi.testclient import TestClient

from clausus.app import create_app

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("LLM_API_KEY"), reason="needs LLM_API_KEY"),
]


def test_live_health_reaches_apertus_and_finds_configured_model():
    r = TestClient(create_app()).get("/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["apertus"] == "ok"
    assert body["model_available"] is True
