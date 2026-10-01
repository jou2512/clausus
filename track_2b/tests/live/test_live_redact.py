import os

import pytest
from fastapi.testclient import TestClient

from clausus.app import create_app

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("LLM_API_KEY"), reason="needs LLM_API_KEY"),
]

# Invented Swiss German meeting snippet (self-written, no real people).
TEXT = (
    "Also, churz zum Stand. De Reto Ammann vo de Gisler Holzbau GmbH het mer geschter aaglüüte. "
    "Si wänd d'Offerte für s'Projekt Seeblick bis am 12. Oktober, s'Budget isch zwüsche 80 und 95 "
    "tuusig Franke. Sini Handynummer isch 079 412 33 87. D'Nadine macht nächscht Wuche de Termin "
    "in Sursee, und d'Kundeliste vo de Raiffeisen Willisau dörf uf kei Fall usse."
)
SECRETS = ["Reto", "Ammann", "Gisler", "Seeblick", "12. Oktober", "079 412 33 87", "Nadine",
           "Sursee", "Raiffeisen", "Willisau"]


def test_live_redact_masks_every_known_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUSUS_RUNS_DIR", str(tmp_path))
    r = TestClient(create_app()).post("/redact", json={"text": TEXT})
    assert r.status_code == 200, r.text
    masked = r.json()["masked_text"]
    leaked = [s for s in SECRETS if s.lower() in masked.lower()]
    assert leaked == [], f"leaked: {leaked}\n{masked}"
