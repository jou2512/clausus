import json
import os

import pytest
from fastapi.testclient import TestClient

from clausus.app import create_app

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("LLM_API_KEY"), reason="needs LLM_API_KEY"),
]

# Invented meeting transcript (self-written, no real people), in the High German form the
# Swiss German speech model produces.
TRANSCRIPT = (
    "Also, kurz zum Stand vom Projekt Nordlicht. Ich habe gestern mit Lea Zimmermann von der "
    "Meier und Söhne Bau AG telefoniert. Sie geben das Budget von 128'500 Franken frei, aber nur "
    "wenn die Offerte bis am 20. Oktober bei ihnen ist. Marco macht das Kick-off in Winterthur. "
    "Ihre Nummer ist 078 555 12 34. Entscheid: wir starten mit dem Pilot in Bern-Bümpliz. "
    "Action Item für Marco: Offerte bis Freitag an Zimmermann schicken."
)
SECRETS = ["Nordlicht", "Lea", "Zimmermann", "Meier", "128'500", "20. Oktober", "Marco",
           "Winterthur", "078 555 12 34", "Bümpliz"]


def test_live_meeting_protocol_never_shows_originals_to_the_protocol_model(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUSUS_RUNS_DIR", str(tmp_path))
    r = TestClient(create_app()).post("/meeting", json={"transcript": TRANSCRIPT})
    assert r.status_code == 200, r.text
    body = r.json()
    seen = body["seen_by_model"]
    leaked = [s for s in SECRETS if s.lower() in seen.lower()]
    assert leaked == [], f"leaked into protocol prompt: {leaked}\n{seen}"
    assert body["destination"] == "local"
    assert body["protocol"]["zusammenfassung"].strip()
    assert body["warnings"]["invented"] == []
    receipt = json.loads((tmp_path / f"{body['receipt_id']}.json").read_text(encoding="utf-8"))
    assert [o["step"] for o in receipt["outgoing"]] == ["detect", "detect_persons", "detect_names", "protocol"]
    assert not any(s in json.dumps(receipt, ensure_ascii=False) for s in SECRETS)
