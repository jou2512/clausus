import os
from pathlib import Path

import pytest

from clausus.config import Settings
from clausus.llm import ApertusClient
from clausus.see import read_image

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("LLM_API_KEY"), reason="needs LLM_API_KEY"),
]

SLIDE = Path(__file__).parent.parent / "fixtures" / "slide-invented.png"  # invented content


def test_live_apertus_reads_slide_verbatim():
    text = read_image(SLIDE.read_bytes(), "image/png", ApertusClient(Settings.from_env()))
    for expected in ["Projekt Nordlicht", "Meier & Söhne Bau AG", "lea.zimmermann@meier-bau.ch",
                     "128'500", "Marco Rossi", "Bern-Bümpliz"]:
        assert expected in text, f"missing {expected!r} in:\n{text}"
