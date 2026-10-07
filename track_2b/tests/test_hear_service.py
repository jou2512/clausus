import io
import math
import shutil
import struct
import wave

import pytest
from fastapi.testclient import TestClient

from hear_service.app import create_app
from hear_service.audio import SAMPLE_RATE, decode, ffmpeg_command
from hear_service.engine import Segment


def wav_bytes(seconds: float = 1.0, rate: int = 44100) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(
            struct.pack("<hh", s, s)
            for s in (int(8000 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(int(rate * seconds)))
        )
        w.writeframes(frames)
    return buf.getvalue()


def test_ffmpeg_reads_from_a_file_not_a_pipe_and_writes_16k_mono_float():
    """Measured 02.10.2026 in the container (ffmpeg 7.1.5): a Windows-recorded .m4a decoded
    to 0 bytes from stdin ("partial file") but fully from a file. So input is a file path."""
    cmd = ffmpeg_command("/tmp/in.audio")
    assert cmd[0] == "ffmpeg"
    assert cmd[cmd.index("-i") + 1] == "/tmp/in.audio"
    assert "pipe:0" not in cmd
    assert cmd[cmd.index("-ar") + 1] == str(SAMPLE_RATE) == "16000"
    assert cmd[cmd.index("-ac") + 1] == "1"
    assert cmd[cmd.index("-f") + 1] == "f32le"
    assert cmd[-1] == "pipe:1"


def test_decode_removes_its_temporary_file(monkeypatch, tmp_path):
    import hear_service.audio as audio_mod

    monkeypatch.setattr(audio_mod.tempfile, "tempdir", str(tmp_path))
    if shutil.which("ffmpeg"):
        decode(wav_bytes(0.2))
    else:
        with pytest.raises((FileNotFoundError, ValueError)):
            decode(b"x")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_decode_turns_stereo_44k_into_16k_mono_of_same_duration():
    audio = decode(wav_bytes(1.0))
    assert abs(len(audio) - 16000) < 200
    assert audio.dtype.name == "float32"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_decode_reads_aac_m4a(tmp_path):
    """Compatibility check for .m4a input. Note: this ffmpeg-made file does NOT reproduce the
    02.10.2026 pipe failure (that needed a Windows-recorded file), so the regression guard is
    the file-input test above, not this one."""
    import subprocess
    src, m4a = tmp_path / "in.wav", tmp_path / "out.m4a"
    src.write_bytes(wav_bytes(1.0))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-c:a", "aac", str(m4a)],
                   check=True)
    data = m4a.read_bytes()
    assert data.find(b"moov") > data.find(b"mdat"), "fixture must have moov after mdat"
    audio = decode(data)
    assert abs(len(audio) - 16000) < 1200


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_decode_rejects_garbage():
    with pytest.raises(ValueError):
        decode(b"this is not audio at all")


class FakeTranscriber:
    device = "cpu"
    model = "fake-model"
    ready = True

    def __init__(self):
        self.got = None

    def transcribe(self, audio):
        self.got = audio
        return [Segment(start=0.0, end=1.0, text=" Hallo"), Segment(start=1.0, end=2.0, text=" zusammen.")]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_transcribe_endpoint_returns_joined_text_segments_and_duration():
    fake = FakeTranscriber()
    c = TestClient(create_app(transcriber=fake))
    r = c.post("/transcribe", files={"file": ("a.wav", wav_bytes(2.0), "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "Hallo zusammen."
    assert body["segments"][1] == {"start": 1.0, "end": 2.0, "text": "zusammen."}
    assert body["duration_s"] == pytest.approx(2.0, abs=0.05)
    assert body["device"] == "cpu"
    assert len(fake.got) > 30000


def test_transcribe_rejects_empty_upload():
    c = TestClient(create_app(transcriber=FakeTranscriber()))
    r = c.post("/transcribe", files={"file": ("a.wav", b"", "audio/wav")})
    assert r.status_code == 400


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_transcribe_rejects_undecodable_audio():
    c = TestClient(create_app(transcriber=FakeTranscriber()))
    r = c.post("/transcribe", files={"file": ("a.wav", b"garbage", "audio/wav")})
    assert r.status_code == 422


def test_health_reports_loading_until_model_is_ready():
    fake = FakeTranscriber()
    fake.ready = False
    c = TestClient(create_app(transcriber=fake))
    assert c.get("/health").json()["status"] == "loading"
    assert c.post("/transcribe", files={"file": ("a.wav", b"x", "audio/wav")}).status_code == 503
    fake.ready = True
    assert c.get("/health").json() == {"status": "ok", "device": "cpu", "model": "fake-model"}


# --- OpenAI-compatible face (07.10.): any tool that speaks /v1/audio/transcriptions can use it ---

def test_openai_transcriptions_returns_text():
    c = TestClient(create_app(transcriber=FakeTranscriber()))
    r = c.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav_bytes(2.0), "audio/wav")},
               data={"model": "anything"})
    assert r.status_code == 200, r.text
    assert r.json() == {"text": "Hallo zusammen."}


def test_openai_transcriptions_text_and_verbose_formats():
    c = TestClient(create_app(transcriber=FakeTranscriber()))
    files = {"file": ("a.wav", wav_bytes(2.0), "audio/wav")}
    t = c.post("/v1/audio/transcriptions", files=files, data={"model": "x", "response_format": "text"})
    assert t.headers["content-type"].startswith("text/plain") and t.text == "Hallo zusammen."
    v = c.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav_bytes(2.0), "audio/wav")},
               data={"model": "x", "response_format": "verbose_json"}).json()
    assert v["text"] == "Hallo zusammen." and v["duration"] == pytest.approx(2.0, abs=0.05) and len(v["segments"]) == 2


def test_openai_transcriptions_errors_and_models():
    c = TestClient(create_app(transcriber=FakeTranscriber()))
    assert c.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"", "audio/wav")}).status_code == 400
    assert c.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"garbage", "audio/wav")}).status_code == 422
    m = c.get("/v1/models").json()
    assert m["object"] == "list" and m["data"][0]["id"]
