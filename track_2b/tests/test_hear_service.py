import io
import math
import shutil
import struct
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hear_service.app import create_app
from hear_service.audio import SAMPLE_RATE, decode, ffmpeg_command
from hear_service.engine import FlixTranscriber, Segment


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
    assert c.get("/health").json() == {"status": "ok", "device": "cpu", "model": "fake-model", "loaded": True}


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


class CountingFlix:
    """FlixTranscriber with the model replaced by a counter: tests load, unload and reload."""

    @staticmethod
    def make(idle_unload_s: float = 600):
        from hear_service.engine import FlixTranscriber

        class T(FlixTranscriber):
            builds = releases = 0

            def _build_pipe(self):
                T.builds += 1
                self.device = "cpu"
                return lambda inp, return_timestamps: {"text": "grüezi", "chunks": []}

            def _release(self):
                T.releases += 1

        t = T(idle_unload_s=idle_unload_s)
        t._load()
        return t, T


def test_idle_model_is_unloaded_only_after_the_idle_time():
    t, T = CountingFlix.make(idle_unload_s=600)
    assert t.ready and t.loaded and T.builds == 1
    assert t.unload_if_idle(now=t._last_used + 599) is False
    assert t.loaded and T.releases == 0
    assert t.unload_if_idle(now=t._last_used + 601) is True
    assert not t.loaded and T.releases == 1
    assert t.ready  # still serves requests: it reloads on demand


def test_request_after_unload_reloads_the_model():
    t, T = CountingFlix.make()
    t.unload_if_idle(now=t._last_used + 10_000)
    segments = t.transcribe(np.zeros(16000, dtype=np.float32))
    assert segments[0].text == "grüezi"
    assert t.loaded and T.builds == 2


def test_idle_unload_zero_keeps_the_model():
    t, T = CountingFlix.make(idle_unload_s=0)
    assert t.unload_if_idle(now=t._last_used + 10**9) is False
    assert t.loaded and T.releases == 0


def test_no_unload_while_a_request_holds_the_model():
    t, _ = CountingFlix.make()
    with t._lock:  # a running transcription
        assert t.unload_if_idle(now=t._last_used + 10_000) is False
    assert t.loaded


def test_failed_reload_answers_503_not_500():
    t, _ = CountingFlix.make()
    t.unload_if_idle(now=t._last_used + 10_000)

    def boom():
        raise MemoryError("CUDA out of memory")

    t._build_pipe = boom
    c = TestClient(create_app(transcriber=t))
    import hear_service.app as app_mod
    orig = app_mod.decode
    app_mod.decode = lambda data: np.zeros(16000, dtype=np.float32)
    try:
        r = c.post("/transcribe", files={"file": ("a.wav", b"x", "audio/wav")})
    finally:
        app_mod.decode = orig
    assert r.status_code == 503
    assert "reloaded" in r.json()["detail"]


def test_health_stays_ok_while_unloaded_and_says_so():
    t, _ = CountingFlix.make()
    c = TestClient(create_app(transcriber=t))
    assert c.get("/health").json()["loaded"] is True
    t.unload_if_idle(now=t._last_used + 10_000)
    body = c.get("/health").json()
    assert body["status"] == "ok" and body["loaded"] is False


def test_the_pipeline_uses_whisper_long_form_not_chunked_batches(monkeypatch):
    """07.10.2026, RTX 3070: chunk_length_s=30 with batch_size=2 hung on a 50 s recording in 4 of 4
    runs (also on the 05.10 image, so not a library update); Whisper's own long-form decoding
    finished in about 14 s in 3 of 4 runs."""
    import sys
    import types

    seen = {}
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False),
                                       float16="fp16", float32="fp32")
    fake_tf = types.SimpleNamespace(pipeline=lambda task, **kw: seen.update(kw, task=task) or "pipe")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    assert FlixTranscriber(model="m")._build_pipe() == "pipe"
    assert seen["task"] == "automatic-speech-recognition"
    assert "chunk_length_s" not in seen and "batch_size" not in seen
