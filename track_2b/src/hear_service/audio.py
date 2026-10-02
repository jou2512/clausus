"""Any audio container in, 16 kHz mono float32 out (what Whisper-type models expect)."""

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

SAMPLE_RATE = 16000


def ffmpeg_command(path: str) -> list[str]:
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "f32le", "pipe:1"]


def decode(data: bytes) -> np.ndarray:
    """Decode with ffmpeg. Input goes through a temporary file inside the container,
    deleted right after: measured 02.10.2026, a Windows-recorded .m4a decoded to nothing
    from stdin but fully from a file (container index at the end of the file).
    Raises ValueError if ffmpeg cannot read the bytes as audio."""
    fd, path = tempfile.mkstemp(suffix=".audio")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        proc = subprocess.run(ffmpeg_command(path), capture_output=True, check=False)
    finally:
        os.unlink(path)
    if proc.returncode != 0 or not proc.stdout:
        raise ValueError("could not decode audio")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()
