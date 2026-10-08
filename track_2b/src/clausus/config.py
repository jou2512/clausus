from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from clausus.errors import ConfigError

DEFAULT_MODEL = "apertus-v1.5-70b"
DEFAULT_FAST_MODEL = "apertus-v1.5-8b"
REQUIRED = ("LLM_BASE_URL", "LLM_API_KEY")

__all__ = ["DEFAULT_MODEL", "ConfigError", "Settings"]


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Variable names follow the Hack Apertus project template."""

    llm_base_url: str
    llm_api_key: str = field(repr=False)
    llm_name: str = DEFAULT_MODEL
    runs_dir: str = "runs"
    fast_model: str = DEFAULT_FAST_MODEL
    routes_path: str = ""
    # Optional external target for masked text (product configuration, off by default).
    external_base_url: str = ""
    external_api_key: str = field(default="", repr=False)
    external_model: str = ""
    hear_url: str = "http://hear:8000"
    guardrail_key: str = field(default="", repr=False)
    # Optional: transcription through an OpenAI-compatible endpoint instead of the local hear container.
    stt_base_url: str = ""
    stt_api_key: str = field(default="", repr=False)
    stt_model: str = "whisper-1"
    # Chat sessions, encrypted at rest. Empty key: one is created next to the sessions (demo only).
    vault_dir: str = ""
    vault_key: str = field(default="", repr=False)
    # Apertus judges finds only one model pass made (detect/judge.py).
    judge: bool = False

    @property
    def external_configured(self) -> bool:
        return bool(self.external_base_url and self.external_api_key)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        missing = [k for k in REQUIRED if not env.get(k, "").strip()]
        if missing:
            raise ConfigError(
                f"Missing environment variable(s): {', '.join(missing)}. "
                "Copy .env.example to .env and fill them in."
            )
        url = env["LLM_BASE_URL"].strip().rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise ConfigError(f"LLM_BASE_URL must start with http:// or https://, got {url!r}.")
        return cls(
            llm_base_url=url,
            llm_api_key=env["LLM_API_KEY"].strip(),
            llm_name=env.get("LLM_NAME", "").strip() or DEFAULT_MODEL,
            runs_dir=env.get("CLAUSUS_RUNS_DIR", "").strip() or "runs",
            fast_model=env.get("CLAUSUS_FAST_MODEL", "").strip() or DEFAULT_FAST_MODEL,
            routes_path=env.get("CLAUSUS_ROUTES", "").strip(),
            external_base_url=env.get("EXTERNAL_BASE_URL", "").strip().rstrip("/"),
            external_api_key=env.get("EXTERNAL_API_KEY", "").strip(),
            external_model=env.get("EXTERNAL_MODEL", "").strip(),
            hear_url=env.get("CLAUSUS_HEAR_URL", "").strip() or "http://hear:8000",
            guardrail_key=env.get("CLAUSUS_GUARDRAIL_KEY", "").strip(),
            stt_base_url=env.get("STT_BASE_URL", "").strip().rstrip("/"),
            stt_api_key=env.get("STT_API_KEY", "").strip(),
            stt_model=env.get("STT_MODEL", "").strip() or "whisper-1",
            vault_dir=env.get("CLAUSUS_VAULT_DIR", "").strip(),
            vault_key=env.get("CLAUSUS_VAULT_KEY", "").strip(),
            judge=env.get("CLAUSUS_JUDGE", "").strip().lower() in ("1", "true", "yes", "on"),
        )
