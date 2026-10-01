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
        )
