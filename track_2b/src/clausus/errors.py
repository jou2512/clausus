from __future__ import annotations


class ClaususError(Exception):
    """Every failure carries a stable code. Callers treat any ClaususError as
    "nothing leaves the house" (SPEC section 4)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ConfigError(ClaususError):
    def __init__(self, message: str) -> None:
        super().__init__("config", message)
