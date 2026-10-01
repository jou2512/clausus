from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator

SpanType = Literal[
    "PERSON", "ORG", "PROJECT", "PLACE", "AMOUNT", "DATE", "PHONE", "EMAIL", "IBAN", "AHV", "OTHER"
]
SPAN_TYPES: tuple[str, ...] = SpanType.__args__  # type: ignore[attr-defined]


class Span(BaseModel):
    """A confidential passage: exact text, its type, and where it sits in the original."""

    text: str
    type: SpanType
    start: int
    end: int
    source: Literal["rule", "model", "known"]

    @model_validator(mode="after")
    def _check_bounds(self) -> Span:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("span must have 0 <= start < end")
        return self

    @property
    def length(self) -> int:
        return self.end - self.start
