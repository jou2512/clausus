"""The receipt: what was found and what left the house, per run.

Append-only: a receipt is written once and never overwritten. Every call to a
model endpoint is listed, including detection calls that carry original text
(by design in the Swiss-cloud architecture). Masked text is stored verbatim,
original text never.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PlaceholderInfo(BaseModel):
    token: str
    type: str
    sources: list[str]


class Outgoing(BaseModel):
    step: str
    destination: str
    kind: Literal["original", "masked"]
    model: str
    text: str | None = None  # masked text verbatim; None for original text

    @model_validator(mode="after")
    def _never_store_originals(self) -> Outgoing:
        if self.kind == "original" and self.text is not None:
            raise ValueError("original text must never be stored in a receipt")
        return self


class Receipt(BaseModel):
    run_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    model: str
    destination: str = "local"
    placeholders: list[PlaceholderInfo]
    outgoing: list[Outgoing]
    notes: list[str] = Field(default_factory=list)


def write_receipt(receipt: Receipt, directory: str | Path) -> Path:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{receipt.run_id}.json"
    with path.open("x", encoding="utf-8") as f:  # "x": refuse to overwrite
        f.write(receipt.model_dump_json(indent=2))
    return path
