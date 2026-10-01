"""The receipt: what was found and what left the house, per run.

Append-only: a receipt is written once and never overwritten. It holds
placeholders and the exact outgoing (already masked) text, never originals.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field


class PlaceholderInfo(BaseModel):
    token: str
    type: str
    sources: list[str]


class Outgoing(BaseModel):
    destination: str
    text: str


class Receipt(BaseModel):
    run_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    model: str
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
