"""Align a masked text back to its original: which stretch of what was really said became which
placeholder. Used by the playground display and by the benchmark (where a secret leaked)."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"\[[A-Z]+_\d+\]")


def _literal_end(original: str, literal: str, pos: int, known: set[str]) -> int:
    """Where the original of a placeholder ends: the first place the following literal text
    starts. A short literal (one space) can also sit inside a multi-word name ("Kessler Partner
    AG"), so a cut that leaves a spelling the vault knows wins over the first match."""
    end = original.find(literal, pos + 1)
    candidate = end
    while known and candidate != -1:
        if original[pos:candidate].casefold() in known:
            return candidate
        candidate = original.find(literal, candidate + 1)
    return end


def split_original(original: str, masked: str,
                   variants: dict[str, list[str]] | None = None) -> list[dict[str, str | None]]:
    """Align the masked text back to what was really said, so the playground can show
    which words became which placeholder ("Brunner" stays "Brunner", not the canonical name).
    The literal text between placeholders is identical in both; the gaps are the originals.
    `variants` (token -> spellings found, from the vault) settles cuts a short literal leaves open."""
    variants = variants or {}
    parts = _TOKEN.split(masked)
    tokens = _TOKEN.findall(masked)
    segs: list[dict[str, str | None]] = []
    pos = 0
    if not original.startswith(parts[0]):
        return [{"text": original, "token": None}]
    if parts[0]:
        segs.append({"text": parts[0], "token": None})
    pos = len(parts[0])
    for i, token in enumerate(tokens):
        literal = parts[i + 1]
        if i == len(tokens) - 1 and not literal:
            end = len(original)
        elif not literal:
            end = pos  # adjacent placeholders: the next one takes the shared text
        else:
            known = {v.casefold() for v in variants.get(token, [])}
            end = _literal_end(original, literal, pos, known) if i < len(tokens) - 1 else original.rfind(literal)
            if end < pos:
                return [{"text": original, "token": None}]
        if end > pos:
            segs.append({"text": original[pos:end], "token": token})
        if literal:
            segs.append({"text": literal, "token": None})
        pos = end + len(literal)
    if "".join(str(s["text"]) for s in segs) != original:
        return [{"text": original, "token": None}]
    return segs
