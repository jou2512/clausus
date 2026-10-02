"""See: slides and screenshots to text, read by Apertus (image input).

An image of a slide is confidential content, so the call is recorded as
"original" in the receipt (the bytes themselves are never stored).
"""

from __future__ import annotations

import base64

from clausus.detect.llm import ChatModel
from clausus.errors import ClaususError

PROMPT = ("Lies diesen Bildschirminhalt bzw. diese Folie vollständig und wörtlich ab. "
          "Gib nur den Text wieder, Zeile für Zeile, ohne Kommentar und ohne Zusammenfassung.")

IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}


def read_image(data: bytes, mime: str, llm: ChatModel) -> str:
    if mime not in IMAGE_TYPES:
        raise ClaususError("unsupported_image", f"Unsupported image type: {mime}.")
    url = f"data:{mime};base64,{base64.b64encode(data).decode()}"
    content = [{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": url}}]
    return llm.chat([{"role": "user", "content": content}], step="see").strip()
