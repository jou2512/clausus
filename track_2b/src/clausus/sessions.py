"""Chat sessions: one vault and the history per session, encrypted at rest.

Each session is one file, AES-256-GCM, a fresh 12-byte nonce per save and the session id as
associated data, so a file cannot be swapped under another id. The file holds the originals
(that is the point of a vault), so it is never written in the clear and never sent anywhere.

The key comes from CLAUSUS_VAULT_KEY (base64, 32 bytes). Without it, a key is created once next
to the sessions. That keeps a demo working, but whoever has the disk then has the key too:
for real use the key belongs in the environment, outside the data volume.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from clausus.errors import ClaususError
from clausus.vault import Vault

_ID = re.compile(r"s-[0-9a-f]{16}")
KEY_FILE = ".vault.key"


def load_or_create_key(env_value: str, directory: str | Path) -> bytes:
    if env_value.strip():
        try:
            key = base64.b64decode(env_value.strip(), validate=True)
        except (ValueError, binascii.Error) as e:
            raise ClaususError("bad_vault_key", "CLAUSUS_VAULT_KEY must be base64.") from e
        if len(key) != 32:
            raise ClaususError("bad_vault_key", "CLAUSUS_VAULT_KEY must decode to 32 bytes.")
        return key
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    path = d / KEY_FILE
    if path.is_file():
        key = path.read_bytes()
        if len(key) != 32:
            raise ClaususError("bad_vault_key", f"{path} is not a 32-byte key.")
        return key
    key = AESGCM.generate_key(bit_length=256)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(key)
    return key


@dataclass
class Session:
    id: str
    vault: Vault = field(default_factory=Vault)
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self, directory: str | Path, key: bytes) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._aead = AESGCM(key)

    def _path(self, session_id: str) -> Path:
        if not _ID.fullmatch(session_id):
            raise ClaususError("session_not_found", "No such session.")
        return self._dir / f"{session_id}.bin"

    def create(self) -> Session:
        return Session(id=f"s-{secrets.token_hex(8)}")

    def save(self, s: Session) -> None:
        s.updated_at = time.time()
        plain = json.dumps({"vault": s.vault.to_dict(), "messages": s.messages,
                            "created_at": s.created_at, "updated_at": s.updated_at},
                           ensure_ascii=False).encode("utf-8")
        nonce = os.urandom(12)
        blob = nonce + self._aead.encrypt(nonce, plain, s.id.encode())
        path = self._path(s.id)
        tmp = path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # owner only, like the key
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
        tmp.replace(path)  # atomic: a crash never leaves half a file

    def load(self, session_id: str) -> Session:
        path = self._path(session_id)
        if not path.is_file():
            raise ClaususError("session_not_found", "No such session.")
        blob = path.read_bytes()
        try:
            plain = self._aead.decrypt(blob[:12], blob[12:], session_id.encode())
        except InvalidTag as e:
            raise ClaususError("session_unreadable",
                               "This session cannot be decrypted (other key, or the file was changed).") from e
        data = json.loads(plain)
        return Session(id=session_id, vault=Vault.from_dict(data["vault"]), messages=data["messages"],
                       created_at=data["created_at"], updated_at=data["updated_at"])

    def list(self) -> list[dict[str, Any]]:
        out = []
        for path in self._dir.glob("s-*.bin"):
            try:
                s = self.load(path.stem)
            except ClaususError:
                out.append({"id": path.stem, "readable": False})
                continue
            first = next((m["original"] for m in s.messages if m.get("role") == "user"), "")
            out.append({"id": s.id, "readable": True, "messages": len(s.messages),
                        "placeholders": len(s.vault.tokens), "created_at": s.created_at,
                        "updated_at": s.updated_at, "title": first[:60]})
        return sorted(out, key=lambda x: x.get("updated_at", 0), reverse=True)

    def delete(self, session_id: str) -> None:
        path = self._path(session_id)
        if path.is_file():
            path.unlink()
