"""Shared receipt identity for a continuity payload; never contains policy."""
from __future__ import annotations

import hashlib
import json


def continuity_digest(messages: list[dict]) -> str:
    """Bind a bridge receipt to exactly the app-selected IDs and rendered text."""
    payload = [(m.get("source_id"), m.get("role"), m.get("content"), m.get("timestamp"))
               for m in messages]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
