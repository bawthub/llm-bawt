"""Client side of the per-caller API key (TASK-2380).

Callers point ``LLM_BAWT_API_KEY_FILE`` at their exported key file. The file
is read on each call (cheap, and a replaced key needs no restart). With no
file configured or readable, no header is added: the server decides, which in
``log`` mode admits and records the call.
"""
from __future__ import annotations

import os
from collections.abc import MutableMapping

from .keys import HEADER

FILE_ENV = "LLM_BAWT_API_KEY_FILE"
HEADER_NAME = "X-LLM-Bawt-Key"
assert HEADER_NAME.lower() == HEADER


def header_value() -> str | None:
    path = (os.environ.get(FILE_ENV) or "").strip()
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            value = handle.read().strip()
    except OSError:
        return None
    return value or None


def auth_headers() -> dict[str, str]:
    """``{"X-LLM-Bawt-Key": ...}`` when a key is available, else ``{}``."""
    value = header_value()
    return {HEADER_NAME: value} if value else {}


def add_auth_header(headers: MutableMapping[str, str]) -> MutableMapping[str, str]:
    """Add the key in place unless the caller already set one; returns ``headers``."""
    if not any(k.lower() == HEADER for k in headers):
        headers.update(auth_headers())
    return headers
