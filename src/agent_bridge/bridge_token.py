"""X-Bridge-Token sender side for the app's bridge gates (TASK-1477).

The app derives each gate's token from its one root key and exports it to a
shared file (``llm_bawt.service.bridge_gate``). A bridge sends the file's
token when the file is readable and non-empty, else the legacy env value,
else no header at all. With no file mounted this is exactly the pre-1477
behaviour, so the bridge release can land before any compose change.

The file is read on every call: the app rewrites it when the root rotates,
and a cached copy would be rejected as ``bad``. Values are never logged.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Mapping

logger = logging.getLogger(__name__)

HEADER = "X-Bridge-Token"


@dataclass(frozen=True)
class Sender:
    file_env: str
    legacy_env: str


BROKER = Sender("BRIDGE_BROKER_TOKEN_FILE", "BRIDGE_CLAUDE_TOKEN_SECRET")
GIT_CREDENTIAL = Sender("BRIDGE_GIT_CREDENTIAL_TOKEN_FILE", "BRIDGE_GIT_CREDENTIAL_TOKEN")

_warned: set[str] = set()


def _read_file(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip() or None
    except OSError as exc:
        if path not in _warned:
            _warned.add(path)
            logger.warning("bridge token file %s unreadable (%s); using fallback",
                           path, type(exc).__name__)
        return None


def token(sender: Sender = BROKER, environ: Mapping[str, str] | None = None) -> str | None:
    env = os.environ if environ is None else environ
    path = (env.get(sender.file_env) or "").strip()
    if path:
        value = _read_file(path)
        if value:
            return value
    return (env.get(sender.legacy_env) or "").strip() or None


def headers(sender: Sender = BROKER, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    value = token(sender, environ)
    return {HEADER: value} if value else {}
