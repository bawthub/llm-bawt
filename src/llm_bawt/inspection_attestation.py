"""Purpose-specific app proof for the fixed BawtHub Tasks observation route."""

from __future__ import annotations

import hmac
import json
import logging
import os
import secrets
import time
from pathlib import Path

from .bawthub_auth.token_export import publish
from .inspection_admission import inspection_is_fresh, inspection_origin
from .task_turn_context import TaskTurnContext, is_inspection_turn_id

INSPECTION_TASKS_PATH = "/internal/inspection/tasks"
INSPECTION_MAC_HEADER = "X-BawtHub-Inspection-Mac"
_PURPOSE = b"bawthub:inspection:tasks:v1\0"
_TTL_MS = 15_000
_MAX_BODY = 2048
_EXPORT_FILE = "BAWTHUB_INSPECTION_KEY_EXPORT_FILE"
_EXPORT_UID = "BAWTHUB_INSPECTION_KEY_EXPORT_UID"

log = logging.getLogger(__name__)
_key: bytes | None = None


def export_inspection_key() -> None:
    """Publish a fresh key from the process that serves MCP tools.

    run_server() calls this whether llm-mcp-server runs standalone or on
    llm-service's thread; the tool signs with this process-local copy.

    Only a file path and reader uid are configurable. No key is accepted in
    the environment or inherited by a standalone MCP process.
    """
    global _key
    _key = None
    path = os.getenv(_EXPORT_FILE, "").strip()
    if not path:
        return
    target = Path(path)
    try:
        key = secrets.token_bytes(32)
        publish(target, key.hex(), os.environ, uid_env=_EXPORT_UID)
    except (ValueError, OSError):
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        log.warning("BawtHub inspection key unavailable")
        return
    _key = key


def inspection_key_available() -> bool:
    return _key is not None


def sign_inspection_tasks(turn: TaskTurnContext, viewport: str) -> tuple[bytes, str] | None:
    """Bind an already verified request-local turn to one bounded operation.

    This takes no claimed subject, operation or URL from the model. Callers
    must obtain the turn via current_task_turn_context(), never parse tool args.
    """
    key = _key
    if key is None or turn.inspection is None or not is_inspection_turn_id(turn.turn_id):
        return None
    account_id, origin, admitted_at = turn.inspection
    if origin != inspection_origin() or viewport not in ("phone", "desktop"):
        return None
    issued_at = time.time_ns() // 1_000_000
    # Scope inspection to recent owner web admission without changing the
    # existing lifetime of ordinary task-turn capabilities.
    if admitted_at > turn.issued_at or not inspection_is_fresh(admitted_at, issued_at // 1000):
        return None
    body = json.dumps(
        {
            "op": "observe_tasks",
            "aud": "bawthub-internal",
            "iat": issued_at,
            "exp": issued_at + _TTL_MS,
            "nonce": secrets.token_hex(16),
            "origin": origin,
            "accountId": account_id,
            "userId": turn.user_id,
            "botId": turn.bot_id,
            "viewport": viewport,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    if len(body) > _MAX_BODY:
        return None
    return body, hmac.digest(key, _PURPOSE + body, "sha256").hex()
