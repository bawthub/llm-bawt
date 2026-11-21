"""Inspection provenance from the first-party, cookie-authenticated BawtHub chat route."""

from __future__ import annotations

import json
import os
import re
import time
from typing import NamedTuple

from .api_auth.middleware import Principal

INSPECTION_ADMISSION_HEADER = "X-LLM-Bawt-Inspection-Admission"
INSPECTION_QUEUE_FIELD = "_inspection_web_admission"
MAX_INSPECTION_AGE_S = 5 * 60
_ACCOUNT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


class WebInspectionAdmission(NamedTuple):
    bot_id: str
    user_id: str
    account_id: str
    origin: str
    admitted_at: int


def inspection_is_fresh(admitted_at: int, now: int) -> bool:
    return type(admitted_at) is int and 0 <= now - admitted_at <= MAX_INSPECTION_AGE_S


def restore_queued_admission(raw, *, bot_id: str, user_id: str) -> WebInspectionAdmission | None:
    """Decode the private durable request envelope, never public request JSON."""
    if not isinstance(raw, (tuple, list)) or len(raw) != 5:
        return None
    bot, user, account, origin, admitted_at = raw
    if (
        bot != bot_id or user != user_id or not bot_id or not user_id
        or not isinstance(account, str) or not _ACCOUNT_RE.fullmatch(account)
        or not isinstance(origin, str) or origin != inspection_origin()
        or not inspection_is_fresh(admitted_at, int(time.time()))
    ):
        return None
    return WebInspectionAdmission(bot, user, account, origin, admitted_at)


def inspection_origin() -> str | None:
    return os.getenv("BAWTHUB_INSPECTION_ORIGIN", "").strip() or None


def read_web_admission(
    principal: object,
    header: str | None,
    *,
    bot_id: str | None,
    user_id: str | None,
) -> WebInspectionAdmission | None:
    """Only the authenticated web program may assert its own sign-in decision.

    The bridge, generic service bearer, MCP OAuth and log-mode anonymous callers
    cannot set this server-side proof. This never changes ordinary chat admission.
    """
    if principal != Principal("bawthub-web", "key") or not header or len(header) > 512:
        return None
    if not bot_id or not user_id:
        return None
    try:
        body = json.loads(header)
    except (ValueError, RecursionError):
        return None
    if not isinstance(body, list) or len(body) != 2:
        return None
    account_id, origin = body
    if not isinstance(account_id, str) or not _ACCOUNT_RE.fullmatch(account_id):
        return None
    if not isinstance(origin, str) or origin != inspection_origin():
        return None
    return WebInspectionAdmission(bot_id, user_id, account_id, origin, int(time.time()))
