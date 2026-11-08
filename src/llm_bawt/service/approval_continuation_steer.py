"""Steer an MCP approval result into a busy bot's running turn (TASK-1357).

An approved MCP tool has already run server-side, so its result is only news
for the bot. Waiting for the target to go idle can hold that news for an hour
behind one long turn. When the target is busy, the continuation instead sends
the same result prompt as a ``steer_only`` inter-bot delivery, keyed on the
continuation id so retries reuse one row.

The continuation counts as delivered only once that steer reports started (the
target turn actually received it). A steer that is skipped because the turn
ended, rejected or left uncertain changes nothing here: the continuation keeps
its normal deferred idle-turn path. Harness approvals are not steered; their
one-shot grant must accompany the turn that retries the tool.
"""
from __future__ import annotations

import logging

from sqlalchemy import text as sa_text

from ..approval_policies import KIND_MCP
from ..bot_identities import SYSTEM_BOT_ID
from ..inter_bot_delivery_steer_fence import steer_started
from ..inter_bot_steer_only import STEER_ONLY_KEY

log = logging.getLogger(__name__)

MESSAGE_KIND = "APPROVAL_RESULT_STEER"


def steer_key(row) -> str:
    return f"approval-cont-steer:{row.id}"


def steerable(row) -> bool:
    return row.request_kind == KIND_MCP


def _dispatcher(service):
    dispatcher = getattr(service, "_inter_bot_dispatcher", None)
    if dispatcher is None or getattr(dispatcher.store, "engine", None) is None:
        return None
    return dispatcher


def steer_landed(service, row) -> bool:
    """Whether this continuation's steer row exists and its turn received it."""
    if not steerable(row):
        return False
    dispatcher = _dispatcher(service)
    if dispatcher is None:
        return False
    with dispatcher.store.engine.connect() as conn:
        found = conn.execute(sa_text(
            "SELECT delivery_mode, metadata_json FROM inter_bot_deliveries"
            " WHERE sender_bot_id=:sender AND target_bot_id=:target AND idempotency_key=:key"
        ), {"sender": SYSTEM_BOT_ID, "target": row.bot_id, "key": steer_key(row)}).mappings().first()
    return bool(found and steer_started(found))


def request_steer(service, row, prompt: str) -> None:
    """Enqueue (or re-find) the steer_only delivery carrying ``prompt``.

    Best effort: any failure is logged and the caller keeps the idle path.
    """
    if not steerable(row):
        return
    dispatcher = _dispatcher(service)
    if dispatcher is None:
        return
    try:
        record, duplicate = dispatcher.store.enqueue(
            sender_bot_id=SYSTEM_BOT_ID, target_bot_id=row.bot_id, message=prompt,
            idempotency_key=steer_key(row), message_kind=MESSAGE_KIND, on_gate="hold",
            metadata={"approval_continuation_id": row.id},
            payload={
                "bot_id": row.bot_id, "user": row.user_id, "session_id": None,
                "messages": [{"role": "user", "content": prompt}],
                "prefer_steer": True, STEER_ONLY_KEY: True, "reply_mode": "async",
                "extract_memory": False, "augment_memory": False, "stream": True,
            },
        )
    except Exception:
        log.exception("Approval continuation id=%s could not request a steer", row.id)
        return
    try:
        if duplicate:
            dispatcher.wake(record.target_bot_id)
        else:
            dispatcher.announce_admitted(record)
    except Exception:  # the row is durable; the dispatcher sweep still finds it
        log.exception("Approval steer announce failed for continuation id=%s", row.id)


__all__ = ["request_steer", "steer_key", "steer_landed", "steerable"]
