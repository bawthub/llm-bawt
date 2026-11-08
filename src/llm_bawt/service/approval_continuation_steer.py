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
from types import SimpleNamespace

from sqlalchemy import text as sa_text

from ..approval_policies import KIND_MCP
from ..bot_identities import SYSTEM_BOT_ID
from ..inter_bot_delivery_steer_fence import steer_started, steer_uncertain
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


NONE = "none"          # no steer was ever requested: normal path
LANDED = "landed"      # the target turn received it: delivered
UNCERTAIN = "uncertain"  # accepted, outcome unknown: never resend
PENDING = "pending"    # not settled yet: wait, never race an idle turn
FALLBACK = "fallback"  # settled without ever reaching the bridge: idle path


def steer_lane(service, row) -> str:
    """Which lane owns this continuation's result, from its steer row.

    Only absorbing states decide. CANCELLED/FAILED are never re-queued, so a
    steer that ended there without ever sending a steer command (skipped,
    fenced, or last attempt proven rejected) can no longer inject the result,
    and the idle turn cannot duplicate it. Anything still live
    (QUEUED, STEERING, DISPATCHING, or DELIVERED awaiting its started frame)
    is PENDING. A steer accepted by the bridge that never proved started is
    UNCERTAIN and, like the delivery fence itself, is not replayed.
    """
    if not steerable(row):
        return NONE
    dispatcher = _dispatcher(service)
    if dispatcher is None:
        return NONE
    with dispatcher.store.engine.connect() as conn:
        found = conn.execute(sa_text(
            "SELECT status, delivery_mode, metadata_json, transport_accepted_at"
            " FROM inter_bot_deliveries"
            " WHERE sender_bot_id=:sender AND target_bot_id=:target AND idempotency_key=:key"
        ), {"sender": SYSTEM_BOT_ID, "target": row.bot_id, "key": steer_key(row)}).mappings().first()
    if found is None:
        return NONE
    if steer_started(found):
        return LANDED
    if steer_uncertain(SimpleNamespace(
        delivery_mode=found.get("delivery_mode"), metadata=found.get("metadata_json") or {},
    )):
        return UNCERTAIN
    if found.get("status") in ("CANCELLED", "FAILED"):
        # A terminal row still in 'steer' mode had a steer RPC sent whose
        # outcome may be unread (timeout, expired claim): a late started frame
        # can still land. Only a row that never entered a steer attempt (or
        # whose last attempt was proven rejected) is proof of non-delivery.
        if found.get("delivery_mode") == "steer" or found.get("transport_accepted_at") is not None:
            return UNCERTAIN
        return FALLBACK
    return PENDING


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


__all__ = ["FALLBACK", "LANDED", "NONE", "PENDING", "UNCERTAIN", "request_steer", "steer_key", "steer_lane", "steerable"]
