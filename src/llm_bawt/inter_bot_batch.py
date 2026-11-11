"""Idle inbox batching: one turn drains a target's contiguous plain backlog.

TASK-1252 S1. When a bot is idle and its FIFO head is about to start a
fallback turn, the queued rows directly behind it ride the same turn instead
of each starting their own. The head stays the turn's one owner: its turn_id,
reservation, lease, heartbeat, recovery and restart retry are unchanged.
Followers are marked ``DISPATCHING`` with ``delivery_mode='batched'`` and
``batch_head_id``, carry no lease, and never head the queue. A database
trigger settles them with their head, as one unit, on every head status
change (any code path): head DELIVERED -> followers DELIVERED; head QUEUED,
or FAILED pending a restart retry -> followers released to QUEUED; any other
FAILED or CANCELLED -> followers take the same terminal status.

Opt-in per bot through the ``inbox_batch`` runtime setting (default off).
Only plain rows join a batch; a waited send, steer-only, reset, scheduled,
routed, approval, voice, Send-now or control row ends the batch, as does a
different user or a non-text message. The head is always admitted.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text as sa_text

from .inter_bot_delivery_ledger import DeliveryTransitionEvents

SETTING_KEY = "inbox_batch"
MAX_ROWS = 8
MAX_CHARS = 24_000

# Metadata keys that mark a row as needing its own turn.
_SOLO_METADATA = (
    "ordinary_chat", "prompt_schedule", "message_route", "new_session_intent",
    "new_session_seed", "control", "send_now", "approval_request_id",
    "approval_continuation_id", "ops_resume", "original_delivery_id",
    "failed_delivery_id",
)
# Payload keys that mark a row as needing its own turn.
_SOLO_PAYLOAD = (
    "steer_only", "voice_input", "session_id", "inter_bot_reset_only",
    "inter_bot_session_policy", "inter_bot_seed_session_id",
)


def enabled(conn, target: str) -> bool:
    """The bot's ``inbox_batch`` runtime setting, read in the claim transaction.

    Off when the settings table doesn't exist yet (a store that never
    bootstrapped runtime settings), so the claim never fails on it.
    """
    if conn.execute(sa_text("SELECT to_regclass('runtime_settings')")).scalar() is None:
        return False
    row = conn.execute(sa_text("""
        SELECT value_json FROM runtime_settings
        WHERE scope_type='bot' AND scope_id=:target AND key=:key
        ORDER BY id DESC LIMIT 1
    """), {"target": target, "key": SETTING_KEY}).first()
    if row is None:
        return False
    try:
        return json.loads(row[0]) is True
    except (TypeError, ValueError):
        return False


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def last_text(payload: dict[str, Any]) -> str | None:
    """The row's final user message text, or None if it is not plain text."""
    messages = payload.get("messages") or []
    if not messages or not isinstance(messages[-1], dict):
        return None
    content = messages[-1].get("content")
    return content if isinstance(content, str) else None


def plain(row: dict[str, Any], *, user_id: str, config) -> bool:
    """Whether ``row`` may share a turn with other rows."""
    from .inter_bot_delivery_user import resolve_delivery_user

    payload, meta = _as_dict(row.get("payload_json")), _as_dict(row.get("metadata_json"))
    if (row.get("session_policy") or "continue") != "continue":
        return False
    if payload.get("reply_mode") == "waited" or any(payload.get(key) for key in _SOLO_PAYLOAD):
        return False
    if any(meta.get(key) for key in _SOLO_METADATA):
        return False
    if last_text(payload) is None:
        return False
    try:
        return resolve_delivery_user(config, payload.get("user")) == user_id
    except ValueError:
        return False


def take_followers(conn, head: dict[str, Any], *, user_id: str, config) -> list[dict[str, Any]]:
    """Lock and return the plain QUEUED rows directly behind ``head``.

    Caller holds the target admission lock and has decided ``head`` starts an
    idle turn. Stops at the first row that is not ready, not plain, or would
    exceed the row or character cap; the head itself is always admitted.
    """
    from .service.approval_fanout import claim_fence

    head_text = last_text(_as_dict(head.get("payload_json")))
    if head_text is None or not plain(head, user_id=user_id, config=config):
        return []
    rows = conn.execute(sa_text("""
        SELECT * FROM inter_bot_deliveries
        WHERE target_bot_id=:target AND lane='normal' AND ordinal > :ordinal
          AND status IN ('QUEUED','STEERING','DISPATCHING')
        ORDER BY ordinal
        LIMIT :limit
        FOR UPDATE
    """), {"target": head["target_bot_id"], "ordinal": head["ordinal"],
           "limit": MAX_ROWS - 1}).mappings().all()
    taken: list[dict[str, Any]] = []
    total = len(head_text)
    for row in rows:
        if (row["status"] != "QUEUED" or row.get("delivery_mode") is not None
                or row.get("next_retry_at") is not None or not plain(row, user_id=user_id, config=config)):
            break
        if row["available_at"] > datetime.now(timezone.utc):
            break
        if claim_fence(conn, row) is not None:
            break
        total += len(last_text(_as_dict(row["payload_json"])) or "")
        if total > MAX_CHARS:
            break
        taken.append(dict(row))
    return taken


def combined_prompt(head: dict[str, Any], followers: list[dict[str, Any]]) -> str:
    """One user message carrying every batched message, oldest first."""
    rows = [head, *followers]
    parts = [
        f"[Inbox: {len(rows)} messages arrived while you were idle, oldest first. "
        "Handle each one; send any reply to its sender separately.]"
    ]
    for index, row in enumerate(rows, 1):
        text = last_text(_as_dict(row["payload_json"])) or ""
        parts.append(f"--- {index}/{len(rows)} · delivery {row['id']} ---\n{text}")
    return "\n\n".join(parts)


def attach(conn, head: dict[str, Any], followers: list[dict[str, Any]],
           payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Mark ``followers`` as riding ``head``'s turn; return the batch payload and prompt."""
    prompt = combined_prompt(head, followers)
    ids = [f["id"] for f in followers]
    updated = conn.execute(sa_text("""
        UPDATE inter_bot_deliveries SET status='DISPATCHING', delivery_mode='batched',
            batch_head_id=:head_id, target_turn_id=:turn_id,
            dispatch_started_at=now(), next_retry_at=NULL, last_error=NULL, updated_at=now()
        WHERE id = ANY(:ids) AND status='QUEUED'
        RETURNING *
    """), {"head_id": head["id"], "turn_id": head["turn_id"], "ids": ids}).mappings().all()
    DeliveryTransitionEvents.stage(conn, *updated, reason_code="inbox_batched")
    conn.execute(sa_text("""
        UPDATE inter_bot_deliveries SET metadata_json=COALESCE(metadata_json, '{}'::jsonb)
            || jsonb_build_object('batch_member_ids', CAST(:ids AS jsonb),
                                  'batch_prompt', CAST(:prompt AS text))
        WHERE id=:id
    """), {"id": head["id"], "ids": json.dumps(ids), "prompt": prompt})
    messages = list(payload.get("messages") or [])
    messages[-1] = {**messages[-1], "content": prompt}
    return {**payload, "messages": messages, "inter_bot_delivery_ids": [head["id"], *ids]}, prompt


def stage_followers(conn, head_id: str) -> None:
    """Stage ledger events for followers the settle trigger moved in this transaction."""
    rows = conn.execute(sa_text("""
        SELECT * FROM inter_bot_deliveries
        WHERE batch_head_id=:id AND updated_at=now()
    """), {"id": head_id}).mappings().all()
    if rows:
        DeliveryTransitionEvents.stage(conn, *rows, reason_code="inbox_batch_settled")


# The head->followers settle trigger. Its follower writes are staged for the
# ledger by ``stage_followers``, which ``DeliveryTransitionEvents.stage`` calls
# for every head transition; it lives outside a function on purpose, since
# the ledger scan pairs each in-function status write with a stage() after it.
_SETTLE_FUNCTION = """
        CREATE OR REPLACE FUNCTION inter_bot_deliveries_settle_batch() RETURNS trigger AS $$
        BEGIN
            IF NEW.status = OLD.status
               OR NOT (COALESCE(OLD.metadata_json, '{}'::jsonb) ? 'batch_member_ids') THEN
                RETURN NEW;
            END IF;
            IF NEW.status = 'DELIVERED' THEN
                UPDATE inter_bot_deliveries SET status='DELIVERED',
                    delivered_at=COALESCE(delivered_at, now()), updated_at=now()
                WHERE batch_head_id=NEW.id AND status='DISPATCHING' AND delivery_mode='batched';
            ELSIF NEW.status = 'QUEUED'
                  OR (NEW.status = 'FAILED'
                      AND COALESCE(NEW.metadata_json->>'restart_retry_pending', '') = 'true') THEN
                UPDATE inter_bot_deliveries SET status='QUEUED', delivery_mode=NULL,
                    target_turn_id=NULL, available_at=now(), next_retry_at=NULL,
                    last_error='released with its inbox batch head', updated_at=now()
                WHERE batch_head_id=NEW.id AND status='DISPATCHING' AND delivery_mode='batched';
            ELSIF NEW.status IN ('FAILED', 'CANCELLED') THEN
                UPDATE inter_bot_deliveries SET status=NEW.status,
                    last_error=COALESCE(NEW.last_error, 'inbox batch head ended'), updated_at=now()
                WHERE batch_head_id=NEW.id AND status='DISPATCHING' AND delivery_mode='batched';
            ELSE
                RETURN NEW;
            END IF;
            IF NEW.status <> 'DELIVERED' THEN
                NEW.metadata_json := NEW.metadata_json - 'batch_member_ids' - 'batch_prompt';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
    """


def ensure_batch_schema(conn) -> None:
    """Follower column, index and the head->followers settle trigger."""
    conn.execute(sa_text(
        "ALTER TABLE inter_bot_deliveries ADD COLUMN IF NOT EXISTS batch_head_id VARCHAR(96)"
    ))
    conn.execute(sa_text("""
        CREATE INDEX IF NOT EXISTS ix_inter_bot_delivery_batch_head
        ON inter_bot_deliveries (batch_head_id) WHERE batch_head_id IS NOT NULL
    """))
    conn.execute(sa_text(_SETTLE_FUNCTION))
    conn.execute(sa_text("""
        CREATE OR REPLACE TRIGGER inter_bot_deliveries_batch_settle
        BEFORE UPDATE OF status ON inter_bot_deliveries
        FOR EACH ROW EXECUTE FUNCTION inter_bot_deliveries_settle_batch()
    """))
