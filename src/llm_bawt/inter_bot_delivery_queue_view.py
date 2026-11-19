"""Per-bot delivery queue read model (TASK-1308 C0b).

One snapshot per agent bot: what is waiting (count, oldest age, head), what is
in flight, whether a turn is open, and the last day's failures. A pure read in
one connection: it takes no locks and decides nothing. It reports facts only;
why a head is not being claimed stays with the claim policy (C3).

Prompt-schedule and user-chat rows are included: they share the FIFO.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from sqlalchemy import text

from .inter_bot_delivery_steer_fence import steer_uncertain

PREVIEW_CHARS = 120
RECENT_FAILURES = 5

_AGENTS_SQL = "SELECT slug FROM bot_profiles WHERE bot_type = 'agent' ORDER BY slug"

_WAITING_SQL = """
    SELECT target_bot_id, count(*) AS n, min(created_at) AS oldest
    FROM inter_bot_deliveries
    WHERE status = 'QUEUED' AND target_bot_id = ANY(:bots)
    GROUP BY target_bot_id
"""

_HEAD_SQL = """
    SELECT DISTINCT ON (target_bot_id)
        target_bot_id, id, sender_bot_id, author_entity_type, author_entity_id,
        message_kind, delivery_mode, lane, available_at, attempt_count, created_at,
        left(message, :preview) AS preview,
        (metadata_json -> 'prompt_schedule') IS NOT NULL AS prompt_schedule
    FROM inter_bot_deliveries
    WHERE status = 'QUEUED' AND target_bot_id = ANY(:bots)
    ORDER BY target_bot_id, ordinal
"""

_IN_FLIGHT_SQL = """
    SELECT target_bot_id, id, status, delivery_mode, attempt_count, sender_bot_id,
           dispatch_started_at, transport_accepted_at, lease_expires_at, metadata_json
    FROM inter_bot_deliveries
    WHERE status IN ('STEERING', 'DISPATCHING') AND target_bot_id = ANY(:bots)
    ORDER BY target_bot_id, ordinal
"""

_OPEN_TURN_SQL = """
    SELECT DISTINCT bot_id FROM turn_logs
    WHERE ended_at IS NULL AND bot_id = ANY(:bots)
"""

_FAILURE_COUNT_SQL = """
    SELECT target_bot_id, count(*) AS n
    FROM inter_bot_deliveries
    WHERE status IN ('FAILED', 'CANCELLED') AND target_bot_id = ANY(:bots)
      AND updated_at >= now() - interval '24 hours'
    GROUP BY target_bot_id
"""

_FAILURES_SQL = """
    SELECT target_bot_id, id, status, sender_bot_id, updated_at, last_error FROM (
        SELECT *, row_number() OVER (PARTITION BY target_bot_id
                                     ORDER BY updated_at DESC, ordinal DESC) AS rn
        FROM inter_bot_deliveries
        WHERE status IN ('FAILED', 'CANCELLED') AND target_bot_id = ANY(:bots)
          AND updated_at >= now() - interval '24 hours'
    ) f WHERE rn <= :limit
    ORDER BY target_bot_id, updated_at DESC, ordinal DESC
"""


def _iso(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def _steer_uncertain(row: Any) -> bool:
    record = SimpleNamespace(delivery_mode=row.delivery_mode, metadata=row.metadata_json or {})
    return steer_uncertain(record) is not None


def delivery_queues(engine: Any) -> dict[str, Any]:
    """The queue snapshot for every agent bot, in slug order."""
    with engine.connect() as conn:
        now = conn.execute(text("SELECT now()")).scalar_one()
        bots = [r.slug for r in conn.execute(text(_AGENTS_SQL))]
        params = {"bots": bots}
        waiting = {r.target_bot_id: r for r in conn.execute(text(_WAITING_SQL), params)}
        heads = {r.target_bot_id: r for r in conn.execute(
            text(_HEAD_SQL), {**params, "preview": PREVIEW_CHARS})}
        in_flight: dict[str, list[dict[str, Any]]] = {}
        for r in conn.execute(text(_IN_FLIGHT_SQL), params):
            in_flight.setdefault(r.target_bot_id, []).append({
                "id": r.id, "status": r.status, "mode": r.delivery_mode,
                "attempt": r.attempt_count, "sender": r.sender_bot_id,
                "dispatch_started_at": _iso(r.dispatch_started_at),
                "transport_accepted_at": _iso(r.transport_accepted_at),
                "lease_expires_at": _iso(r.lease_expires_at),
                "steer_uncertain": _steer_uncertain(r),
            })
        open_turns = {r.bot_id for r in conn.execute(text(_OPEN_TURN_SQL), params)}
        failure_counts = {r.target_bot_id: r.n for r in conn.execute(text(_FAILURE_COUNT_SQL), params)}
        failures: dict[str, list[dict[str, Any]]] = {}
        for r in conn.execute(text(_FAILURES_SQL), {**params, "limit": RECENT_FAILURES}):
            failures.setdefault(r.target_bot_id, []).append({
                "id": r.id, "status": r.status, "sender": r.sender_bot_id,
                "at": _iso(r.updated_at), "last_error": (r.last_error or "")[:200] or None,
            })

    queues = []
    for bot in bots:
        w, h = waiting.get(bot), heads.get(bot)
        queues.append({
            "bot_id": bot,
            "waiting": {
                "count": w.n if w else 0,
                "oldest_age_seconds": (now - w.oldest).total_seconds() if w else None,
                "head": None if h is None else {
                    "id": h.id, "sender": h.sender_bot_id,
                    "author_type": h.author_entity_type, "author": h.author_entity_id,
                    "kind": h.message_kind, "mode": h.delivery_mode, "lane": h.lane,
                    "prompt_schedule": bool(h.prompt_schedule),
                    "available_at": _iso(h.available_at), "attempt": h.attempt_count,
                    "created_at": _iso(h.created_at), "preview": h.preview,
                },
            },
            "in_flight": in_flight.get(bot, []),
            "open_turn": bot in open_turns,
            "recent_failures": {"count": failure_counts.get(bot, 0), "latest": failures.get(bot, [])},
        })
    return {"now": _iso(now), "queues": queues}
