"""Best-effort, turn-exact accounting attribution (TASK-2019).

Snapshot the starting delivery or the sole currently assigned executable task,
before model execution. Never infer from text, a previous turn, or a whole
session. Explicit trusted association can replace the assignment snapshot.
This telemetry grants no task/conversation association authority.
"""
from __future__ import annotations

import asyncio
import logging

import httpx
from sqlalchemy import text

from ..bawthub_auth import service_auth
from ..turn_task import normalize_task_ref

logger = logging.getLogger(__name__)


def write_attribution(engine, *, turn_id, bot_id, user_id, task_ref, source, explicit=False):
    """A late start snapshot must never overwrite an explicit association."""
    if engine is None:
        return
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE turn_logs SET task_ref=:ref, task_attribution_source=:source
            WHERE id=:turn AND bot_id=:bot AND user_id=:user
              AND (:explicit OR task_attribution_source IS NULL)
        """), {"ref": normalize_task_ref(task_ref), "source": source,
               "turn": turn_id, "bot": bot_id, "user": user_id, "explicit": explicit})


def starting_delivery(engine, turn_id, bot_id, user_id):
    if engine is None:
        return None
    with engine.connect() as conn:
        return conn.execute(text("""
            SELECT task_id FROM inter_bot_deliveries
            WHERE turn_id=:turn AND target_bot_id=:bot
              AND payload_json::jsonb->>'user'=:user
            LIMIT 1
        """), {"turn": turn_id, "bot": bot_id, "user": user_id}).scalar_one_or_none()


def assigned_task(payload, bot_id):
    """Require a complete, unique result; never take the first of several."""
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        return None, "unavailable"
    rows = payload["tasks"]
    total = payload.get("total")
    if not isinstance(total, int) or isinstance(total, bool) or total < 0:
        return None, "unavailable"
    if total > 1:
        return None, "ambiguous"
    if total == 0 and not rows:
        return None, "unassigned"
    if total != 1 or len(rows) != 1:
        return None, "unavailable"
    row = rows[0]
    if (not isinstance(row, dict) or row.get("agentBotId") != bot_id
            or row.get("status") != "IN_PROGRESS" or row.get("archivedAt") is not None):
        return None, "unavailable"
    ref = normalize_task_ref(row.get("id"))
    return (ref, "active_assignment") if ref else (None, "unavailable")


async def stamp_turn_task(store, *, turn_id, bot_id, user_id):
    """Both stream paths call after admission, outside locks, before execution.

    Failure is visible in the source field, but never fails a chat turn. No
    provider calls, mutable task GET, association writes, or retries.
    """
    engine = getattr(store, "engine", None)
    if engine is None:
        return
    ref, source = None, "unavailable"
    try:
        # Accounting must not make an unhealthy task service hold model
        # execution. One total budget, not an unbounded stack of IO timeouts.
        async with asyncio.timeout(0.5):
            ref = await asyncio.to_thread(starting_delivery, engine, turn_id, bot_id, user_id)
            if ref:
                source = "delivery"
            else:
                from ..mcp_server.task_api import API_PREFIX, BASE_URL

                async with httpx.AsyncClient(
                    base_url=BASE_URL, timeout=0.5, auth=service_auth(),
                ) as client:
                    response = await client.get(f"{API_PREFIX}/tasks", params={
                        "agentBotId": bot_id, "status": "IN_PROGRESS",
                        "archived": "false", "limit": 2,
                    })
                    response.raise_for_status()
                    ref, source = assigned_task(response.json(), bot_id)
    except Exception:
        logger.warning("Task accounting snapshot unavailable for %s", turn_id, exc_info=True)
    try:
        await asyncio.to_thread(
            write_attribution, engine, turn_id=turn_id, bot_id=bot_id,
            user_id=user_id, task_ref=ref, source=source,
        )
    except Exception:
        logger.warning("Task accounting snapshot write failed for %s", turn_id, exc_info=True)
