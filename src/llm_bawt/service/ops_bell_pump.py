"""Deliver staged ops bell entries to BawtHub (TASK-1183).

Woken by the commit-wake hub when a job transition stages a row
(:func:`llm_bawt.ops.bell.stage`); the backstop timer also picks up retries
that come due. Each row is POSTed once to BawtHub's internal route, which
upserts on ``ops-job-<id>-<phase>``, so a resend after a lost reply cannot add
a second entry. After a delivery the ``notifications`` state topic tells open
UIs to refetch the bell; nothing polls.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from ..integrations.notify_bell import bell_http
from ..ops import bell
from . import state_events
from .commit_wake import BACKSTOP_SECONDS, wait_for_wake

logger = logging.getLogger(__name__)

OPS_BELL_PATH = "/internal/notifications/ops"
BATCH_SIZE = 20
#: Without a wake hub, or for retries coming due between commits.
RETRY_POLL_SECONDS = 30.0
#: 404: an older BawtHub without the route, mid-rollout. Retry until it arrives.
_RETRYABLE_4XX = frozenset({404, 408, 425, 429})


async def deliver(engine: Any, http: httpx.AsyncClient, *, limit: int = BATCH_SIZE) -> tuple[int, int]:
    """One pass: (claimed, delivered). Never raises for a single row."""
    rows = await asyncio.to_thread(bell.claim_due, engine, limit=limit)
    delivered = 0
    for row in rows:
        if row["payload"] is None:
            await asyncio.to_thread(bell.give_up, engine, row["id"], "job not found")
            continue
        error = None
        try:
            response = await http.post(OPS_BELL_PATH, json=row["payload"])
            if not response.is_success:
                error = f"HTTP {response.status_code}"
                if 400 <= response.status_code < 500 and response.status_code not in _RETRYABLE_4XX:
                    # Refused, not unavailable (bad payload, nobody to tell): a resend can't help.
                    logger.warning("ops bell %s refused: %s", row["event_key"], error)
                    await asyncio.to_thread(bell.give_up, engine, row["id"], error)
                    continue
        except httpx.HTTPError as exc:
            error = type(exc).__name__
        if error:
            logger.warning("ops bell %s not delivered: %s", row["event_key"], error)
        else:
            delivered += 1
        await asyncio.to_thread(bell.settle, engine, row["id"], ok=error is None,
                                attempts=row["attempts"], error=error)
    return len(rows), delivered


async def run_ops_bell_pump(service: Any, store_factory, *, http_factory=bell_http) -> None:
    """The app loop. ``store_factory`` returns the ops store, which owns the outbox table."""
    store = await asyncio.to_thread(store_factory)
    engine = getattr(store, "engine", None)
    if engine is None:
        return
    hub = getattr(service, "_commit_wake_hub", None)
    wake = hub.subscribe() if hub is not None else None
    # Retries come due on the backstop timer, not a commit.
    interval = BACKSTOP_SECONDS if wake is not None else RETRY_POLL_SECONDS
    async with http_factory() as http:
        while True:
            claimed = delivered = 0
            try:
                claimed, delivered = await deliver(engine, http)
                if delivered:
                    state_events.mark(service, state_events.NOTIFICATIONS)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - staged rows wait for the next pass
                logger.exception("ops bell pass failed")
            # A full batch may leave more; a delivered start may release its finish.
            if claimed >= BATCH_SIZE or delivered:
                continue
            await wait_for_wake(wake, interval)
