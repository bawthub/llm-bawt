"""Bounded, restart-safe projection of committed system events to UI streams."""

from __future__ import annotations

import asyncio
import logging
import time

import redis

from ..system_events import SystemEventEmitter
from .commit_wake import BACKSTOP_SECONDS, wait_for_wake

logger = logging.getLogger(__name__)
OUTBOX_INTERVAL_SECONDS = 2
PRUNE_EVERY_SECONDS = 3600  # was every 1800 passes of the old 2s loop
BATCH_SIZE = 50


async def run_system_event_outbox(
    emitter: SystemEventEmitter, redis_url: str, *, report_only: bool = False, hub=None,
) -> None:
    """One app-owned loop; conditional row locks also support multiple app replicas.

    Redis and SQLAlchemy are synchronous here, so the entire batch runs in a
    worker thread. Redis timeouts bound how long a locked batch holds a DB slot.
    """
    client = redis.Redis.from_url(redis_url, decode_responses=True,
                                  socket_connect_timeout=2, socket_timeout=2)
    wake = hub.subscribe() if hub is not None else None
    # Woken on each commit (TASK-2108); the timer only repairs a missed notification.
    interval = BACKSTOP_SECONDS if wake is not None else OUTBOX_INTERVAL_SECONDS
    next_prune = time.monotonic() + PRUNE_EVERY_SECONDS
    try:
        while True:
            count = 0
            try:
                count = await asyncio.to_thread(emitter.publish_batch, client,
                                                limit=BATCH_SIZE, report_only=report_only)
                if report_only and count:
                    logger.info("system event outbox report-only: %d pending in first batch", count)
                if time.monotonic() >= next_prune:
                    next_prune = time.monotonic() + PRUNE_EVERY_SECONDS
                    pruned = await asyncio.to_thread(emitter.prune, report_only=report_only)
                    logger.info("system event retention %s: %d rows",
                                "report-only" if report_only else "pruned", pruned)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - committed rows await the next pass
                logger.exception("system event outbox pass failed")
            if count >= BATCH_SIZE and not report_only:
                continue  # more rows are waiting; drain before sleeping
            await wait_for_wake(wake, interval)
    finally:
        await asyncio.to_thread(client.close)
