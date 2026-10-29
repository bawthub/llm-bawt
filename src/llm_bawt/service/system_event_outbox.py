"""Bounded, restart-safe projection of committed system events to UI streams."""

from __future__ import annotations

import asyncio
import logging

import redis

from ..system_events import SystemEventEmitter

logger = logging.getLogger(__name__)
OUTBOX_INTERVAL_SECONDS = 2
PRUNE_EVERY_PASSES = 1800
BATCH_SIZE = 50


async def run_system_event_outbox(
    emitter: SystemEventEmitter, redis_url: str, *, report_only: bool = False,
) -> None:
    """One app-owned loop; conditional row locks also support multiple app replicas.

    Redis and SQLAlchemy are synchronous here, so the entire batch runs in a
    worker thread. Redis timeouts bound how long a locked batch holds a DB slot.
    """
    client = redis.Redis.from_url(redis_url, decode_responses=True,
                                  socket_connect_timeout=2, socket_timeout=2)
    passes = 0
    try:
        while True:
            try:
                count = await asyncio.to_thread(emitter.publish_batch, client,
                                                limit=BATCH_SIZE, report_only=report_only)
                if report_only and count:
                    logger.info("system event outbox report-only: %d pending in first batch", count)
                passes += 1
                if passes % PRUNE_EVERY_PASSES == 0:
                    pruned = await asyncio.to_thread(emitter.prune, report_only=report_only)
                    logger.info("system event retention %s: %d rows",
                                "report-only" if report_only else "pruned", pruned)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - committed rows await the next pass
                logger.exception("system event outbox pass failed")
            await asyncio.sleep(OUTBOX_INTERVAL_SECONDS)
    finally:
        await asyncio.to_thread(client.close)
