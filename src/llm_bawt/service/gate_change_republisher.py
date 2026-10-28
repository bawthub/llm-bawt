"""Re-announce waiting chat deliveries when a system gate changes (TASK-1676).

A queued chat send's label says why it waits (:mod:`.delivery_wait_reason`).
That reason changes without the row changing: a quiesce or freeze opens or
is released (often in the ops process, not this one), or its lease lapses.
This loop hears every open and release on ``LISTEN system_gate_changed``
(:func:`.turn_gates.notify_gate_change`) and arms a one-shot timer for the
earliest live gate's expiry. On either, it republishes the lifecycle event of
every QUEUED ordinary chat row, so each tab relabels through the normal live
event path, and wakes the dispatcher so a released head is claimed at once
rather than at the next recovery pass.

Between gate changes it does no work: with no live gate the listener just
waits on the socket (:data:`IDLE_WAIT_SECONDS` only bounds a reconnect). Any
failure is logged; the next change or timer retries.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text as sa_text

from ..ops.reconciler import JobCreatedWaiter
from .turn_gates import GATE_CHANGED_CHANNEL

log = logging.getLogger(__name__)

IDLE_WAIT_SECONDS = 600.0
EXPIRY_SLACK_SECONDS = 0.5
REPUBLISH_LIMIT = 500

_WAITING_CHAT = (
    "SELECT id FROM inter_bot_deliveries"
    " WHERE status='QUEUED' AND COALESCE(metadata_json, '{}'::jsonb) ? 'ordinary_chat'"
    " ORDER BY ordinal LIMIT :limit"
)
_NEXT_EXPIRY = "SELECT min(expires_at) FROM system_gates WHERE released_at IS NULL AND expires_at > now()"


def next_expiry_wait(engine, *, now: datetime | None = None) -> float | None:
    """Seconds until the earliest live gate lapses (plus slack), else None."""
    with engine.connect() as conn:
        expires = conn.execute(sa_text(_NEXT_EXPIRY)).scalar_one_or_none()
    if expires is None:
        return None
    now = now or datetime.now(timezone.utc)
    return max(0.0, (expires - now).total_seconds()) + EXPIRY_SLACK_SECONDS


def republish_waiting(dispatcher: Any) -> int:
    """Announce every QUEUED chat row's lifecycle again and wake the dispatcher."""
    store = dispatcher.store
    with store.engine.connect() as conn:
        ids = list(conn.execute(sa_text(_WAITING_CHAT), {"limit": REPUBLISH_LIMIT}).scalars())
    announced = 0
    for delivery_id in ids:
        record = store.get(delivery_id)
        if record is not None and record.status == "QUEUED":
            dispatcher.announce_lifecycle(record)
            announced += 1
    dispatcher.wake()
    return announced


async def run_gate_change_republisher(service: Any, *, waiter_factory=JobCreatedWaiter) -> None:
    """The app loop: wait for a gate change or expiry, then republish."""
    dispatcher = getattr(service, "_inter_bot_dispatcher", None)
    engine = getattr(getattr(dispatcher, "store", None), "engine", None)
    if engine is None or engine.dialect.name != "postgresql":
        return
    waiter = waiter_factory(engine, channel=GATE_CHANGED_CHANNEL)
    stop = threading.Event()
    deadline: float | None = None
    try:
        while True:
            try:
                deadline = await asyncio.to_thread(next_expiry_wait, engine)
            except Exception:  # noqa: BLE001 - a failed read waits for the next notify
                log.warning("gate expiry read failed", exc_info=True)
                deadline = None
            woke = await asyncio.to_thread(waiter.wait, stop, deadline if deadline is not None else IDLE_WAIT_SECONDS)
            if not woke and deadline is None:
                continue
            try:
                await asyncio.to_thread(republish_waiting, dispatcher)
            except Exception:  # noqa: BLE001 - the next change retries
                log.warning("could not republish waiting chat deliveries", exc_info=True)
    finally:
        stop.set()
        waiter.close()
