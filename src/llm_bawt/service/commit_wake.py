"""Wake the app's outbox loops the moment work commits (TASK-2108).

Every delivery, ops-job and approval transition commits a ``system_events``
row through :class:`..system_events.CommitStagedAppends`. That hook also runs
``pg_notify(WAKE_CHANNEL)`` inside the same transaction, so Postgres delivers
the notification only if the work commits, from any process (app, MCP,
ops-reconciler). Other writers we own (home-audio enqueue) notify the same
channel.

One app listener (:func:`run_commit_wake_listener`) turns each notification
into a wake for every subscribed loop and the inter-bot dispatcher. The loops
keep a slow timer only as a backstop for a missed notification (listener down,
reconnect, an older process image still running). Every (re)connect also wakes
everyone once, since a commit made while nothing listened was not heard.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from sqlalchemy import text as sa_text

log = logging.getLogger(__name__)

WAKE_CHANNEL = "llm_bawt_work_committed"
#: Backstop for loops woken by :data:`WAKE_CHANNEL`; it only repairs missed notifications.
BACKSTOP_SECONDS = 60.0
IDLE_WAIT_SECONDS = 600.0
RECONNECT_SECONDS = 5.0


def notify_in_transaction(conn: Any, payload: str = "") -> None:
    """Queue a wake on ``conn``'s transaction; Postgres sends it only on commit.

    No-op on other dialects. Never raises: a wake never blocks its write.
    """
    try:
        if conn.dialect.name != "postgresql":
            return
        conn.execute(sa_text("SELECT pg_notify(:channel, :payload)"),
                     {"channel": WAKE_CHANNEL, "payload": payload})
    except Exception:  # noqa: BLE001 - the backstop timer still finds the work
        log.warning("commit wake notify failed; loops fall back to their backstop", exc_info=True)


class WakeHub:
    """Fan one wake out to every subscribed asyncio loop (thread-safe ``fire``)."""

    def __init__(self) -> None:
        self._events: list[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []
        self._lock = threading.Lock()
        self.fired = 0

    def subscribe(self) -> asyncio.Event:
        """A new event for the calling loop; set on every wake."""
        event = asyncio.Event()
        with self._lock:
            self._events.append((asyncio.get_running_loop(), event))
        return event

    def fire(self) -> None:
        self.fired += 1
        with self._lock:
            targets = list(self._events)
        for loop, event in targets:
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:  # loop closed (shutdown)
                pass


async def wait_for_wake(event: asyncio.Event | None, timeout: float) -> bool:
    """Wait for ``event`` or ``timeout``; True when woken. Clears the event.

    With no event (no hub), this is the plain backstop sleep.
    """
    if event is None:
        await asyncio.sleep(timeout)
        return False
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        return False
    finally:
        event.clear()


def hub_of(service: Any) -> WakeHub | None:
    return getattr(service, "_commit_wake_hub", None)


def subscribe(service: Any) -> asyncio.Event | None:
    hub = hub_of(service)
    return hub.subscribe() if hub is not None else None


def backstop(service: Any, legacy_seconds: float) -> float:
    """The loop's sleep: the backstop when a hub wakes it, else its old interval."""
    return BACKSTOP_SECONDS if hub_of(service) is not None else legacy_seconds


async def run_commit_wake_listener(service: Any, engine: Any, *, waiter_factory=None) -> None:
    """LISTEN on :data:`WAKE_CHANNEL`; wake the hub and dispatcher on each notify."""
    from .gate_change_republisher import GateChangeWaiter

    hub = hub_of(service)
    if hub is None or engine is None or engine.dialect.name != "postgresql":
        return
    waiter = (waiter_factory or GateChangeWaiter)(engine, channel=WAKE_CHANNEL)
    service._commit_wake_waiter = waiter  # exposed so acceptance can drop the connection
    stop = threading.Event()

    def wake_all() -> None:
        hub.fire()
        dispatcher = getattr(service, "_inter_bot_dispatcher", None)
        if dispatcher is not None:
            dispatcher.wake()

    try:
        while True:
            try:
                if await asyncio.to_thread(waiter.ensure_listening):
                    wake_all()  # anything committed while nothing listened
                if await asyncio.to_thread(waiter.wait, stop, IDLE_WAIT_SECONDS):
                    wake_all()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - loops run on their backstop meanwhile
                log.warning("commit wake listener dropped; reconnecting", exc_info=True)
                waiter.close()
                await asyncio.sleep(RECONNECT_SECONDS)
    finally:
        stop.set()
        waiter.close()


__all__ = ["BACKSTOP_SECONDS", "WAKE_CHANNEL", "WakeHub", "backstop", "hub_of", "notify_in_transaction",
           "run_commit_wake_listener", "subscribe", "wait_for_wake"]
