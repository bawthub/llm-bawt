"""Per-house command routing (TASK-2077, houses T6).

A *house* is an isolated bridge slot. A bot profile's ``house`` field (empty =
mansion) is the only routing input. The mansion keeps the historical
``agent:commands`` stream, so with zero houses every key below collapses to
today's behaviour.

Keys (all plain Redis strings unless noted):

``agent:commands:house:<H>``            house H's command stream
``agent:house:<H>:lease:<backend>``     liveness lease, value = epoch, TTL
``agent:house:<H>:epoch:<backend>``     INCR'd by each bridge start
``agent:turn:req:<request_id>``         stream a running turn was admitted on
``agent:turn:sess:<backend>:<session>`` same, by session (steer/answer)

Fencing: the dispatch decision for a new ``chat.send`` is made exactly once,
at publish time. Nothing is ever republished or claimed across streams, so an
in-flight turn never moves. A ``chat.send`` routed to a house carries the
lease epoch it saw; a house bridge refuses one from an older epoch, so a
command queued while the house was dead cannot run late after a restart.
"""

from __future__ import annotations

import os
import re

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from .publisher import MANSION_COMMANDS_STREAM

logger = logging.getLogger(__name__)

HOUSE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,30}$")
HOUSE_ENV = "BAWT_HOUSE"
LEASE_TTL_S = 30
LEASE_REFRESH_S = 10
TURN_TTL_S = 90
HOUSE_STREAM_PREFIX = "agent:commands:house:"
EPOCH_FIELD = "house_epoch"
# TASK-2080: the app asks a vacated house to restart itself; see HouseSlot.recycle.
RECYCLE_ACTION = "house.recycle"
RECYCLE_POLL_S = 1.0
_RELEASE_LEASE_LUA = (
    "if redis.call('GET', KEYS[1]) == ARGV[1] then "
    "return redis.call('DEL', KEYS[1]) else return 0 end"
)


def normalize_house(value: object) -> str:
    """Return a valid house id, '' for the mansion; raise ValueError otherwise."""
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    if not HOUSE_ID_RE.fullmatch(text):
        raise ValueError(
            "house must match ^[a-z0-9][a-z0-9-]{0,30}$ (empty = mansion)"
        )
    return text


def commands_stream(house: str) -> str:
    return f"{HOUSE_STREAM_PREFIX}{house}" if house else MANSION_COMMANDS_STREAM


def lease_key(house: str, backend: str) -> str:
    return f"agent:house:{house}:lease:{backend}"


def epoch_key(house: str, backend: str) -> str:
    return f"agent:house:{house}:epoch:{backend}"


def turn_request_key(request_id: str) -> str:
    return f"agent:turn:req:{request_id}"


def turn_session_key(backend: str, session_key: str) -> str:
    return f"agent:turn:sess:{backend}:{session_key}"


def house_from_env(environ=None) -> str:
    """The bridge's own slot from ``BAWT_HOUSE``; a bad value fails startup."""
    import os

    env = os.environ if environ is None else environ
    return normalize_house(env.get(HOUSE_ENV, ""))


def fallback_notice_key(house: str, backend: str) -> str:
    return f"agent:house:{house}:fallback_notified:{backend}"


FALLBACK_NOTICE_TTL_S = 900

FallbackNotifier = Callable[[str, str, str], Awaitable[None]]
"""``(house, backend, bot_id)``: tell the user new work fell back to the mansion."""


# ---------------------------------------------------------------- app side


async def route_new_command(
    redis: Any,
    *,
    house: str | None,
    backend: str | None,
    bot_id: str | None,
    notify: FallbackNotifier | None = None,
) -> tuple[str, str | None]:
    """Pick the stream for a NEW ``chat.send``: ``(stream, epoch)``.

    No house -> the mansion, no Redis read (zero-house byte-identical). A
    house with a live lease for this backend -> its stream plus the lease
    epoch the house bridge will fence on. Otherwise the mansion, with one
    deduplicated notice per outage. Callers decide once; nothing re-routes.
    """
    try:
        house = normalize_house(house)
    except ValueError:
        logger.warning("bot %s has an invalid house %r; using the mansion", bot_id, house)
        return MANSION_COMMANDS_STREAM, None
    if not house:
        return MANSION_COMMANDS_STREAM, None
    backend = (backend or "").strip()
    epoch = await redis.get(lease_key(house, backend)) if backend else None
    if epoch:
        return commands_stream(house), str(epoch)
    logger.warning("house %s/%s has no live lease; bot %s falls back to the mansion",
                   house, backend or "?", bot_id)
    if notify is not None and backend:
        try:
            first = await redis.set(fallback_notice_key(house, backend), "1",
                                    nx=True, ex=FALLBACK_NOTICE_TTL_S)
            if first:
                await notify(house, backend, bot_id or "")
        except Exception:
            logger.exception("house fallback notice failed: house=%s", house)
    return MANSION_COMMANDS_STREAM, None


async def follow_up_stream(
    redis: Any,
    *,
    request_id: str | None = None,
    backend: str | None = None,
    session_key: str | None = None,
) -> str:
    """Where a follow-up (steer/answer/approval/abort) for a turn must go.

    Every housed bot's turn has a record (house or mansion fallback); a
    missing record means the mansion (always true with zero houses). A
    follow-up naming its request uses only that request's record, never a
    session record that may belong to an older turn.
    """
    keys: list[str] = []
    if request_id:
        keys.append(turn_request_key(request_id))
    elif backend and session_key:
        keys.append(turn_session_key(backend, session_key))
    try:
        for key in keys:
            stream = await redis.get(key)
            if stream:
                return str(stream)
    except Exception:
        # A failed lookup must not block an abort: house turns are only ever
        # reached through their record, so the mansion is the safe default.
        logger.warning("turn location lookup failed; using the mansion", exc_info=True)
    return MANSION_COMMANDS_STREAM


def recycle_command(house: str, backend: str, epoch: str) -> dict:
    """The ``house.recycle`` entry for one lease holder (app side, TASK-2080).

    Sent on the house's own command stream, stamped with the live lease epoch,
    so only the bridge holding the lease now acts on it. A house with no live
    lease is down or already recycling; it comes back on a fresh container.
    """
    return {"action": RECYCLE_ACTION, "backend": backend, "house": house, EPOCH_FIELD: epoch}


# ------------------------------------------------------------- bridge side


class HouseSlot:
    """A bridge's house membership: lease heartbeat, epoch fence, turn records.

    For the mansion (``house == ''``) every method is a no-op and ``admit``
    accepts everything, so mansion bridges behave exactly as before.
    """

    GRACE_S = 10.0

    def __init__(self, house: str, backend: str, queue: Any) -> None:
        self.house = normalize_house(house)
        self.backend = backend
        self.stream = commands_stream(self.house)
        self._queue = queue
        self.epoch: str | None = None
        self._turns: dict[tuple[str, str], float] = {}
        self._task: asyncio.Task | None = None
        self.superseded = False
        self.recycling = False
        # Set once a recycle decided to exit: nothing is admitted after it.
        self.closed = False
        self._recycle_tasks: set[asyncio.Task] = set()
        # What this house's preview serves (house_preview.PreviewPublisher).
        self._preview: Any = None

    @property
    def is_house(self) -> bool:
        return bool(self.house)

    async def start(self, redis: Any) -> None:
        if not self.is_house or self._task is not None:
            return
        self._redis = redis
        self.epoch = str(await redis.incr(epoch_key(self.house, self.backend)))
        await redis.set(lease_key(self.house, self.backend), self.epoch, ex=LEASE_TTL_S)
        await redis.delete(fallback_notice_key(self.house, self.backend))
        logger.info("house %s/%s lease held, epoch %s", self.house, self.backend, self.epoch)
        self._task = asyncio.create_task(self._heartbeat())
        # TASK-2135: the house serves its occupant's preview (or a page saying
        # why there is none) from its own startup, so a restart brings it back.
        # The pool renderer turns it on; tests and ad hoc bridges leave it off.
        if os.environ.get("BAWT_HOUSE_PREVIEW", "").strip().lower() == "on":
            from .house_preview_supervisor import PreviewSupervisor
            self._supervisor = PreviewSupervisor(self.house, redis)
            self._supervisor.start()

    def admit(self, fields: dict) -> bool:
        """False for a send routed under an older epoch (fenced, never run)."""
        if not self.is_house:
            return True
        if self.superseded or self.closed:
            return False
        return (fields.get(EPOCH_FIELD) or "") == (self.epoch or "-")

    def track(self, fields: dict) -> None:
        if not self.is_house:
            return
        request_id = (fields.get("request_id") or "").strip()
        session_key = (fields.get("session_key") or "").strip()
        if not request_id or not session_key:
            return
        self._turns[(session_key, request_id)] = time.monotonic()
        asyncio.create_task(self._write_turn(session_key, request_id))

    async def _write_turn(self, session_key: str, request_id: str) -> None:
        try:
            await self._redis.set(turn_request_key(request_id), self.stream, ex=TURN_TTL_S)
            await self._redis.set(turn_session_key(self.backend, session_key), self.stream,
                                  ex=TURN_TTL_S)
        except Exception:
            logger.exception("turn record write failed: request=%s", request_id)

    async def refresh(self) -> None:
        """One heartbeat: renew the lease and live turns, drop finished ones.

        Only the current owner renews: a lease holding a newer epoch means a
        replacement started, so this slot stops admitting and renewing.
        """
        key = lease_key(self.house, self.backend)
        current = await self._redis.get(key)
        if current and int(current) > int(self.epoch or 0):
            if not self.superseded:
                logger.warning("house %s/%s superseded by epoch %s (ours %s)",
                               self.house, self.backend, current, self.epoch)
            self.superseded = True
            return
        if not self.recycling:
            # A recycling house released its lease and must not take it back.
            await self._redis.set(key, self.epoch, ex=LEASE_TTL_S)
            await self._publish_preview()
        now = time.monotonic()
        for (session_key, request_id), seen in list(self._turns.items()):
            live = self._queue.request_phase(session_key, request_id) is not None
            if live or now - seen < self.GRACE_S:
                # Restore a record that expired while we couldn't heartbeat:
                # the request is ours outright; the session only if no newer
                # turn has claimed it meanwhile.
                if not await self._redis.expire(turn_request_key(request_id), TURN_TTL_S):
                    await self._redis.set(turn_request_key(request_id), self.stream, ex=TURN_TTL_S)
                sess = turn_session_key(self.backend, session_key)
                if not await self._redis.expire(sess, TURN_TTL_S):
                    await self._redis.set(sess, self.stream, nx=True, ex=TURN_TTL_S)
                continue
            self._turns.pop((session_key, request_id), None)
            await self._redis.delete(turn_request_key(request_id))
            sess_key = turn_session_key(self.backend, session_key)
            if await self._redis.get(sess_key) == self.stream and not any(
                    s == session_key for s, _ in self._turns):
                await self._redis.delete(sess_key)

    async def _publish_preview(self) -> None:
        """Campus preview status (TASK-2131); never fails the heartbeat."""
        try:
            if self._preview is None:
                from .house_preview import PreviewPublisher
                self._preview = PreviewPublisher(self.house)
            await self._preview.publish(self._redis)
        except Exception:
            logger.warning("house %s preview probe failed", self.house, exc_info=True)

    def busy(self) -> bool:
        """A turn admitted here is running, or was admitted within GRACE_S.

        The grace covers a send between ``gate_send`` and its task's queue
        registration, when the queue does not know it yet.
        """
        now = time.monotonic()
        return any(
            self._queue.request_phase(session_key, request_id) is not None
            or now - seen < self.GRACE_S
            for (session_key, request_id), seen in list(self._turns.items())
        )

    async def recycle(self, fields: dict, msg_id: str, redis: Any, *, group: str,
                      exit_process: Callable[[], None]) -> bool:
        """Restart this house for a fresh container, never under a running turn.

        Run as its own task so the listener keeps serving the in-flight turn's
        steers and aborts. Only the current lease holder acts: the control is
        acked and the lease released first (new work for this house now falls
        back to the mansion), then the bridge drains and exits. Docker's
        restart policy brings it back; its tmpfs paths start empty. Returns
        True when it exited.

        Drain: every entry after the control is either unread (group lag),
        read but not yet admitted (still pending), or admitted (tracked, so
        ``busy``). The lag is read before the pending range, so an entry
        delivered between the two reads is caught by the second. The final
        ``busy`` check, closing admission and the exit run with no await in
        between, so the listener cannot admit a send after the decision; one
        it reads later is refused at receipt (nothing runs).
        """
        stale = (not self.is_house or self.superseded or self.recycling
                 or (fields.get("backend") or "") != self.backend
                 or (fields.get(EPOCH_FIELD) or "") != (self.epoch or "-"))
        if stale:
            logger.info("house %s ignored a recycle for epoch %s (ours %s)",
                        self.house, fields.get(EPOCH_FIELD), self.epoch)
            await redis.xack(self.stream, group, msg_id)
            return False
        self.recycling = True
        await redis.xack(self.stream, group, msg_id)
        await redis.eval(_RELEASE_LEASE_LUA, 1, lease_key(self.house, self.backend), self.epoch)
        logger.info("house %s/%s recycling: lease released, draining", self.house, self.backend)
        while True:
            unread = await self._unread(redis, group)
            pending = await self._pending_after(redis, group, msg_id)
            # No await from here to exit_process: the decision is atomic with
            # respect to the listener on this event loop.
            if not unread and not pending and not self.busy():
                self.closed = True
                logger.info("house %s/%s idle; exiting for a fresh container",
                            self.house, self.backend)
                exit_process()
                return True
            await asyncio.sleep(RECYCLE_POLL_S)

    async def _unread(self, redis: Any, group: str) -> int:
        """Entries on this house's stream not yet read by its group."""
        try:
            for info in await redis.xinfo_groups(self.stream):
                if info.get("name") == group:
                    lag = info.get("lag")
                    if lag is None:
                        # Redis can't compute lag (e.g. after deletions):
                        # never treat unknown as drained.
                        return 1
                    return int(lag)
        except Exception:
            logger.warning("house %s stream lag unreadable; treating as busy", self.house,
                           exc_info=True)
            return 1
        return 0

    async def _pending_after(self, redis: Any, group: str, msg_id: str) -> int:
        """Entries after ``msg_id`` read by the group but not yet acked.

        Sends ahead of the control were read (count=1, in order) before it and
        are tracked; only later ones can be read but not yet admitted. Older
        pending entries (a predecessor process's leftovers) never block.
        """
        try:
            return len(await redis.xpending_range(
                self.stream, group, min=f"({msg_id}", max="+", count=1))
        except Exception:
            logger.warning("house %s pending entries unreadable; treating as busy",
                           self.house, exc_info=True)
            return 1

    async def _heartbeat(self) -> None:
        while True:
            await asyncio.sleep(LEASE_REFRESH_S)
            try:
                await self.refresh()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("house %s heartbeat failed", self.house)


def exit_self() -> None:
    """Graceful bridge shutdown: both ``__main__`` loops stop on SIGTERM."""
    import os
    import signal

    os.kill(os.getpid(), signal.SIGTERM)


def is_recycle(slot: "HouseSlot | None", fields: dict) -> bool:
    """A ``house.recycle`` this bridge's house slot handles (else plain-acked)."""
    return (fields.get("action") == RECYCLE_ACTION
            and slot is not None and slot.is_house)


def start_recycle(slot: "HouseSlot", fields: dict, msg_id: str, redis: Any, *,
                  group: str) -> asyncio.Task:
    """Listener entry for ``house.recycle``: drain and exit in its own task."""
    task = asyncio.create_task(
        slot.recycle(fields, msg_id, redis, group=group, exit_process=exit_self))
    # Hold a reference: the loop keeps only weak references to tasks.
    slot._recycle_tasks.add(task)
    task.add_done_callback(slot._recycle_tasks.discard)
    return task


async def gate_send(slot: "HouseSlot | None", fields: dict, msg_id: str, redis: Any, *,
                    group: str, publisher: Any) -> bool:
    """Admit a ``chat.send`` read by this bridge; refuse a fenced one.

    A refused send gets the run's ``done`` sentinel (``disconnect=refused``,
    nothing ran) and is acked, so the app's wait ends and it never runs.
    """
    if slot is None or slot.admit(fields):
        if slot is not None:
            slot.track(fields)
        return True
    from .run_contract import REFUSED

    request_id = (fields.get("request_id") or "").strip()
    logger.warning("house %s fenced a stale send: request=%s epoch=%s current=%s",
                   slot.house, request_id, fields.get(EPOCH_FIELD), slot.epoch)
    if request_id:
        publisher.publish_run_done(request_id, teardown=REFUSED)
    await redis.xack(slot.stream, group, msg_id)
    return False
