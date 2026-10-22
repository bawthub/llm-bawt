"""App-side loop that writes B1 (b)/(c) exit evidence for held quiesce members (TASK-1233).

Each pass reads the undrained members of every live QUIESCING job's current
gate (:func:`..ops.quiesce_exit.undrained_members`), asks each member's own
backend ``run.status`` (:meth:`.recovery.orphan_turns.BridgeRunProbe.status`,
the one RPC path, at most :data:`CONCURRENCY` at once), reads the run's
terminal record, and applies :func:`..ops.quiesce_exit.judge`: a live answer
keeps a sighting, proof is stamped (turn row, else member row), anything
ambiguous writes nothing. The driver then drains on that record like any other.

It runs in the app because the app owns the Redis transport the bridges
answer on. Passes start :data:`INTERVAL_SECONDS` apart, or back to back when
a pass runs longer. A pass lasts at most :data:`PASS_BUDGET_SECONDS`: no probe
starts after ``PASS_BUDGET_SECONDS - PROBE_TIMEOUT_SECONDS -
WRITE_RESERVE_SECONDS``. Members are taken least recently probed first, so a
member left out of one pass leads the next.

Deadlines (Loopy r1 A1/A2). A member's network work, ``run.status`` (its
publication included) plus the terminal record read, runs under one
``PROBE_TIMEOUT_SECONDS`` deadline inside its slot; a member that misses it is
ambiguous and writes nothing. Its writes then run off the slot, bounded by
the pass deadline (:class:`..ops.quiesce_exit.WriteDeadline`): a write that
can't finish by then gives up and writes nothing. The pass never cancels a
member that has started writing; it waits for it, so nothing is written after
:meth:`QuiesceExitSweep.run_once` returns. Only a member still in its network
phase at the budget is cancelled (``cut``); with the reserve that takes
scheduling drift, never a member that started on time.

Bound (Vex r1 S1, r2 R1). One member holds a slot for at most one probe
timeout, so a pass covers at least :data:`PASS_CAPACITY` members
(:func:`pass_capacity`): each of the ``CONCURRENCY`` slots starts a probe at
0, T+e, 2(T+e), ... and a start later than ``PASS_BUDGET_SECONDS - T - R`` is
deferred, so with every probe timing out a slot fits
``ceil((PASS_BUDGET_SECONDS - R - T) / T)`` (T the probe timeout, R the write
reserve, e the scheduling drift, always > 0). Up to that many held members, each is probed at least every
``2 * PASS_BUDGET_SECONDS`` <= ``QUIESCE_SWEEP_INTERVAL_MAX_S``, the gap that
sizes ``RUN_TERMINAL_RETENTION_S``. Beyond it the gap grows to about
``ceil(n / PASS_CAPACITY)`` passes; the pass logs a warning when it defers a
member, and the terminal record's retention margin
(``RUN_TERMINAL_RETENTION_MARGIN_S``) absorbs the excess first. Not leader-only: every stamp is conditional and idempotent, so two app
processes only repeat no-ops. A failed pass logs and retries; it never kills
the loop.

Abort. When the judgement asks for it (``Judgement.abort``: ``not_found``, no
holder record, no terminal), the sweep sends one backend-matched ``chat.abort``
per member and answering boot. It runs outside the pass: it is not held by a
probe slot or the budget, so it never shrinks :data:`PASS_CAPACITY`, and it
waits up to ``ABORT_RPC_TIMEOUT_S`` because a bridge replies only after the
cancelled run's teardown. That is a hard bound on the whole call, publication
included (Loopy r1 B1), and an abort whose member is no longer held is
cancelled at the next pass. Its reply is logged, never stamped: the evidence
is the refused or cancelled run's own terminal.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any, Callable

from agent_bridge.run_contract import QUIESCE_SWEEP_INTERVAL_MAX_S, RunContractError, RunTerminal

from ..ops.quiesce_exit import (SweepMember, WriteDeadline, ensure_sighting_schema, judge, purge_sightings,
                                record_sighting, stamp, undrained_members)
from .run_accepts import purge_accepts
from .dependencies import get_ops_service
from .recovery.orphan_turns import BridgeRunProbe

log = logging.getLogger(__name__)

INTERVAL_SECONDS = 10.0
PROBE_TIMEOUT_SECONDS = 5.0
CONCURRENCY = 8
PASS_BUDGET_SECONDS = QUIESCE_SWEEP_INTERVAL_MAX_S / 2 - PROBE_TIMEOUT_SECONDS
"""Longest pass; two passes plus slack fit ``QUIESCE_SWEEP_INTERVAL_MAX_S``."""
WRITE_RESERVE_SECONDS = 1.0
"""Time kept at the end of a pass for the writes of the last members probed."""


def pass_capacity(budget_seconds: float, probe_timeout_seconds: float, concurrency: int,
                  write_reserve_seconds: float = 0.0) -> int:
    """Members one pass is guaranteed to probe even when every probe times out."""
    span = budget_seconds - write_reserve_seconds - probe_timeout_seconds
    return concurrency * max(0, math.ceil(span / probe_timeout_seconds))


PASS_CAPACITY = pass_capacity(PASS_BUDGET_SECONDS, PROBE_TIMEOUT_SECONDS, CONCURRENCY, WRITE_RESERVE_SECONDS)
"""32 at the defaults (8 slots × 4 probes in a 25s budget less the 1s write reserve)."""
PURGE_EVERY_SECONDS = 3600.0
DEFERRED = "deferred"
CUT = "cut"
WRITE_DEADLINE = "write_deadline"


class QuiesceExitSweep:
    """One pass at a time over the held members; see the module docstring."""

    def __init__(self, engine, *, subscriber: Callable[[], Any] | None = None,
                 probe: BridgeRunProbe | None = None, concurrency: int = CONCURRENCY):
        if subscriber is None:
            from ..agent_backends.agent_bridge import get_agent_subscriber

            subscriber = get_agent_subscriber
        self.engine = engine
        self._subscriber = subscriber
        self.probe = probe or BridgeRunProbe(subscriber, timeout_seconds=PROBE_TIMEOUT_SECONDS)
        self._limit = asyncio.Semaphore(concurrency)
        self._probed_at: dict[int, float] = {}
        self._aborts: dict[tuple[int, str], asyncio.Task] = {}
        self._writing: set[asyncio.Task] = set()
        self.budget_seconds = PASS_BUDGET_SECONDS
        self.write_reserve_seconds = WRITE_RESERVE_SECONDS

    async def _terminal(self, request_id: str) -> RunTerminal | None:
        subscriber = self._subscriber()
        if subscriber is None:
            return None
        try:
            return await subscriber.read_run_terminal(request_id)
        except RunContractError as exc:
            log.warning("quiesce exit sweep: malformed terminal record for %s: %s", request_id, exc)
            return None

    async def _one(self, member: SweepMember, start_by: float = float("inf"), deadline: float | None = None) -> str:
        if member.unstampable:
            # The pin and the turn row disagree, so neither evidence home
            # would take a stamp (stamp_turn_exit refuses it): don't probe.
            return "pin_mismatch"
        async with self._limit:
            if time.monotonic() > start_by:
                return DEFERRED
            self._probed_at[member.member_id] = time.monotonic()
            reply = terminal = None
            try:
                # One deadline for the member's whole network phase (Loopy r1 A1).
                async with asyncio.timeout(PROBE_TIMEOUT_SECONDS):
                    reply, _result = await self.probe.status(backend=member.backend, session_key=member.session_key,
                                                             request_id=member.request_id, label=member.turn_id)
                    if reply is not None and member.request_id:
                        terminal = await self._terminal(member.request_id)
            except TimeoutError:
                # A reply without its terminal read is not a judgement: ambiguous.
                log.info("quiesce exit sweep: member %s turn %s: no complete answer within %.1fs",
                         member.member_id, member.turn_id, PROBE_TIMEOUT_SECONDS)
                reply = terminal = None
        verdict = judge(member, reply, terminal)
        if verdict.abort:
            self._abort(member, reply.boot_id)
        if verdict.sighting is None and verdict.record is None:
            return verdict.reason
        # From here the pass waits for this member instead of cancelling it:
        # cancelling the coroutine would not stop the write thread (Loopy r1 A2).
        task = asyncio.current_task()
        self._writing.add(task)
        try:
            return await asyncio.to_thread(self._write, member, verdict, deadline)
        finally:
            self._writing.discard(task)

    def _write(self, member: SweepMember, verdict, deadline: float | None) -> str:
        try:
            if verdict.sighting is not None:
                record_sighting(self.engine, verdict.sighting, deadline=deadline)
            if verdict.record is not None:
                stamped = stamp(self.engine, member, verdict.record, deadline=deadline)
                log.info("quiesce exit sweep: member %s turn %s %s (source %s, exited %s, stamped %s)",
                         member.member_id, member.turn_id, verdict.reason, verdict.record["source"],
                         verdict.record["exited"], stamped)
        except WriteDeadline as exc:
            log.warning("quiesce exit sweep: member %s turn %s %s not written: %s",
                        member.member_id, member.turn_id, verdict.reason, exc)
            return WRITE_DEADLINE
        return verdict.reason

    def _abort(self, member: SweepMember, boot_id: str) -> None:
        """Start this member's ``chat.abort`` unless one already went to this boot."""
        key = (member.member_id, boot_id)
        if key in self._aborts:
            return

        async def send() -> None:
            from agent_bridge import abort_contract

            reply = None
            try:
                # The sweep bounds the call itself, whatever the probe does (Loopy r1 B1).
                async with asyncio.timeout(abort_contract.ABORT_RPC_TIMEOUT_S):
                    reply = await self.probe.abort(backend=member.backend, session_key=member.session_key,
                                                   request_id=member.request_id)
            except TimeoutError:
                pass
            log.info("quiesce exit sweep: member %s turn %s chat.abort %s -> %s", member.member_id,
                     member.turn_id, member.request_id, "unconfirmed" if reply is None else reply.state.value)

        self._aborts[key] = asyncio.create_task(send())

    async def settle_aborts(self) -> None:
        """Wait for every abort in flight (tests; :meth:`aclose` cancels them instead)."""
        await asyncio.gather(*(t for t in self._aborts.values() if not t.done()), return_exceptions=True)

    async def aclose(self) -> None:
        pending = [t for t in self._aborts.values() if not t.done()]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    async def run_once(self) -> dict[str, int]:
        """One bounded pass; returns a count per verdict reason (``deferred``/``cut`` for the budget)."""
        started = time.monotonic()
        members = await asyncio.to_thread(undrained_members, self.engine)
        live = {m.member_id for m in members}
        self._probed_at = {k: v for k, v in self._probed_at.items() if k in live}
        await self._retire_aborts(live)
        members.sort(key=lambda m: (self._probed_at.get(m.member_id, 0.0), m.member_id))
        deadline = started + self.budget_seconds
        start_by = deadline - PROBE_TIMEOUT_SECONDS - self.write_reserve_seconds
        tasks = [asyncio.create_task(self._one(m, start_by, deadline)) for m in members]
        counts: dict[str, int] = {}
        if not tasks:
            return counts
        try:
            _done, pending = await asyncio.wait(tasks, timeout=max(0.0, deadline - time.monotonic()))
            self._cut(pending)
            if pending:
                # Cut members end at once; members writing end by the write deadline.
                await asyncio.wait(pending)
        except asyncio.CancelledError:
            await self._join(tasks)  # never leave a write running past the pass
            raise
        for task in tasks:
            if task.cancelled():
                outcome = CUT
            elif task.exception() is not None:
                log.error("quiesce exit sweep: member probe failed", exc_info=task.exception())
                outcome = "error"
            else:
                outcome = task.result()
            counts[outcome] = counts.get(outcome, 0) + 1
        left = counts.get(DEFERRED, 0) + counts.get(CUT, 0) + counts.get(WRITE_DEADLINE, 0)
        if left:
            log.warning("quiesce exit sweep: %d of %d held members left for the next pass (budget %.0fs)",
                        left, len(members), self.budget_seconds)
        return counts

    async def _join(self, tasks) -> None:
        """Cut the members still probing and wait for every member writing, through any further cancel.

        A repeated cancel of the pass would otherwise interrupt the wait and
        return while a write thread is still in its transaction (Loopy r2).
        The caller re-raises the cancellation once this returns.
        """
        while True:
            self._cut(tasks)
            pending = [task for task in tasks if not task.done()]
            if not pending:
                return
            try:
                await asyncio.wait(pending)
            except asyncio.CancelledError:
                continue

    def _cut(self, tasks) -> None:
        """Cancel every member still in its network phase; a member that is writing is waited for."""
        for task in tasks:
            if not task.done() and task not in self._writing:
                task.cancel()

    async def _retire_aborts(self, live: set[int]) -> None:
        """Cancel and drop the aborts of members no longer held; keep the rest (once per member and boot)."""
        stale = [task for key, task in self._aborts.items() if key[0] not in live]
        for task in stale:
            task.cancel()
        if stale:
            await asyncio.gather(*stale, return_exceptions=True)
        self._aborts = {key: task for key, task in self._aborts.items() if key[0] in live}


def build_quiesce_exit_sweep(service) -> QuiesceExitSweep | None:
    """The sweep for this app, or None without a PostgreSQL ops store."""
    engine = getattr(get_ops_service(service.config).store, "engine", None)
    if getattr(getattr(engine, "dialect", None), "name", None) != "postgresql":
        return None
    ensure_sighting_schema(engine)
    return QuiesceExitSweep(engine)


async def run_quiesce_exit_sweep(service, *, interval_seconds: float = INTERVAL_SECONDS) -> None:
    """Lifespan loop: passes start ``interval_seconds`` apart; holder records purged hourly.

    The interval is capped at the pass budget, so the per-member bound in the
    module docstring holds whatever the caller passes.
    """
    interval_seconds = min(interval_seconds, PASS_BUDGET_SECONDS)
    sweep = None
    purged_at = 0.0
    try:
        while True:
            started = time.monotonic()
            try:
                sweep = sweep or build_quiesce_exit_sweep(service)
                if sweep is None:
                    log.info("quiesce exit sweep disabled: no PostgreSQL ops store")
                    return
                await sweep.run_once()
                if started - purged_at >= PURGE_EVERY_SECONDS:
                    await asyncio.to_thread(purge_sightings, sweep.engine)
                    await asyncio.to_thread(purge_accepts, sweep.engine)
                    purged_at = started
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("quiesce exit sweep pass failed; retrying")
            await asyncio.sleep(max(0.0, interval_seconds - (time.monotonic() - started)))
    finally:
        if sweep is not None:
            await sweep.aclose()


__all__ = ["CONCURRENCY", "INTERVAL_SECONDS", "PASS_BUDGET_SECONDS", "PASS_CAPACITY", "PROBE_TIMEOUT_SECONDS",
           "WRITE_RESERVE_SECONDS", "pass_capacity",
           "QuiesceExitSweep",
           "build_quiesce_exit_sweep", "run_quiesce_exit_sweep"]
