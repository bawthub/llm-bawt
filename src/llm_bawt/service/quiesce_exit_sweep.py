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
starts after ``PASS_BUDGET_SECONDS - PROBE_TIMEOUT_SECONDS`` and whatever is
still running at the budget is cancelled. Members are taken least recently
probed first, so a member left out of one pass leads the next.

Bound (Vex r1 S1, r2 R1). One member costs at most one probe timeout, so a
pass covers at least :data:`PASS_CAPACITY` members (:func:`pass_capacity`):
each of the ``CONCURRENCY`` slots starts a probe at 0, T+e, 2(T+e), ... and a
start later than ``PASS_BUDGET_SECONDS - T`` is deferred, so with every probe
timing out a slot fits ``ceil((PASS_BUDGET_SECONDS - T) / T)`` (T the probe
timeout, e the scheduling drift, always > 0). Up to that many held members, each is probed at least every
``2 * PASS_BUDGET_SECONDS`` <= ``QUIESCE_SWEEP_INTERVAL_MAX_S``, the gap that
sizes ``RUN_TERMINAL_RETENTION_S``. Beyond it the gap grows to about
``ceil(n / PASS_CAPACITY)`` passes; the pass logs a warning when it defers a
member, and the terminal record's retention margin
(``RUN_TERMINAL_RETENTION_MARGIN_S``) absorbs the excess first. Not leader-only: every stamp is conditional and idempotent, so two app
processes only repeat no-ops. A failed pass logs and retries; it never kills
the loop.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any, Callable

from agent_bridge.run_contract import QUIESCE_SWEEP_INTERVAL_MAX_S, RunContractError, RunTerminal

from ..ops.quiesce_exit import (SweepMember, ensure_sighting_schema, judge, purge_sightings, record_sighting, stamp,
                                undrained_members)
from .dependencies import get_ops_service
from .recovery.orphan_turns import BridgeRunProbe

log = logging.getLogger(__name__)

INTERVAL_SECONDS = 10.0
PROBE_TIMEOUT_SECONDS = 5.0
CONCURRENCY = 8
PASS_BUDGET_SECONDS = QUIESCE_SWEEP_INTERVAL_MAX_S / 2 - PROBE_TIMEOUT_SECONDS
"""Longest pass; two passes plus slack fit ``QUIESCE_SWEEP_INTERVAL_MAX_S``."""


def pass_capacity(budget_seconds: float, probe_timeout_seconds: float, concurrency: int) -> int:
    """Members one pass is guaranteed to probe even when every probe times out."""
    return concurrency * max(0, math.ceil((budget_seconds - probe_timeout_seconds) / probe_timeout_seconds))


PASS_CAPACITY = pass_capacity(PASS_BUDGET_SECONDS, PROBE_TIMEOUT_SECONDS, CONCURRENCY)
"""32 at the defaults (8 slots × 4 probes in a 25s budget)."""
PURGE_EVERY_SECONDS = 3600.0
DEFERRED = "deferred"


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
        self.budget_seconds = PASS_BUDGET_SECONDS

    async def _terminal(self, request_id: str) -> RunTerminal | None:
        subscriber = self._subscriber()
        if subscriber is None:
            return None
        try:
            return await subscriber.read_run_terminal(request_id)
        except RunContractError as exc:
            log.warning("quiesce exit sweep: malformed terminal record for %s: %s", request_id, exc)
            return None

    async def _one(self, member: SweepMember, start_by: float = float("inf")) -> str:
        if member.unstampable:
            # The pin and the turn row disagree, so neither evidence home
            # would take a stamp (stamp_turn_exit refuses it): don't probe.
            return "pin_mismatch"
        async with self._limit:
            if time.monotonic() > start_by:
                return DEFERRED
            self._probed_at[member.member_id] = time.monotonic()
            reply, _result = await self.probe.status(backend=member.backend, session_key=member.session_key,
                                                     request_id=member.request_id, label=member.turn_id)
            terminal = None
            if reply is not None and member.request_id:
                terminal = await self._terminal(member.request_id)
        verdict = judge(member, reply, terminal)
        if verdict.sighting is not None:
            await asyncio.to_thread(record_sighting, self.engine, verdict.sighting)
        if verdict.record is not None:
            stamped = await asyncio.to_thread(stamp, self.engine, member, verdict.record)
            log.info("quiesce exit sweep: member %s turn %s %s (source %s, exited %s, stamped %s)",
                     member.member_id, member.turn_id, verdict.reason, verdict.record["source"],
                     verdict.record["exited"], stamped)
        return verdict.reason

    async def run_once(self) -> dict[str, int]:
        """One bounded pass; returns a count per verdict reason (``deferred``/``cut`` for the budget)."""
        started = time.monotonic()
        members = await asyncio.to_thread(undrained_members, self.engine)
        live = {m.member_id for m in members}
        self._probed_at = {k: v for k, v in self._probed_at.items() if k in live}
        members.sort(key=lambda m: (self._probed_at.get(m.member_id, 0.0), m.member_id))
        start_by = started + self.budget_seconds - PROBE_TIMEOUT_SECONDS
        tasks = [asyncio.create_task(self._one(m, start_by)) for m in members]
        counts: dict[str, int] = {}
        if not tasks:
            return counts
        try:
            done, pending = await asyncio.wait(
                tasks, timeout=max(0.0, started + self.budget_seconds - time.monotonic()))
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            raise
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
            counts["cut"] = len(pending)
        for task in done:
            outcome = task.exception() or task.result()
            if isinstance(outcome, BaseException):
                log.error("quiesce exit sweep: member probe failed", exc_info=outcome)
                outcome = "error"
            counts[outcome] = counts.get(outcome, 0) + 1
        if counts.get(DEFERRED) or counts.get("cut"):
            log.warning("quiesce exit sweep: %d of %d held members left for the next pass (budget %.0fs)",
                        counts.get(DEFERRED, 0) + counts.get("cut", 0), len(members), self.budget_seconds)
        return counts


def build_quiesce_exit_sweep(service) -> QuiesceExitSweep | None:
    """The sweep for this app, or None without a PostgreSQL ops store."""
    engine = getattr(get_ops_service(service.config).store, "engine", None)
    if getattr(getattr(engine, "dialect", None), "name", None) != "postgresql":
        return None
    ensure_sighting_schema(engine)
    return QuiesceExitSweep(engine)


async def run_quiesce_exit_sweep(service, *, interval_seconds: float = INTERVAL_SECONDS) -> None:
    """Lifespan loop: passes start ``interval_seconds`` apart; sightings purged hourly.

    The interval is capped at the pass budget, so the per-member bound in the
    module docstring holds whatever the caller passes.
    """
    interval_seconds = min(interval_seconds, PASS_BUDGET_SECONDS)
    sweep = None
    purged_at = 0.0
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
                purged_at = started
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("quiesce exit sweep pass failed; retrying")
        await asyncio.sleep(max(0.0, interval_seconds - (time.monotonic() - started)))


__all__ = ["CONCURRENCY", "INTERVAL_SECONDS", "PASS_BUDGET_SECONDS", "PASS_CAPACITY", "PROBE_TIMEOUT_SECONDS", "pass_capacity",
           "QuiesceExitSweep",
           "build_quiesce_exit_sweep", "run_quiesce_exit_sweep"]
