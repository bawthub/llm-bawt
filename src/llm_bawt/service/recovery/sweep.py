"""The startup recovery check (TASK-1232 Phase 2).

One procedure, run once per process: when this process takes dispatcher
leadership at startup, after the dispatcher's first ``recover_expired``
(``StartupRecoveryCheck``, the dispatcher's ``StartupCheck``). It never runs
on a timer. ``RecoverySweep.run`` is also callable directly; no HTTP route
calls it yet. A mutating pass holds this process's ``asyncio.Lock`` and a
PostgreSQL advisory transaction lock for its whole run; a pass that finds
either busy skips and reports ``busy``. A dry run takes no lock and changes
nothing.

``purge_ops`` is never a default step: only an explicit
``run(purge_ops=True)`` runs it. The startup check never passes it.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text as sa_text

from .audit import RunStamp
from .base import Plan, RecoveryStep, StepReport, SweepScope
from .maintenance import CancelledContinuationsStep, PostCheckStep, UnresolvedToolsStep
from .orphan_turns import OrphanTurnsStep, RunProbe
from .purge_ops import publish_purge_resolutions, purge_ops
from .reservations import OrphanedReservationsStep

logger = logging.getLogger(__name__)

RECOVERY_LOCK_KEY = "llm-bawt:recovery-sweep"


class PurgeOpsStep(RecoveryStep):
    """Explicit-only wrapper (``run(purge_ops=True)``): one transaction, not per candidate."""

    name = "purge_ops"

    async def detect(self, scope: SweepScope) -> list[dict[str, Any]]:
        return []

    async def plan(self, scope: SweepScope, candidate: dict[str, Any]) -> Plan:
        return Plan(None)

    async def run(self, scope: SweepScope) -> StepReport:
        result = await asyncio.to_thread(purge_ops, scope.engine, stamp=scope.stamp, dry_run=scope.dry_run)
        if not scope.dry_run:
            await publish_purge_resolutions(
                scope.engine, [row["id"] for row in result["approvals"] + result["unrecallable"]])
        report = StepReport(self.name)
        rows = result["approvals"] + result["jobs"]
        report.found = rows + result["unrecallable"]
        (report.planned if scope.dry_run else report.done).extend(rows)
        report.ambiguous = [{**row, "why": "continuation already dispatching or delivered"}
                            for row in result["unrecallable"]]
        return report


class RecoverySweep:
    # ``orphan_turns`` first: a settled turn leaves its target idle, which a
    # delivery reservation's release requires.
    DEFAULT_STEPS = ("orphan_turns", "reservations", "continuations", "unresolved_tools", "post_check")

    def __init__(self, engine, *, probe: RunProbe | None = None, started_at: datetime | None = None):
        """``started_at``: when this process started. ``orphan_turns`` only
        touches turns created before it and treats newer turns as this
        process's own (limits: ``orphan_turns`` module docstring)."""
        self.engine = engine
        self.started_at = started_at or datetime.now(timezone.utc)
        self._steps: dict[str, RecoveryStep] = {step.name: step for step in (
            OrphanTurnsStep(probe, started_at=self.started_at), OrphanedReservationsStep(),
            CancelledContinuationsStep(), UnresolvedToolsStep(), PostCheckStep(),
        )}
        self._lock = asyncio.Lock()

    @property
    def step_names(self) -> tuple[str, ...]:
        return tuple(self._steps)

    async def run(self, *, dry_run: bool, steps: list[str] | None = None, bots: list[str] | None = None,
                  purge_ops: bool = False, actor: str = "recovery_sweep") -> dict[str, Any]:
        unknown = sorted(set(steps or ()) - set(self._steps))
        if unknown:
            raise ValueError(f"unknown recovery steps: {unknown}")
        selected = [self._steps[name] for name in (steps or self.DEFAULT_STEPS)]
        if purge_ops:
            selected.insert(0, PurgeOpsStep())
        scope = SweepScope(engine=self.engine, now=datetime.now(timezone.utc), dry_run=dry_run,
                           bots=frozenset(bots) if bots else None, stamp=RunStamp.new(actor))
        if dry_run:
            return await self._run_steps(scope, selected)
        if self._lock.locked():
            return {"dry_run": False, "busy": True, "steps": []}
        async with self._lock:
            hold = await asyncio.to_thread(self._try_cluster_lock)
            if hold is None:
                return {"dry_run": False, "busy": True, "steps": []}
            try:
                return await self._run_steps(scope, selected)
            finally:
                await asyncio.to_thread(self._release_cluster_lock, hold)

    async def _run_steps(self, scope: SweepScope, steps: list[RecoveryStep]) -> dict[str, Any]:
        reports = []
        for step in steps:
            try:
                reports.append((await step.run(scope)).to_dict())
            except Exception as exc:  # one broken step never hides the others' results
                logger.exception("recovery step %s failed", step.name)
                reports.append({**StepReport(step.name).to_dict(), "error": f"{type(exc).__name__}: {exc}"})
        result = {"dry_run": scope.dry_run, "busy": False, "at": scope.now.isoformat(),
                  **scope.stamp.as_dict(), "steps": reports}
        counts = {r["step"]: {k: len(r[k]) for k in ("done", "ambiguous", "raced")} for r in reports}
        # Changes are worth a warning; anything else stays at debug (the
        # returned report carries it in full, and the startup check logs its
        # own summary).
        if not scope.dry_run and any(r["done"] or r.get("error") for r in reports):
            logger.warning("recovery sweep %s changed state: %s", scope.stamp.run_id, counts)
        else:
            logger.debug("recovery sweep %s: %s", scope.stamp.run_id, counts)
        return result

    def _try_cluster_lock(self):
        conn = self.engine.connect()
        transaction = conn.begin()
        if conn.dialect.name == "postgresql" and not conn.execute(
            sa_text("SELECT pg_try_advisory_xact_lock(hashtext(:key))"), {"key": RECOVERY_LOCK_KEY},
        ).scalar():
            transaction.rollback()
            conn.close()
            return None
        return conn, transaction

    @staticmethod
    def _release_cluster_lock(hold) -> None:
        conn, transaction = hold
        try:
            transaction.rollback()  # ends the transaction: releases the xact lock
        finally:
            conn.close()


class StartupRecoveryCheck:
    """The dispatcher's ``StartupCheck``: every default step, once, at startup.

    Idempotent: every step re-checks under its own fence, so a second run
    finds only what is still stuck. It always leaves one INFO line, so every
    boot shows what the check found and changed."""

    ACTOR = "recovery_startup"

    def __init__(self, sweep: RecoverySweep):
        self.sweep = sweep

    async def run_at_startup(self) -> dict[str, Any]:
        result = await self.sweep.run(dry_run=False, actor=self.ACTOR)
        if result.get("busy"):
            logger.info("startup recovery check skipped: another pass holds the sweep")
            return result
        summary = {step["step"]: {key: len(step[key]) for key in ("found", "done", "ambiguous", "raced")}
                   | ({"error": step["error"]} if step.get("error") else {})
                   for step in result["steps"]}
        logger.info("startup recovery check %s: %s", result.get("sweep_run_id"), summary)
        for step in result["steps"]:
            if step["step"] == "orphan_turns":
                for entry in step["ambiguous"]:
                    logger.info("startup recovery left open turn %s (%s): %s",
                                entry["turn_id"], entry["bot_id"], entry["why"])
        return result
