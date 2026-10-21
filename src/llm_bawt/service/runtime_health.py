"""Read-only capability health for the app-owned delivery and scheduling loops."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from time import monotonic

from sqlalchemy import text

from .health_checks import DEGRADED, DOWN, OK, _check


class LoopHealth:
    """Records actual completed sweeps, not a timer claiming the worker is alive."""

    def __init__(self):
        self.last_success = monotonic()
        self.error: str | None = None

    def succeeded(self):
        self.last_success = monotonic()
        self.error = None

    def failed(self, exc: Exception):
        self.error = f"{type(exc).__name__}: {exc}"

    def check(self, name, task, *, stale_seconds):
        if task is None or task.done():
            return _check(name, DOWN, "worker task is not running")
        if self.error:
            return _check(name, DEGRADED, f"last sweep failed: {self.error}")
        age = monotonic() - self.last_success
        if age > stale_seconds:
            return _check(name, DEGRADED, f"no completed sweep for {int(age)}s (busy or stalled)")
        return _check(name, OK)


class RuntimeHealthChecks:
    def __init__(self, service):
        self.service = service

    def deliveries(self, now=None):
        dispatcher = getattr(self.service, "_inter_bot_dispatcher", None)
        if dispatcher is None or dispatcher.store.engine is None:
            return _check("deliveries.queue", DOWN, "delivery dispatcher/database unavailable")
        leader = dispatcher._leader_task
        if leader is None or leader.done():
            return _check("deliveries.queue", DOWN, "delivery leadership task is not running")
        now = now or datetime.now(UTC)
        # Health describes live work, not historical terminal request failures.
        # Failed deliveries remain available in delivery history for diagnosis.
        # Aggregate the entire queue, not the newest page. Future backoffs are
        # intentional waits; active work is only suspect after its lease expires.
        with dispatcher.store.engine.connect() as conn:
            counts = conn.execute(text("""
                SELECT
                  SUM(CASE WHEN status='QUEUED' AND available_at <= :now
                            AND created_at < :stale THEN 1 ELSE 0 END) AS waiting,
                  SUM(CASE WHEN status IN ('STEERING','DISPATCHING')
                            AND lease_expires_at < :now THEN 1 ELSE 0 END) AS expired
                FROM inter_bot_deliveries
                WHERE status IN ('QUEUED','STEERING','DISPATCHING')
            """), {"now": now, "stale": now - timedelta(minutes=2)}).mappings().one()
        problems = []
        if counts["waiting"]:
            problems.append(f"{counts['waiting']} deliveries waiting over 2 minutes (targets may be busy)")
        if counts["expired"]:
            problems.append(f"{counts['expired']} active deliveries with expired leases")
        return _check("deliveries.queue", DEGRADED if problems else OK, "; ".join(problems) or None)

    def scheduler(self):
        if not self.service.config.SCHEDULER_ENABLED:
            return [_check("scheduler", OK, "disabled by configuration")]
        scheduler = getattr(self.service, "_scheduler", None)
        if scheduler is None or not scheduler._running:
            return [_check("scheduler", DOWN, "enabled but scheduler is not running")]
        # Maintenance intentionally awaits model work; a delayed full sweep is
        # degradation, not proof the task is dead. Prompt cadence is independent.
        checks = [scheduler.maintenance_health.check(
            "scheduler.maintenance", scheduler._task,
            stale_seconds=max(900, scheduler.check_interval * 4),
        )]
        if scheduler._prompt_sweep is not None:
            checks.append(scheduler.prompt_health.check(
                "scheduler.prompts", scheduler._prompt_task, stale_seconds=120,
            ))
        return checks
