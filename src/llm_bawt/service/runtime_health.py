"""Read-only capability health for the app-owned delivery and scheduling loops."""
from __future__ import annotations

from datetime import UTC, datetime
import time

from .health_checks import DEGRADED, DOWN, OK, _check


class LoopHealth:
    """Records actual completed sweeps, not a timer claiming the worker is alive."""

    def __init__(self):
        self.last_success = time.monotonic()
        self.error: str | None = None

    def succeeded(self):
        self.last_success = time.monotonic()
        self.error = None

    def failed(self, exc: Exception):
        self.error = f"{type(exc).__name__}: {exc}"

    def check(self, name, task, *, stale_seconds):
        if task is None or task.done():
            return _check(name, DOWN, "worker task is not running")
        if self.error:
            return _check(name, DEGRADED, f"last sweep failed: {self.error}")
        age = time.monotonic() - self.last_success
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
        # Health describes live work, not historical terminal request failures.
        # Failed deliveries remain available in delivery history for diagnosis.
        # The whole queue is read, not the newest page. Future backoffs are
        # intentional waits; active work is only suspect after its lease expires.
        from .delivery_queue_health import DeliveryQueueHealth

        return DeliveryQueueHealth(dispatcher.store.engine).check(now or datetime.now(UTC))

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
