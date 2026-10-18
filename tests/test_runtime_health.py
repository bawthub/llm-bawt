"""TASK-1157: capability health follows actual workers and the whole outbox."""
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine, text

from llm_bawt.service.health_checks import collect
from llm_bawt.service.runtime_health import LoopHealth, RuntimeHealthChecks

NOW = datetime(2026, 10, 18, 18, tzinfo=UTC)


def task(done=False):
    return SimpleNamespace(done=lambda: done)


def host(enabled=True):
    scheduler = SimpleNamespace(_running=True, _task=task(), _prompt_task=task(),
                                _prompt_sweep=object(), check_interval=30,
                                maintenance_health=LoopHealth(), prompt_health=LoopHealth())
    return SimpleNamespace(config=SimpleNamespace(SCHEDULER_ENABLED=enabled), _scheduler=scheduler)


def test_scheduler_disabled_is_explicit_not_a_dead_worker():
    service = host(False)
    service._scheduler = None
    assert RuntimeHealthChecks(service).scheduler() == [
        {"name": "scheduler", "status": "ok", "reason": "disabled by configuration"}]


@pytest.mark.parametrize("scheduler", [None, SimpleNamespace(_running=False)])
def test_enabled_missing_scheduler_is_down(scheduler):
    service = host()
    service._scheduler = scheduler
    assert RuntimeHealthChecks(service).scheduler()[0]["status"] == "down"


def test_scheduler_tracks_independent_task_death_errors_and_recovery():
    service = host()
    probe = RuntimeHealthChecks(service)
    assert all(row["status"] == "ok" for row in probe.scheduler())
    service._scheduler._prompt_task = task(True)
    assert probe.scheduler()[1]["status"] == "down"
    service._scheduler._prompt_task = task()
    service._scheduler.prompt_health.failed(ValueError("sweep failed"))
    assert "sweep failed" in probe.scheduler()[1]["reason"]
    service._scheduler.prompt_health.succeeded()
    assert probe.scheduler()[1]["status"] == "ok"
    service._scheduler.prompt_health.last_success -= 121
    service._scheduler.maintenance_health.last_success -= 121
    rows = probe.scheduler()
    assert rows[0]["status"] == "ok"  # maintenance awaits legitimate model work
    assert rows[1]["status"] == "degraded"
    assert "busy or stalled" in rows[1]["reason"]


@pytest.mark.parametrize("prompt", [False, True])
def test_real_scheduler_loop_records_failure_then_success(monkeypatch, prompt):
    import asyncio
    from unittest.mock import AsyncMock

    from llm_bawt.service.scheduler import JobScheduler

    scheduler = JobScheduler(None, None, check_interval=0, prompt_sweep=AsyncMock())
    monitor = scheduler.prompt_health if prompt else scheduler.maintenance_health
    loop = scheduler._run_prompt_loop if prompt else scheduler._run_loop
    attribute = "_prompt_sweep" if prompt else "_check_and_run_due_jobs"
    monkeypatch.setattr("llm_bawt.service.scheduler.asyncio.sleep", AsyncMock())

    async def failed():
        scheduler._running = False
        raise ValueError("test sweep error")

    setattr(scheduler, attribute, failed)
    scheduler._running = True
    asyncio.run(loop())
    assert "test sweep error" in monitor.error

    async def succeeded():
        scheduler._running = False

    setattr(scheduler, attribute, succeeded)
    scheduler._running = True
    asyncio.run(loop())
    assert monitor.error is None


@pytest.fixture
def delivery_host():
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("""CREATE TABLE inter_bot_deliveries (
            status TEXT, created_at TIMESTAMP, updated_at TIMESTAMP,
            available_at TIMESTAMP, lease_expires_at TIMESTAMP)"""))
    service = host()
    service._inter_bot_dispatcher = SimpleNamespace(store=SimpleNamespace(engine=engine), _leader_task=task())
    yield service
    engine.dispose()


def insert(service, status, *, age=0, delay=0, lease=60):
    with service._inter_bot_dispatcher.store.engine.begin() as conn:
        conn.execute(text("INSERT INTO inter_bot_deliveries VALUES (:status,:created,:updated,:available,:lease)"),
                     {"status": status, "created": NOW-timedelta(seconds=age),
                      "updated": NOW-timedelta(seconds=age), "available": NOW+timedelta(seconds=delay),
                      "lease": NOW+timedelta(seconds=lease)})


def test_backlog_is_not_limited_to_latest_page(delivery_host):
    insert(delivery_host, "QUEUED", age=121)
    for _ in range(210):
        insert(delivery_host, "QUEUED", age=0)
    result = RuntimeHealthChecks(delivery_host).deliveries(NOW)
    assert result["status"] == "degraded"
    assert "1 deliveries waiting" in result["reason"]


def test_deliveries_ignore_future_backoff_valid_lease_and_old_failure(delivery_host):
    insert(delivery_host, "QUEUED", age=600, delay=30)
    insert(delivery_host, "DISPATCHING", age=600, lease=30)
    insert(delivery_host, "FAILED", age=3700)
    assert RuntimeHealthChecks(delivery_host).deliveries(NOW)["status"] == "ok"


def test_deliveries_surface_recent_failures_and_expired_leases(delivery_host):
    insert(delivery_host, "STEERING", age=180, lease=-1)
    insert(delivery_host, "FAILED", age=60)
    row = RuntimeHealthChecks(delivery_host).deliveries(NOW)
    assert row["status"] == "degraded"
    assert "expired leases" in row["reason"] and "failed in the last hour" in row["reason"]
    delivery_host._inter_bot_dispatcher._leader_task = task(True)
    assert RuntimeHealthChecks(delivery_host).deliveries(NOW)["status"] == "down"


def test_broken_capabilities_do_not_erase_remaining_report(monkeypatch):
    monkeypatch.setattr("llm_bawt.service.health_checks.check_database", lambda _: {"name": "database", "status": "ok"})
    monkeypatch.setattr("llm_bawt.service.health_checks.check_redis", lambda _: {"name": "redis", "status": "ok"})
    monkeypatch.setattr(RuntimeHealthChecks, "deliveries", Mock(side_effect=RuntimeError("DB unavailable")))
    ops = SimpleNamespace(executors=Mock(side_effect=RuntimeError("executor unavailable")))
    store = SimpleNamespace(engine=None)
    result = collect(host().config, store, ops, service=host())
    checks = {row["name"]: row for row in result["checks"]}
    assert result["status"] == "down"
    assert checks["deliveries.queue"]["status"] == "down"
    assert checks["ops.executors"]["status"] == "down"
    assert checks["scheduler.prompts"]["status"] == "ok"
