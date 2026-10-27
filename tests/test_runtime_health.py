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
            ordinal INTEGER, id TEXT, status TEXT, target_bot_id TEXT, sender_bot_id TEXT,
            author_entity_type TEXT, author_entity_id TEXT, task_id TEXT, turn_id TEXT,
            created_at TIMESTAMP, updated_at TIMESTAMP,
            available_at TIMESTAMP, lease_expires_at TIMESTAMP,
            lane TEXT DEFAULT 'normal', transport_accepted_at TIMESTAMP)"""))
        conn.execute(text("""CREATE TABLE turn_logs (
            id TEXT, bot_id TEXT, status TEXT, agent_session_key TEXT,
            agent_request_id TEXT, agent_backend TEXT, agent_steer_caps TEXT, user_id TEXT,
            created_at TIMESTAMP, ended_at TIMESTAMP)"""))
    service = host()
    service._inter_bot_dispatcher = SimpleNamespace(store=SimpleNamespace(engine=engine), _leader_task=task())
    yield service
    engine.dispose()


_SEQ = iter(range(1, 100_000))


def insert(service, status, *, age=0, delay=None, lease=60, target="rook", sender="snark",
           task_id=None, turn_id=None, author="bot", lane="normal"):
    """``delay`` sets available_at relative to NOW; by default it is created_at, as on enqueue."""
    n = next(_SEQ)
    with service._inter_bot_dispatcher.store.engine.begin() as conn:
        conn.execute(text("""INSERT INTO inter_bot_deliveries (ordinal, id, status, target_bot_id,
                sender_bot_id, author_entity_type, author_entity_id, task_id, turn_id, created_at,
                updated_at, available_at, lease_expires_at, lane)
            VALUES (:n,:id,:status,:target,:sender,:author,:author_id,:task,:turn,:created,:updated,
                    :available,:lease,:lane)"""),
                     {"n": n, "id": f"delivery-{n}", "status": status, "target": target, "sender": sender,
                      "author": author, "author_id": "someone" if author == "user" else sender,
                      "task": task_id, "turn": turn_id or f"turn-delivery-{n}",
                      "created": NOW-timedelta(seconds=age), "updated": NOW-timedelta(seconds=age),
                      "available": NOW + timedelta(seconds=-age if delay is None else delay),
                      "lease": NOW+timedelta(seconds=lease), "lane": lane})
    return f"delivery-{n}"


def open_turn(service, bot, *, age=60, status="streaming", turn_id=None):
    with service._inter_bot_dispatcher.store.engine.begin() as conn:
        conn.execute(text("INSERT INTO turn_logs VALUES (:id,:bot,:status,NULL,NULL,NULL,NULL,'u',:created,NULL)"),
                     {"id": turn_id or f"turn-{bot}-{age}", "bot": bot, "status": status,
                      "created": NOW-timedelta(seconds=age)})


def deliveries(service):
    return RuntimeHealthChecks(service).deliveries(NOW)


def test_backlog_is_not_limited_to_latest_page(delivery_host):
    insert(delivery_host, "QUEUED", age=121)
    for _ in range(210):
        insert(delivery_host, "QUEUED", age=0)
    result = deliveries(delivery_host)
    assert result["status"] == "degraded"
    assert result["reason"].startswith("1 delivery stuck: rook ← snark")
    assert len(result["items"]) == 1


def test_wait_behind_a_busy_target_is_ok_and_named(delivery_host):
    # TASK-1458: a when_idle/queued send waiting for a live turn is normal.
    open_turn(delivery_host, "rook", age=900)
    open_turn(delivery_host, "byte", age=400, status="reserved")
    insert(delivery_host, "QUEUED", age=840, task_id="TASK-1249")
    insert(delivery_host, "QUEUED", age=480, target="byte", task_id="TASK-1249")
    row = deliveries(delivery_host)
    assert row["status"] == "ok"
    assert row["reason"] == (
        "2 waiting as expected: "
        "rook ← snark (TASK-1249, 14m, target busy: executing), "
        "byte ← snark (TASK-1249, 8m, target busy: reserved)")
    assert [(i["target"], i["wait"], i["degraded"]) for i in row["items"]] == [
        ("rook", "busy", False), ("byte", "busy", False)]
    assert row["items"][0]["age_seconds"] == 840


def test_wait_for_an_idle_target_is_degraded_and_named(delivery_host):
    open_turn(delivery_host, "byte", age=60)  # another bot's turn explains nothing
    insert(delivery_host, "QUEUED", age=300, task_id="TASK-7", author="user")
    row = deliveries(delivery_host)
    assert row["status"] == "degraded"
    assert row["reason"] == "1 delivery stuck: rook ← someone (TASK-7, 5m, target idle, not picked up)"
    assert row["items"][0]["wait"] == "idle"


def test_the_deliverys_own_reservation_is_not_a_busy_target(delivery_host):
    # Admission classifies the target excluding the head's own turn id.
    open_turn(delivery_host, "rook", status="reserved", turn_id="turn-own")
    insert(delivery_host, "QUEUED", age=300, turn_id="turn-own")
    assert deliveries(delivery_host)["items"][0]["wait"] == "idle"


def test_wait_behind_a_turn_no_watchdog_ends_is_degraded(delivery_host):
    open_turn(delivery_host, "rook", age=3 * 3600)
    insert(delivery_host, "QUEUED", age=600)
    row = deliveries(delivery_host)
    assert row["status"] == "degraded"
    assert "behind a turn open 3h" in row["reason"]
    assert row["items"][0]["wait"] == "long_turn"


def test_wait_held_by_a_live_gate_is_ok(delivery_host, monkeypatch):
    from llm_bawt.service import delivery_queue_health

    asked = []

    def gate_blocks(conn, bot_id, actor, **_):
        asked.append((bot_id, actor.kind, actor.turn_id))
        return SimpleNamespace(mode="hold", owner_kind="ops", owner_id="quiesce-1")

    monkeypatch.setattr(delivery_queue_health, "gate_blocks", gate_blocks)
    insert(delivery_host, "QUEUED", age=300, turn_id="turn-q")
    row = deliveries(delivery_host)
    assert row["status"] == "ok"
    assert "held by hold gate (ops quiesce-1)" in row["reason"]
    assert asked == [("rook", "delivery_claim", "turn-q")]


def test_expired_lease_is_degraded_and_named(delivery_host):
    insert(delivery_host, "DISPATCHING", age=400, lease=-90, target="caid", sender="al", task_id="TASK-9")
    row = deliveries(delivery_host)
    assert row["status"] == "degraded"
    assert row["reason"] == "1 delivery stuck: caid ← al (TASK-9, 6m, lease expired 1m ago)"
    assert row["items"][0]["wait"] == "lease_expired"


def test_named_deliveries_are_capped_at_three(delivery_host):
    open_turn(delivery_host, "byte")
    for bot in ("a", "b", "c", "d", "e"):
        insert(delivery_host, "QUEUED", age=200, target=bot)
    insert(delivery_host, "QUEUED", age=200, target="byte")
    row = deliveries(delivery_host)
    assert row["status"] == "degraded"
    assert row["reason"] == (
        "5 deliveries stuck: a ← snark (3m, target idle, not picked up), "
        "b ← snark (3m, target idle, not picked up), "
        "c ← snark (3m, target idle, not picked up) +2 more; "
        "1 waiting as expected: byte ← snark (3m, target busy: executing)")
    assert len(row["items"]) == 6


def _item(row, delivery_id):
    return next(item for item in row["items"] if item["id"] == delivery_id)


@pytest.mark.parametrize("head_kw,wait,detail", [
    # Codex r1 Blocker 1: strict FIFO holds a due row behind these heads.
    ({"age": 900, "delay": 3600}, "backoff", "retry due in 1h"),
    ({"age": 900, "delay": -30}, "pending", "target idle, claim due"),
])
def test_a_due_row_behind_a_head_that_isnt_claimable_yet_is_ok(delivery_host, head_kw, wait, detail):
    head = insert(delivery_host, "QUEUED", **head_kw)
    behind = insert(delivery_host, "QUEUED", age=300, task_id="TASK-3")
    row = deliveries(delivery_host)
    assert row["status"] == "ok"
    item = _item(row, behind)
    assert (item["wait"], item["behind"]) == (wait, head)
    assert item["detail"] == f"behind {head}: {detail}"


def test_a_row_behind_an_in_flight_head_is_ok(delivery_host):
    head = insert(delivery_host, "DISPATCHING", age=200, lease=120)
    behind = insert(delivery_host, "QUEUED", age=150)
    assert (row := deliveries(delivery_host))["status"] == "ok"
    assert _item(row, behind)["detail"] == f"behind {head}: dispatching in flight"


def test_rows_behind_a_stuck_head_are_stuck_too_and_name_it(delivery_host):
    head = insert(delivery_host, "QUEUED", age=600)
    behind = insert(delivery_host, "QUEUED", age=300)
    row = deliveries(delivery_host)
    assert row["status"] == "degraded"
    assert [(i["id"], i["wait"], i["behind"]) for i in row["items"]] == [
        (head, "idle", None), (behind, "idle", head)]


def test_a_head_just_out_of_backoff_is_not_reported_yet(delivery_host):
    insert(delivery_host, "QUEUED", age=900, delay=-30)
    assert deliveries(delivery_host) == {"name": "deliveries.queue", "status": "ok", "reason": None}


def test_deliveries_ignore_future_backoff_valid_lease_and_old_failure(delivery_host):
    insert(delivery_host, "QUEUED", age=600, delay=30)
    insert(delivery_host, "DISPATCHING", age=600, lease=30)
    insert(delivery_host, "FAILED", age=3700)
    assert RuntimeHealthChecks(delivery_host).deliveries(NOW)["status"] == "ok"


def test_recent_failed_requests_do_not_make_live_queue_unhealthy(delivery_host):
    # Terminal request outcomes are history, not work requiring intervention.
    for _ in range(8):
        insert(delivery_host, "FAILED", age=60)
    row = RuntimeHealthChecks(delivery_host).deliveries(NOW)
    assert row["status"] == "ok"
    assert row["reason"] is None


def test_deliveries_surface_expired_leases_not_historical_failures(delivery_host):
    insert(delivery_host, "STEERING", age=180, lease=-1)
    insert(delivery_host, "FAILED", age=60)
    row = RuntimeHealthChecks(delivery_host).deliveries(NOW)
    assert row["status"] == "degraded"
    assert "lease expired" in row["reason"]
    assert "failed in the last hour" not in row["reason"]
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
