"""TASK-1458: deliveries.queue health on a private PostgreSQL (fixture bots only).

The real claim authorities decide each wait: open turn_logs rows through the
admission classifier, a live hold gate through ``gate_blocks``, and real
timestamptz columns. Only rows for this test's bots are asserted, because the
check reads the whole queue.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import text as sa_text

from llm_bawt.service.delivery_queue_health import DeliveryQueueHealth
from llm_bawt.service.turn_gates import GateScope, GateSpec, insert_gate
from tests.pg_isolation import require_test_db_config

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def rig():
    from types import SimpleNamespace

    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
    from llm_bawt.runtime_settings import BotProfileStore

    config = require_test_db_config()
    deliveries = InterBotDeliveryStore(config)  # also provisions the gate schema
    profiles = BotProfileStore(config)
    assert deliveries.engine is not None
    tag = uuid.uuid4().hex[:12]
    ctx = SimpleNamespace(engine=deliveries.engine, deliveries=deliveries, bots=[],
                          owner=f"job-queue-health-{tag}")

    def bot():
        slug = f"test-qh-{tag}-{len(ctx.bots)}"
        profiles.upsert({"slug": slug, "name": slug, "bot_type": "agent",
                         "agent_backend": "claude-code"})
        ctx.bots.append(slug)
        return slug

    ctx.bot = bot
    yield ctx
    with ctx.engine.begin() as conn:
        conn.execute(sa_text("DELETE FROM system_gates WHERE owner_id=:o"), {"o": ctx.owner})
        for slug in ctx.bots:
            for sql in (
                "DELETE FROM inter_bot_deliveries WHERE target_bot_id=:b",
                "DELETE FROM turn_logs WHERE bot_id=:b",
                "DELETE FROM bot_profiles WHERE slug=:b",
            ):
                conn.execute(sa_text(sql), {"b": slug})


def _turn(engine, bot, *, age="1 minute"):
    with engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id, status,
                                   user_prompt, response_text)
            VALUES (:id, now() - CAST(:age AS interval), '/test', true, :bot, 'fixture-user',
                    'streaming', '', '')
        """), {"id": f"turn-{uuid.uuid4().hex}", "bot": bot, "age": age})


def _queued(rig, sender, target, *, task_id=None, age="5 minutes"):
    record, _ = rig.deliveries.enqueue(
        sender_bot_id=sender, target_bot_id=target, message="handoff", task_id=task_id,
        payload={"messages": [{"role": "user", "content": "handoff"}], "bot_id": target,
                 "user": "fixture-user", "stream": True},
    )
    with rig.engine.begin() as conn:
        conn.execute(sa_text("""
            UPDATE inter_bot_deliveries
            SET created_at = now() - CAST(:age AS interval), available_at = now() - CAST(:age AS interval)
            WHERE id = :id
        """), {"id": record.id, "age": age})
    return record.id


def _items(rig):
    row = DeliveryQueueHealth(rig.engine).check(datetime.now(UTC))
    return {item["target"]: item for item in row.get("items", ()) if item["target"] in rig.bots}


def test_each_wait_is_classified_by_the_claim_authorities(rig):
    sender, busy, held, idle, stuck_turn, leased = (rig.bot() for _ in range(6))
    _turn(rig.engine, busy)
    _turn(rig.engine, stuck_turn, age="3 hours")
    with rig.engine.begin() as conn:
        insert_gate(conn, GateSpec(owner_kind="job", owner_id=rig.owner, mode="hold",
                                   scope=GateScope.for_bots(held), ttl_seconds=600,
                                   reason="fixture hold"))
    for target in (busy, held, idle, stuck_turn):
        _queued(rig, sender, target, task_id="TASK-1")
    lease = _queued(rig, sender, leased)
    with rig.engine.begin() as conn:
        conn.execute(sa_text("""
            UPDATE inter_bot_deliveries SET status='DISPATCHING',
                lease_expires_at = now() - interval '90 seconds' WHERE id=:id
        """), {"id": lease})

    items = _items(rig)

    assert {bot: items[bot]["wait"] for bot in items} == {
        busy: "busy", held: "held", idle: "idle", stuck_turn: "long_turn", leased: "lease_expired",
    }
    assert items[busy]["detail"] == "target busy: executing"
    assert items[held]["detail"] == f"held by hold gate (job {rig.owner})"
    assert items[stuck_turn]["detail"].startswith("behind a turn open 3h")
    assert items[busy]["sender"] == sender and items[busy]["task_id"] == "TASK-1"
    assert 290 <= items[idle]["age_seconds"] <= 400
    assert [items[bot]["degraded"] for bot in (busy, held, idle, stuck_turn, leased)] == [
        False, False, True, True, True]


def test_a_fresh_wait_is_not_reported(rig):
    sender, target = rig.bot(), rig.bot()
    _queued(rig, sender, target, age="30 seconds")
    assert _items(rig) == {}
