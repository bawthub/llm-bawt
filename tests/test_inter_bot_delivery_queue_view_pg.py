"""TASK-1308 C0b: the per-bot delivery queue read model on a private PostgreSQL."""

from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy import text as sa_text

from tests.pg_isolation import require_test_db_config

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def rig():
    from types import SimpleNamespace

    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
    from llm_bawt.runtime_settings import BotProfileStore

    config = require_test_db_config()
    store = InterBotDeliveryStore(config)
    profiles = BotProfileStore(config)
    tag = uuid.uuid4().hex[:12]
    sender, agent, chat = (f"test-qv-{tag}-{n}" for n in ("sender", "agent", "chat"))
    for slug in (sender, agent):
        profiles.upsert({"slug": slug, "name": slug, "bot_type": "agent", "agent_backend": "claude-code"})
    profiles.upsert({"slug": chat, "name": chat, "bot_type": "chat"})
    yield SimpleNamespace(store=store, engine=store.engine, sender=sender, agent=agent, chat=chat)
    with store.engine.begin() as conn:
        for slug in (sender, agent, chat):
            conn.execute(sa_text("DELETE FROM inter_bot_deliveries WHERE target_bot_id=:b"), {"b": slug})
            conn.execute(sa_text("DELETE FROM bot_profiles WHERE slug=:b"), {"b": slug})


def _enqueue(rig, target, message="fixture", **update):
    record, _ = rig.store.enqueue(
        sender_bot_id=rig.sender, target_bot_id=target, message=message,
        payload={"messages": [{"role": "user", "content": message}], "bot_id": target,
                 "user": "fixture-user", "stream": True},
    )
    if update:
        sets = ", ".join(f"{k} = {v}" for k, v in update.items())
        with rig.engine.begin() as conn:
            conn.execute(sa_text(f"UPDATE inter_bot_deliveries SET {sets} WHERE id = :id"), {"id": record.id})
    return record.id


def _queue(rig, bot):
    from llm_bawt.inter_bot_delivery_queue_view import delivery_queues

    snapshot = delivery_queues(rig.engine)
    return next((q for q in snapshot["queues"] if q["bot_id"] == bot), None)


def test_head_is_the_lowest_ordinal_queued_row_and_prompt_schedules_count(rig):
    first = _enqueue(rig, rig.agent, "first " + "x" * 300,
                     metadata_json="'" + json.dumps({"prompt_schedule": {"id": "s1"}}) + "'::jsonb")
    _enqueue(rig, rig.agent, "second")
    _enqueue(rig, rig.agent, "done", status="'DELIVERED'")

    q = _queue(rig, rig.agent)

    assert q["waiting"]["count"] == 2
    head = q["waiting"]["head"]
    assert head["id"] == first
    assert head["prompt_schedule"] is True
    assert len(head["preview"]) == 120
    assert q["waiting"]["oldest_age_seconds"] >= 0


def test_in_flight_rows_and_open_turn_fact(rig):
    steering = _enqueue(rig, rig.agent, status="'STEERING'", delivery_mode="'steer'")

    q = _queue(rig, rig.agent)

    assert [r["id"] for r in q["in_flight"]] == [steering]
    assert q["in_flight"][0]["steer_uncertain"] is False
    assert q["waiting"] == {"count": 0, "oldest_age_seconds": None, "head": None}
    assert q["open_turn"] is False


def test_failures_inside_the_last_day_only(rig):
    _enqueue(rig, rig.agent, status="'FAILED'", last_error="'" + "e" * 400 + "'")
    _enqueue(rig, rig.agent, status="'CANCELLED'")
    _enqueue(rig, rig.agent, status="'FAILED'", updated_at="now() - interval '25 hours'")

    failures = _queue(rig, rig.agent)["recent_failures"]

    assert failures["count"] == 2
    assert len(failures["latest"]) == 2
    assert max(len(f["last_error"] or "") for f in failures["latest"]) == 200


def test_chat_bots_are_excluded(rig):
    _enqueue(rig, rig.chat)

    assert _queue(rig, rig.chat) is None
    assert _queue(rig, rig.sender) is not None


def test_queues_route_is_registered_before_the_delivery_id_route():
    from llm_bawt.service.routes.inter_bot_deliveries import router

    paths = [r.path for r in router.routes if "GET" in getattr(r, "methods", set())]
    assert paths.index("/v1/inter-bot-deliveries/queues") < paths.index(
        "/v1/inter-bot-deliveries/{delivery_id}")
