"""TASK-1754: which task each turn works on, on a private PostgreSQL.

Real turn_logs, tool_call_records and inter_bot_deliveries rows for fixture
bots only. Covers the source order (association > starting delivery >
mutation), ignored read-only/failed/steered records, ref normalization,
Claude and Codex tool names, the time window, and bot live state.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text as sa_text

from llm_bawt.bot_live_state import BotIdentity, BotLiveStateReader
from llm_bawt.turn_task import normalize_task_ref, resolve_turn_tasks
from tests.pg_isolation import require_test_db_config

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def rig():
    from types import SimpleNamespace

    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
    from llm_bawt.runtime_settings import BotProfileStore

    config = require_test_db_config()
    deliveries = InterBotDeliveryStore(config)
    profiles = BotProfileStore(config)
    assert deliveries.engine is not None
    tag = uuid.uuid4().hex[:12]
    ctx = SimpleNamespace(engine=deliveries.engine, deliveries=deliveries, bots=[])

    def bot():
        slug = f"test-tt-{tag}-{len(ctx.bots)}"
        profiles.upsert({"slug": slug, "name": slug, "bot_type": "agent",
                         "agent_backend": "claude-code"})
        ctx.bots.append(slug)
        return slug

    ctx.bot = bot
    yield ctx
    with ctx.engine.begin() as conn:
        for slug in ctx.bots:
            for sql in (
                "DELETE FROM tool_call_records WHERE bot_id=:b",
                "DELETE FROM inter_bot_deliveries WHERE target_bot_id=:b OR sender_bot_id=:b",
                "DELETE FROM turn_logs WHERE bot_id=:b",
                "DELETE FROM bot_profiles WHERE slug=:b",
            ):
                conn.execute(sa_text(sql), {"b": slug})


def _turn(engine, bot, *, turn_id=None, ended=False, age=timedelta(0)):
    turn_id = turn_id or f"turn-{uuid.uuid4().hex}"
    with engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id, status,
                                   user_prompt, response_text, ended_at)
            VALUES (:id, now() - :age, '/test', true, :bot, 'fixture-user',
                    CASE WHEN :ended THEN 'ok' ELSE 'streaming' END, '', '',
                    CASE WHEN :ended THEN now() - :age END)
        """), {"id": turn_id, "bot": bot, "ended": ended, "age": age})
    return turn_id


_SECONDS = iter(range(1, 10_000))


def _call(engine, bot, turn_id, tool, args, *, error=False):
    # Strictly increasing timestamps: "newest" must not depend on insert speed.
    at = datetime(2026, 11, 4, tzinfo=timezone.utc) + timedelta(seconds=next(_SECONDS))
    with engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO tool_call_records (created_at, tool_name, iteration, bot_id, turn_id,
                                           arguments_json, is_error)
            VALUES (:at, :tool, 0, :bot, :turn, :args, :error)
        """), {"at": at, "tool": tool, "bot": bot, "turn": turn_id,
               "args": json.dumps(args), "error": error})


def _deliver(rig, sender, target, task_id):
    record, _ = rig.deliveries.enqueue(
        sender_bot_id=sender, target_bot_id=target, message="work", task_id=task_id,
        payload={"messages": [{"role": "user", "content": "work"}], "bot_id": target,
                 "user": "fixture-user", "stream": True},
    )
    return record


def _resolve(rig, turn_id):
    with rig.engine.connect() as conn:
        found = resolve_turn_tasks(conn, turn_ids=[turn_id])
    return (found[0].task_ref, found[0].source) if found else None


def test_association_beats_delivery_beats_mutation(rig):
    sender, bot = rig.bot(), rig.bot()
    delivery = _deliver(rig, sender, bot, "TASK-11")
    turn = _turn(rig.engine, bot, turn_id=f"turn-{delivery.id}")
    _call(rig.engine, bot, turn, "mcp__bawthub__tasks_update", {"task_id": "TASK-12"})
    assert _resolve(rig, turn) == ("TASK-11", "delivery")

    _call(rig.engine, bot, turn, "mcp__bawthub__tasks_associate_current", {"task_id": "TASK-13"})
    assert _resolve(rig, turn) == ("TASK-13", "association")


def test_newest_mutation_wins_and_reads_never_count(rig):
    bot = rig.bot()
    turn = _turn(rig.engine, bot)
    _call(rig.engine, bot, turn, "mcp__bawthub__tasks_get_context", {"task_id": "TASK-20"})
    _call(rig.engine, bot, turn, "tasks_get", {"task_id": "TASK-21"})
    assert _resolve(rig, turn) is None

    _call(rig.engine, bot, turn, "mcp__bawthub__steps_update", {"task_id": "TASK-22"})
    _call(rig.engine, bot, turn, "tasks_reviews_add", {"task_id": "task-23"})  # Codex name
    _call(rig.engine, bot, turn, "mcp__bawthub__tasks_reviews_list", {"task_id": "TASK-24"})
    _call(rig.engine, bot, turn, "mcp__bawthub__tasks_update", {"task_id": "TASK-25"}, error=True)
    assert _resolve(rig, turn) == ("TASK-23", "mutation")


def test_link_flag_on_tasks_update_is_an_association(rig):
    bot = rig.bot()
    turn = _turn(rig.engine, bot)
    _call(rig.engine, bot, turn, "mcp__bawthub__tasks_update",
          {"task_id": "TASK-30", "associate_current_turn": True})
    _call(rig.engine, bot, turn, "mcp__bawthub__tasks_update", {"task_id": "TASK-31"})
    assert _resolve(rig, turn) == ("TASK-30", "association")


def test_steered_deliveries_do_not_assign_the_turn(rig):
    sender, bot = rig.bot(), rig.bot()
    start = _deliver(rig, sender, bot, "TASK-40")
    turn = _turn(rig.engine, bot, turn_id=f"turn-{start.id}")
    steered = _deliver(rig, sender, bot, "TASK-41")
    with rig.engine.begin() as conn:  # a later message steered into the same turn
        conn.execute(sa_text("UPDATE inter_bot_deliveries SET target_turn_id=:t WHERE id=:d"),
                     {"t": turn, "d": steered.id})
    assert _resolve(rig, turn) == ("TASK-40", "delivery")


def test_a_delivery_without_a_task_falls_through_to_mutation(rig):
    sender, bot = rig.bot(), rig.bot()
    start = _deliver(rig, sender, bot, None)
    turn = _turn(rig.engine, bot, turn_id=f"turn-{start.id}")
    _call(rig.engine, bot, turn, "mcp__bawthub__steps_add", {"task_id": "cmabc123cuid"})
    assert _resolve(rig, turn) == ("cmabc123cuid", "mutation")


def test_window_and_bot_filters(rig):
    one, two = rig.bot(), rig.bot()
    old = _turn(rig.engine, one, ended=True, age=timedelta(hours=30))
    fresh = _turn(rig.engine, one, ended=True)
    running_since_long = _turn(rig.engine, two, age=timedelta(hours=30))
    for bot, turn in ((one, old), (one, fresh), (two, running_since_long)):
        _call(rig.engine, bot, turn, "mcp__bawthub__tasks_update", {"task_id": "TASK-50"})
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    with rig.engine.connect() as conn:
        found = {t.turn_id for t in resolve_turn_tasks(conn, bots=[one, two], since=since)}
        only_two = resolve_turn_tasks(conn, bots=[two], since=since)
    assert found == {fresh, running_since_long}  # an open turn counts however old
    assert [t.turn_id for t in only_two] == [running_since_long]


def test_live_state_carries_the_running_and_last_turn_task(rig):
    busy, idle = rig.bot(), rig.bot()
    done = _turn(rig.engine, idle, ended=True)
    _call(rig.engine, idle, done, "mcp__bawthub__tasks_associate_current", {"task_id": "TASK-60"})
    running = _turn(rig.engine, busy)
    _call(rig.engine, busy, running, "steps_set", {"task_id": "TASK-61"})

    states = BotLiveStateReader(rig.engine).read(
        BotIdentity(slug, "agent", "claude-code") for slug in (busy, idle))

    assert states[busy]["turn"]["task"] == {"task_ref": "TASK-61", "source": "mutation"}
    assert states[idle]["turn"]["task"] is None
    assert states[idle]["last_turn"]["turn_id"] == done
    assert states[idle]["last_turn"]["task"] == {"task_ref": "TASK-60", "source": "association"}


@pytest.mark.parametrize(("raw", "ref"), [
    ("TASK-7", "TASK-7"), ("task-7", "TASK-7"), (" Task-007 ", "TASK-7"),
    ("cmvu3go2d00m80mt5n6z3f73t", "cmvu3go2d00m80mt5n6z3f73t"), ("", None), (None, None),
])
def test_normalize_task_ref(raw, ref):
    assert normalize_task_ref(raw) == ref
