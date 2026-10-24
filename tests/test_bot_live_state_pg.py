"""TASK-1400: batched live bot state on a private PostgreSQL (fixture bots only).

Every source is read for real: open/ended turn_logs rows, a queued delivery, a
pending approval, a live gate and the session pointer. Turn state must equal
the admission classifier's own answer, and the statement count must not grow
with the number of bots.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import event, text as sa_text

from llm_bawt.bot_live_state import BotIdentity, BotLiveStateReader
from llm_bawt.bot_session_state import BotSessionState
from llm_bawt.service.turn_admission import peek_target
from llm_bawt.service.turn_gates import GateScope, GateSpec, insert_gate
from tests.pg_isolation import require_test_db_config

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def rig():
    from types import SimpleNamespace

    from llm_bawt.approval_policy_store import ToolApprovalPolicyStore
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
    from llm_bawt.runtime_settings import BotProfileStore

    config = require_test_db_config()
    deliveries = InterBotDeliveryStore(config)  # also provisions the gate schema
    approvals = ToolApprovalPolicyStore(config)
    profiles = BotProfileStore(config)
    assert deliveries.engine is not None and approvals.engine is not None
    tag = uuid.uuid4().hex[:12]
    ctx = SimpleNamespace(engine=deliveries.engine, deliveries=deliveries, approvals=approvals,
                          bots=[], owner=f"job-live-state-{tag}")

    def bot():
        slug = f"test-live-{tag}-{len(ctx.bots)}"
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
                "DELETE FROM tool_approval_requests WHERE bot_id=:b",
                "DELETE FROM turn_logs WHERE bot_id=:b",
                "UPDATE bot_profiles SET last_session_id=NULL WHERE slug=:b",
                "DELETE FROM sessions WHERE bot_id=:b",
                "DELETE FROM bot_profiles WHERE slug=:b",
            ):
                conn.execute(sa_text(sql), {"b": slug})


def _turn(engine, bot, *, status="streaming", ended=False, end_reason=None):
    turn_id = f"turn-{uuid.uuid4().hex}"
    with engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id, status,
                                   user_prompt, response_text, ended_at, end_reason)
            VALUES (:id, now(), '/test', true, :bot, 'fixture-user', :status, '', '',
                    CASE WHEN :ended THEN now() END, :reason)
        """), {"id": turn_id, "bot": bot, "status": status, "ended": ended, "reason": end_reason})
    return turn_id


def _identities(bots):
    return [BotIdentity(slug, "agent", "claude-code") for slug in bots]


def test_busy_and_idle_bots_report_every_source(rig):
    busy, idle = rig.bot(), rig.bot()
    _turn(rig.engine, idle, status="ok", ended=True, end_reason="complete")
    running = _turn(rig.engine, busy)
    rig.deliveries.enqueue(
        sender_bot_id=idle, target_bot_id=busy, message="callback",
        payload={"messages": [{"role": "user", "content": "callback"}], "bot_id": busy,
                 "user": "fixture-user", "stream": True},
    )
    rig.approvals.record_request(
        request_id=f"req-{uuid.uuid4().hex}", bot_id=busy, user_id="fixture-user",
        turn_id=running, backend="claude-code", tool_name="Bash", tool_arguments={},
        subject="x", grant_key="g", policy_id=None, severity="medium", prompt="",
    )
    session_id = str(uuid.uuid4())
    with rig.engine.begin() as conn:
        gate = insert_gate(conn, GateSpec(owner_kind="job", owner_id=rig.owner, mode="hold",
                                          scope=GateScope.for_bots(busy), ttl_seconds=600,
                                          reason="fixture hold"))
        conn.execute(sa_text("""
            INSERT INTO sessions (id, bot_id, user_id, status, started_at)
            VALUES (:id, :bot, 'fixture-user', 'active', now())
        """), {"id": session_id, "bot": busy})
        BotSessionState.accept(conn, bot_id=busy, session_id=session_id)

    states = BotLiveStateReader(rig.engine).read(_identities([busy, idle]))

    hot = states[busy]
    assert hot["turn"]["kind"] == "executing" and hot["turn"]["turn_id"] == running
    assert hot["turn"]["started_at"]
    assert hot["last_turn"] is None
    assert hot["queued_deliveries"]["count"] == 1 and hot["queued_deliveries"]["oldest_at"]
    assert hot["gated"] is True and hot["gate"]["id"] == gate.id and hot["gate"]["mode"] == "hold"
    assert hot["pending_approvals"] == 1
    assert hot["session"]["id"] == session_id and hot["session"]["started_at"]
    assert "unavailable" not in hot

    cold = states[idle]
    assert cold["turn"] == {"kind": "idle", "turn_id": None, "started_at": None}
    assert cold["last_turn"]["status"] == "ok" and cold["last_turn"]["end_reason"] == "complete"
    assert cold["queued_deliveries"] == {"count": 0, "oldest_at": None}
    assert cold["gated"] is False and cold["gate"] is None
    assert cold["pending_approvals"] == 0 and cold["session"] is None
    assert "unavailable" not in cold


def test_turn_state_is_the_admission_classifier(rig):
    busy, reserved, idle = rig.bot(), rig.bot(), rig.bot()
    _turn(rig.engine, busy)
    _turn(rig.engine, reserved, status="reserved")

    states = BotLiveStateReader(rig.engine).read(_identities([busy, reserved, idle]))

    for slug in (busy, reserved, idle):
        target = peek_target(rig.engine, slug, exclude_turn_id=None)
        assert states[slug]["turn"]["kind"] == target.kind
        assert states[slug]["turn"]["turn_id"] == target.blocking_id()


def test_last_turn_is_the_newest_ended_row(rig):
    bot = rig.bot()
    _turn(rig.engine, bot, status="error", ended=True, end_reason="error")
    _turn(rig.engine, bot, status="aborted", ended=True, end_reason="user_stop")
    _turn(rig.engine, bot)  # open rows are never the last turn

    last = BotLiveStateReader(rig.engine).read(_identities([bot]))[bot]["last_turn"]

    assert last["status"] == "aborted" and last["end_reason"] == "user_stop"


def _statements(engine, bots) -> int:
    count = 0

    def tick(*_args):
        nonlocal count
        count += 1

    event.listen(engine, "before_cursor_execute", tick)
    try:
        BotLiveStateReader(engine).read(_identities(bots))
    finally:
        event.remove(engine, "before_cursor_execute", tick)
    return count


def test_statement_count_does_not_grow_with_bots(rig):
    few = [rig.bot() for _ in range(2)]
    many = few + [rig.bot() for _ in range(4)]
    for slug in many:
        _turn(rig.engine, slug)

    assert _statements(rig.engine, few) == _statements(rig.engine, many)
    assert _statements(rig.engine, many) <= len(BotLiveStateReader.SOURCES)


def test_an_invalid_session_pointer_is_unavailable_not_none(rig):
    bot = rig.bot()
    session_id = str(uuid.uuid4())
    with rig.engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO sessions (id, bot_id, user_id, status, started_at)
            VALUES (:id, :bot, 'fixture-user', 'deleted', now())
        """), {"id": session_id, "bot": bot})
        conn.execute(sa_text("UPDATE bot_profiles SET last_session_id=:s WHERE slug=:b"),
                     {"s": session_id, "b": bot})

    state = BotLiveStateReader(rig.engine).read(_identities([bot]))[bot]

    assert state["session"] is None and state["unavailable"] == ["session"]
