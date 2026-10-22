"""Unit tests for the TASK-1233 B3 quiesce control steer's app side.

The pump's database behaviour is in ``test_quiesce_control_pg``; here are the
steer route's control path, the production RPC adapter and the runner.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from llm_bawt.service import quiesce_control as qc
from llm_bawt.service import system_gate_admission as gates
from llm_bawt.service.system_gate_admission import ControlGrant, ControlGrantRefused, ControlTarget

UNTIL = datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)
GRANT = ControlGrant(gate_id=8, bot_id="bot", turn_id="turn", claim_token="tok")
PINNED = ControlTarget(request_id="req-pinned", backend="claude-code", session_key="sess-pinned")


def _service():
    turn = SimpleNamespace(id="turn", bot_id="bot", user_id="user", user_prompt="work", ended_at=None,
                           status="streaming", agent_session_key="sess-current",
                           agent_request_id="req-current", trigger_message_id=None)
    engine = object()
    store = SimpleNamespace(get_turn=lambda _id: turn, engine=engine)
    return SimpleNamespace(_turn_log_store=store, config=None, get_memory_client=lambda *_a: None), engine


def _request(**overrides):
    from llm_bawt.service.routes.chat import ChatSteerRequest

    return ChatSteerRequest(**{"turn_id": "turn", "bot_id": "bot", "user_id": "user",
                               "message_id": str(uuid.uuid4()), "message": "please stop", **overrides})


class Subscriber:
    def __init__(self, result=None):
        self.result = result or {"ok": True}
        self.sent: list[dict] = []

    async def send_steer(self, **kwargs):
        self.sent.append(kwargs)
        return self.result


@pytest.fixture
def bridge(monkeypatch):
    subscriber = Subscriber()
    monkeypatch.setattr("llm_bawt.agent_backends.agent_bridge.get_agent_subscriber", lambda: subscriber)
    return subscriber


# --- steer_active_turn's control path -------------------------------------


def test_a_control_steer_addresses_only_the_pinned_run(monkeypatch, bridge):
    from llm_bawt.service.routes.chat import steer_active_turn

    asked = []

    def check(engine, bot_id, *, turn_id, grant=None):
        asked.append((engine, bot_id, turn_id, grant))
        return PINNED

    monkeypatch.setattr(gates, "check_steer", check)
    monkeypatch.setattr("llm_bawt.bots.BotManager", lambda *_a: pytest.fail("no current-backend lookup"))
    service, engine = _service()
    response = asyncio.run(steer_active_turn(service, _request(), steer_request_id="steer-q", control=GRANT))

    assert response.ok and asked == [(engine, "bot", "turn", GRANT)]
    [sent] = bridge.sent
    assert (sent["session_key"], sent["target_request_id"], sent["backend"]) == (
        "sess-pinned", "req-pinned", "claude-code")
    assert (sent["request_id"], sent["origin"], sent["priority"]) == ("steer-q", "system", "next")


def test_a_control_steer_to_a_pinned_run_on_another_backend_is_refused(monkeypatch, bridge):
    from llm_bawt.service.routes.chat import steer_active_turn

    monkeypatch.setattr(gates, "check_steer", lambda *_a, **_k: ControlTarget("req", "codex", "sess"))
    service, _ = _service()
    with pytest.raises(HTTPException) as raised:
        asyncio.run(steer_active_turn(service, _request(), steer_request_id="steer-q", control=GRANT))
    assert raised.value.status_code == 409 and bridge.sent == []


def test_a_control_steer_needs_its_dedupe_id(monkeypatch, bridge):
    from llm_bawt.service.routes.chat import steer_active_turn

    monkeypatch.setattr(gates, "check_steer", lambda *_a, **_k: pytest.fail("refused before the gate"))
    service, _ = _service()
    with pytest.raises(HTTPException) as raised:
        asyncio.run(steer_active_turn(service, _request(), control=GRANT))
    assert raised.value.status_code == 409 and bridge.sent == []


def test_a_refused_grant_never_reaches_the_bridge(monkeypatch, bridge):
    from llm_bawt.service.routes.chat import steer_active_turn

    def check(*_a, grant=None, **_k):
        raise ControlGrantRefused(grant)

    monkeypatch.setattr(gates, "check_steer", check)
    service, _ = _service()
    with pytest.raises(ControlGrantRefused):
        asyncio.run(steer_active_turn(service, _request(), steer_request_id="steer-q", control=GRANT))
    assert bridge.sent == []


def test_the_public_route_carries_no_grant(monkeypatch, bridge):
    from llm_bawt.service.routes import chat

    assert "control" not in chat.ChatSteerRequest.model_fields
    asked = []

    def check(engine, bot_id, *, turn_id):  # the public path passes no grant keyword
        asked.append(turn_id)

    monkeypatch.setattr(gates, "check_steer", check)
    monkeypatch.setattr("llm_bawt.bots.BotManager",
                        lambda *_a: SimpleNamespace(get_bot=lambda _b: SimpleNamespace(agent_backend="claude-code")))
    service, _ = _service()
    monkeypatch.setattr(chat, "get_service", lambda: service)
    asyncio.run(chat.chat_steer(_request()))
    assert asked == ["turn"]
    assert (bridge.sent[0]["session_key"], bridge.sent[0]["origin"]) == ("sess-current", "user")


def test_check_steer_refuses_a_grant_for_another_bot_or_turn_or_off_postgresql():
    with pytest.raises(ControlGrantRefused):
        gates.check_steer(None, "other", turn_id="turn", grant=GRANT)
    with pytest.raises(ControlGrantRefused):
        gates.check_steer(None, "bot", turn_id="other", grant=GRANT)
    with pytest.raises(ControlGrantRefused):
        gates.check_steer(None, "bot", turn_id="turn", grant=GRANT)


@pytest.mark.parametrize("member,expected", [
    ({"agent_request_id": None}, None),
    ({"agent_request_id": "r1", "backend": "claude-code", "agent_session_key": "s1",
      "turn_request_id": "r2", "turn_backend": "codex", "turn_session_key": "s2"},
     ControlTarget("r1", "claude-code", "s1")),
    ({"agent_request_id": "r1", "backend": None, "agent_session_key": None,
      "turn_request_id": "r1", "turn_backend": "claude-code", "turn_session_key": "s1"},
     ControlTarget("r1", "claude-code", "s1")),
    ({"agent_request_id": "r1", "backend": None, "agent_session_key": None,
      "turn_request_id": "r2", "turn_backend": "claude-code", "turn_session_key": "s2"},
     ControlTarget("r1", None, None)),
])
def test_the_control_target_is_the_pinned_request_binding(member, expected):
    assert gates.control_target(member) == expected


# --- the production RPC adapter ------------------------------------------


def _claim():
    return qc.ControlClaim(member_id=1, grant=GRANT, request_id="steer_quiesce_turn_req", user_id="user",
                           attempts=1)


def _adapter(monkeypatch, result):
    from llm_bawt.ops import quiesce_prompts
    from llm_bawt.service.routes import chat

    calls = []
    monkeypatch.setattr(quiesce_prompts, "resolve", lambda key, bot: f"{key}@{bot}")

    async def steer(service, request, **kwargs):
        calls.append((service, request, kwargs))
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(chat, "steer_active_turn", steer)
    return calls


def test_the_adapter_steers_from_system_with_the_registry_text_and_a_stable_message_id(monkeypatch):
    from llm_bawt.bot_identities import SYSTEM_BOT_ID
    from llm_bawt.service.routes.chat import ChatSteerResponse

    calls = _adapter(monkeypatch, ChatSteerResponse(ok=True, detail="steered", turn_id="turn", message_id="m"))
    service = object()
    outcome = asyncio.run(qc.service_steer(service)(_claim()))
    assert outcome == qc.SteerOutcome(qc.STEERED, "steered")
    asyncio.run(qc.service_steer(service)(_claim()))
    [(svc, request, kwargs), (_, again, _)] = calls
    assert svc is service and request.message == "ops.quiesce_request@bot"
    assert (request.turn_id, request.bot_id, request.user_id, request.priority) == ("turn", "bot", "user", "next")
    assert request.message_id == again.message_id == str(uuid.uuid5(uuid.NAMESPACE_URL, "steer_quiesce_turn_req"))
    assert kwargs["steer_request_id"] == "steer_quiesce_turn_req" and kwargs["control"] is GRANT
    assert (kwargs["author"].entity_type, kwargs["author"].entity_id) == ("bot", SYSTEM_BOT_ID)


@pytest.mark.parametrize("error,kind", [
    (HTTPException(404, "Turn not found"), qc.GONE),
    (HTTPException(409, "no_active_run"), qc.GONE),
    (HTTPException(409, "active_run_mismatch"), qc.GONE),
    (HTTPException(409, "Active bridge run is not ready"), qc.RETRY),
    (HTTPException(502, "steer failed"), qc.RETRY),
    (HTTPException(504, "timed out"), qc.RETRY),
    (ControlGrantRefused(GRANT), qc.RETRY),
    (gates.SteerGateRefused(gates.GateBlock(id=1, mode="deny", owner_kind="operator", owner_id="o",
                                            reason="r", expires_at=UNTIL)), qc.RETRY),
])
def test_the_adapter_settles_a_gone_run_and_retries_everything_else(monkeypatch, error, kind):
    _adapter(monkeypatch, error)
    assert asyncio.run(qc.service_steer(object())(_claim())).kind == kind


def test_the_backoff_doubles_to_its_cap():
    pump = qc.QuiesceControlPump.__new__(qc.QuiesceControlPump)
    assert [pump.backoff(n) for n in range(1, 7)] == [2, 4, 8, 16, 30, 30]


def test_no_pump_is_built_off_postgresql():
    assert qc.build_pump(object(), None) is None
    assert qc.build_pump(object(), SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))) is None


# --- the runner ----------------------------------------------------------


class FakePump:
    def __init__(self, *open_work, fail=False):
        self.open_work = list(open_work)
        self.passes = 0
        self.fail = fail
        self.release = asyncio.Event()

    async def run_pass(self):
        self.passes += 1
        if self.fail:
            raise RuntimeError("db down")
        await self.release.wait()
        return 0

    def has_open_work(self):
        return self.open_work.pop(0) if self.open_work else False


def test_the_runner_is_single_flight_and_gated_on_elapsed_time():
    async def scenario():
        now = [100.0]
        pump = FakePump()
        built = []
        runner = qc.QuiesceControlRunner(lambda: built.append(1) or pump, clock=lambda: now[0])
        assert runner.maybe_start() is True
        now[0] += 10
        assert runner.maybe_start() is False  # one task at a time
        pump.release.set()
        await asyncio.sleep(0.01)
        assert runner._task.done() and pump.passes == 1
        now[0] += 1
        assert runner.maybe_start() is True  # 11s since the last start
        await asyncio.sleep(0.01)
        assert runner.maybe_start() is False  # done, but within the interval
        now[0] += runner.INTERVAL_SECONDS
        assert runner.maybe_start() is True
        await runner.stop()
        assert built == [1]  # one pump, built once

    asyncio.run(scenario())


def test_the_runner_drains_while_work_is_open_then_exits():
    async def scenario():
        pump = FakePump(True, True, False)
        pump.release.set()
        runner = qc.QuiesceControlRunner(lambda: pump)
        runner.INTERVAL_SECONDS = 0
        assert runner.maybe_start()
        await asyncio.wait_for(runner._task, 1)
        assert pump.passes == 3

    asyncio.run(scenario())


def test_a_failing_pass_ends_the_task_for_the_next_dispatcher_pass():
    async def scenario():
        pump = FakePump(True, fail=True)
        runner = qc.QuiesceControlRunner(lambda: pump)
        runner.maybe_start()
        await asyncio.wait_for(runner._task, 1)
        assert pump.passes == 1 and runner._task.exception() is None

    asyncio.run(scenario())


def test_no_pump_means_no_task():
    async def scenario():
        runner = qc.QuiesceControlRunner(lambda: None)
        assert runner.maybe_start() is False and runner._task is None

    asyncio.run(scenario())


def test_stop_cancels_a_running_pass():
    async def scenario():
        pump = FakePump()
        runner = qc.QuiesceControlRunner(lambda: pump)
        runner.maybe_start()
        task = runner._task
        await asyncio.sleep(0)
        await runner.stop()
        assert task.cancelled()

    asyncio.run(scenario())
