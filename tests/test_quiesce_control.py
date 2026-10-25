"""Unit tests for the quiesce control steer's app side (TASK-1233 B3, TASK-1431 S3).

The producer's database behaviour is in ``test_quiesce_control_producer_pg``
and the control lane's in ``test_inter_bot_control_lane_pg``; here are the
steer route's control path, the producer's pure helpers and its wiring into
the dispatcher leader's sweep (``LeaderSidePumps``).
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from llm_bawt.inter_bot_control_lane import DeliveryControlGrant
from llm_bawt.service import quiesce_control as qc
from llm_bawt.service import system_gate_admission as gates
from llm_bawt.service.leader_side_pumps import CONTROL_RECHECK_CAP_SECONDS, LeaderSidePumps
from llm_bawt.service.system_gate_admission import ControlGrantRefused, ControlTarget

GRANT = DeliveryControlGrant(bot_id="bot", turn_id="turn", delivery_id="d1", claim_token="tok")
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


# --- the producer's pure helpers ------------------------------------------


def test_the_run_key_is_one_per_pinned_run():
    assert qc.control_request_id("turn", "req") == "steer_quiesce_turn_req"


def test_successor_keys_round_trip_and_reject_other_keys():
    run = qc.control_request_id("turn", "req")
    assert [qc.successor_key(run, k) for k in (0, 1, 12)] == [run, f"{run}:r1", f"{run}:r12"]
    for k in (0, 1, 12):
        assert qc.successor_index(run, qc.successor_key(run, k)) == k
    for other in (None, "", f"{run}:r", f"{run}:rx", f"{run}x", f"{run}:r1:r2", "steer_quiesce_turn_re"):
        assert qc.successor_index(run, other) is None


def test_no_producer_is_built_off_postgresql():
    assert qc.build_producer(SimpleNamespace()) is None
    assert qc.build_producer(SimpleNamespace(engine=None)) is None
    assert qc.build_producer(SimpleNamespace(engine=SimpleNamespace(dialect=SimpleNamespace(name="sqlite")))) is None
    store = SimpleNamespace(engine=SimpleNamespace(dialect=SimpleNamespace(name="postgresql")))
    producer = qc.build_producer(store)
    assert isinstance(producer, qc.QuiesceControlProducer) and producer.engine is store.engine


def test_the_old_pump_is_gone():
    for name in ("QuiesceControlPump", "QuiesceControlRunner", "service_steer", "ControlClaim", "build_pump"):
        assert not hasattr(qc, name)
    assert not hasattr(gates, "ControlGrant")


# --- the producer on the leader's sweep -----------------------------------


class Dispatcher:
    def __init__(self):
        self.wakes = 0

    def wake(self):
        self.wakes += 1


class Producer:
    def __init__(self, *passes):
        self.passes = list(passes)
        self.calls = 0

    def run_pass(self):
        self.calls += 1
        return self.passes.pop(0)


def _pumps(producer):
    dispatcher = Dispatcher()
    pumps = LeaderSidePumps(SimpleNamespace(_inter_bot_dispatcher=dispatcher), SimpleNamespace(engine=None),
                            quiesce_control=producer)
    return pumps, dispatcher


async def _noop_emit(_record):
    return None


def test_a_new_control_row_wakes_the_dispatcher_at_once():
    async def scenario():
        pumps, dispatcher = _pumps(Producer(qc.ProducerPass(linked=1, enqueued=1)))
        await pumps.run_pass(_noop_emit)
        assert dispatcher.wakes == 1 and pumps._control_recheck is None

    asyncio.run(scenario())


def test_a_link_to_an_existing_row_needs_no_wake():
    async def scenario():
        pumps, dispatcher = _pumps(Producer(qc.ProducerPass(linked=1, enqueued=0)))
        await pumps.run_pass(_noop_emit)
        assert dispatcher.wakes == 0 and pumps._control_recheck is None

    asyncio.run(scenario())


def test_a_recheck_is_one_timer_replaced_by_the_next_pass_and_floored():
    async def scenario():
        pumps, dispatcher = _pumps(Producer(qc.ProducerPass(retry_in=30.0), qc.ProducerPass(retry_in=0.0)))
        loop = asyncio.get_running_loop()
        await pumps.run_pass(_noop_emit)
        first = pumps._control_recheck
        assert first.when() - loop.time() == pytest.approx(CONTROL_RECHECK_CAP_SECONDS, abs=0.5)
        await pumps.run_pass(_noop_emit)
        second = pumps._control_recheck
        assert first.cancelled() and second is not first and not second.cancelled()
        assert dispatcher.wakes == 0
        await asyncio.sleep(0.6)  # the 0.5 s floor
        assert dispatcher.wakes == 1

    asyncio.run(scenario())


def test_stop_cancels_the_recheck():
    async def scenario():
        pumps, dispatcher = _pumps(Producer(qc.ProducerPass(retry_in=0.0)))
        await pumps.run_pass(_noop_emit)
        handle = pumps._control_recheck
        await pumps.stop()
        assert handle.cancelled() and pumps._control_recheck is None
        await asyncio.sleep(0.6)
        assert dispatcher.wakes == 0

    asyncio.run(scenario())


def test_a_failing_producer_pass_is_logged_and_the_later_pumps_still_run(caplog):
    class Broken:
        def run_pass(self):
            raise RuntimeError("db down")

    class Notices:
        calls = 0

        def run_pass(self):
            Notices.calls += 1
            return SimpleNamespace(notified=[])

    async def scenario():
        pumps, dispatcher = _pumps(Broken())
        pumps.cancel_notices = Notices()
        await pumps.run_pass(_noop_emit)
        assert Notices.calls == 1 and dispatcher.wakes == 0 and pumps._control_recheck is None

    asyncio.run(scenario())
    assert "Quiesce control producer pass failed" in caplog.text
