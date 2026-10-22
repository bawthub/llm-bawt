"""What each producer and steer edge does with a system-gate decision (TASK-1233 C3).

``gate_blocks`` is substituted here; the PostgreSQL suite
(``test_system_gate_admission_pg``) drives the same edges against real gates.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from cryptography.fernet import Fernet
from fastapi import HTTPException

from llm_bawt.request_identity import AdmissionUnavailable
from llm_bawt.service import system_gate_admission as gates
from llm_bawt.service.turn_admission import SystemGateRefused
from llm_bawt.service.turn_gates import GateBlock
from tests.test_prompt_delivery import rig  # noqa: F401 (pytest fixture registration)

UNTIL = datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)
FREEZE = GateBlock(id=7, mode="deny", owner_kind="operator", owner_id="op-1",
                   reason="db migration", expires_at=UNTIL)
QUIESCE = GateBlock(id=8, mode="hold", owner_kind="job", owner_id="job-1",
                    reason="restart app", expires_at=UNTIL)
PG = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))


@pytest.fixture
def gate(monkeypatch):
    """Set the live gate the edges see; records each actor asked about."""
    state = SimpleNamespace(block=None, actors=[])

    def gate_blocks(conn, bot_id, actor):
        state.actors.append((bot_id, actor))
        return state.block

    monkeypatch.setattr(gates, "gate_blocks", gate_blocks)
    return state


# --- enqueue -------------------------------------------------------------


@pytest.mark.parametrize("block,refused", [(None, False), (FREEZE, True), (QUIESCE, False)])
def test_a_deny_producer_is_refused_only_under_a_deny_gate(gate, block, refused):
    gate.block = block
    if refused:
        with pytest.raises(SystemGateRefused) as raised:
            gates.admit_enqueue(PG, "byte", on_gate="deny")
        assert (raised.value.code, raised.value.status_code) == ("system_frozen", 503)
    else:
        gates.admit_enqueue(PG, "byte", on_gate="deny")
    (bot, actor), = gate.actors
    assert (bot, actor.kind, actor.turn_id) == ("byte", "enqueue", None)


@pytest.mark.parametrize("block", [None, FREEZE, QUIESCE])
def test_a_hold_producer_is_always_admitted_without_asking(gate, block):
    gate.block = block
    gates.admit_enqueue(PG, "byte", on_gate="hold")
    assert gate.actors == []


def test_the_refusal_rides_the_existing_admission_error_mapping(gate):
    gate.block = FREEZE
    with pytest.raises(AdmissionUnavailable) as raised:
        gates.admit_enqueue(PG, "byte", on_gate="deny")
    assert str(raised.value) == "system_frozen: db migration (until 2026-10-22T12:00:00+00:00)"


def test_enqueue_rejects_an_unknown_disposition(gate):
    with pytest.raises(ValueError):
        gates.admit_enqueue(PG, "byte", on_gate="maybe")
    assert gate.actors == []


# --- schedule ------------------------------------------------------------


@pytest.mark.parametrize("block", [FREEZE, QUIESCE])
def test_any_live_gate_holds_a_scheduled_occurrence(gate, block):
    gate.block = block
    assert gates.schedule_held(PG, "byte") is True
    assert gate.actors[0][1].kind == "schedule"
    gate.block = None
    assert gates.schedule_held(PG, "byte") is False


def test_held_schedule_is_a_no_op_that_leaves_the_occurrence_pending(rig, monkeypatch):  # noqa: F811 (pytest injects imported fixture)
    from tests.test_prompt_delivery import NOW, enqueue, occurrence, reserve

    held = SimpleNamespace(value=True, bots=[])

    def schedule_held(conn, bot_id):
        held.bots.append(bot_id)
        return held.value

    monkeypatch.setattr(gates, "schedule_held", schedule_held)
    _, run = reserve(rig)
    assert enqueue(rig, run) is None
    assert occurrence(rig, run).state == "pending"
    assert rig.delivery.calls == 0

    held.value = False
    record = enqueue(rig, run, NOW)
    assert record is not None and rig.delivery.calls == 1
    assert occurrence(rig, run).state == "queued"
    assert held.bots == [rig.bot.slug, rig.bot.slug]


# --- steer ---------------------------------------------------------------


@pytest.mark.parametrize("block,status", [(QUIESCE, 409), (FREEZE, 503)])
def test_steer_gate_refusal_is_a_typed_http_error(block, status):
    refused = gates.SteerGateRefused(block)
    assert isinstance(refused, HTTPException)
    assert (refused.status_code, refused.code, refused.block) == (status, block.refusal, block)
    assert refused.detail == str(SystemGateRefused(block))


def test_check_steer_skips_off_postgresql():
    sqlite = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))
    assert gates.check_steer(None, "byte", turn_id="t") is None
    assert gates.check_steer(sqlite, "byte", turn_id="t") is None


def _steer_service(prompt="work"):
    turn = SimpleNamespace(id="turn", bot_id="bot", user_id="user", user_prompt=prompt,
                           ended_at=None, status="streaming", agent_session_key=None,
                           trigger_message_id=None)
    engine = object()
    return SimpleNamespace(_turn_log_store=SimpleNamespace(get_turn=lambda _id: turn, engine=engine)), engine


def _steer(service, message="hi"):
    from llm_bawt.service.routes.chat import ChatSteerRequest, steer_active_turn

    request = ChatSteerRequest(turn_id="turn", bot_id="bot", user_id="user",
                               message_id=str(uuid.uuid4()), message=message)
    return asyncio.run(steer_active_turn(service, request))


@pytest.mark.parametrize("block", [QUIESCE, FREEZE])
def test_steer_route_refuses_under_a_gate_before_the_run_checks(monkeypatch, block):
    calls = []

    def check(engine, bot_id, *, turn_id):
        calls.append((engine, bot_id, turn_id))
        raise gates.SteerGateRefused(block)

    monkeypatch.setattr(gates, "check_steer", check)
    service, engine = _steer_service()
    with pytest.raises(gates.SteerGateRefused) as raised:
        _steer(service)
    assert raised.value.code == block.refusal
    assert calls == [(engine, "bot", "turn")]


def test_admitted_steer_continues_to_the_normal_checks(monkeypatch):
    monkeypatch.setattr(gates, "check_steer", lambda *_a, **_k: None)
    service, _ = _steer_service()
    with pytest.raises(HTTPException) as raised:
        _steer(service)
    assert raised.value.detail == "Active bridge run is not ready"


def test_a_composer_steer_to_an_ended_turn_reports_the_turn_not_the_gate(monkeypatch):
    monkeypatch.setattr(gates, "check_steer", lambda *_a, **_k: pytest.fail("ended turn is not gated"))
    service, _ = _steer_service()
    turn = service._turn_log_store.get_turn("turn")
    turn.ended_at, turn.status = UNTIL, "completed"
    with pytest.raises(HTTPException) as raised:
        _steer(service)
    assert raised.value.detail == "Turn is no longer active"


def test_a_durable_steer_retry_to_an_ended_turn_is_still_held(monkeypatch):
    """The dispatcher's retry waits out the gate; its ended-run fallback runs after release."""
    from llm_bawt.service.routes.chat import ChatSteerRequest, steer_active_turn

    monkeypatch.setattr(gates, "check_steer", lambda *_a, **_k: (_ for _ in ()).throw(
        gates.SteerGateRefused(QUIESCE)))
    service, _ = _steer_service()
    turn = service._turn_log_store.get_turn("turn")
    turn.ended_at, turn.status = UNTIL, "completed"
    request = ChatSteerRequest(turn_id="turn", bot_id="bot", user_id="user",
                               message_id=str(uuid.uuid4()), message="hi")
    with pytest.raises(gates.SteerGateRefused):
        asyncio.run(steer_active_turn(service, request, steer_request_id="steer-1"))


def _escalate(monkeypatch, service):
    from llm_bawt.service.routes import chat

    monkeypatch.setattr(chat, "get_service", lambda: service)
    return asyncio.run(chat.chat_steer_escalate(chat.ChatSteerEscalateRequest(
        turn_id="turn", message_id="m1", bot_id="bot", user_id="user")))


@pytest.mark.parametrize("block", [QUIESCE, FREEZE])
def test_escalation_interrupts_a_tool_so_a_gate_refuses_it(monkeypatch, block):
    calls = []

    def check(engine, bot_id, *, turn_id):
        calls.append((engine, bot_id, turn_id))
        raise gates.SteerGateRefused(block)

    monkeypatch.setattr(gates, "check_steer", check)
    monkeypatch.setattr("llm_bawt.agent_backends.agent_bridge.get_agent_subscriber",
                        lambda: pytest.fail("no bridge RPC under a gate"))
    service, engine = _steer_service()
    service._turn_log_store.get_turn("turn").agent_session_key = "session"
    service._turn_log_store.get_turn("turn").agent_request_id = "req"
    with pytest.raises(gates.SteerGateRefused) as raised:
        _escalate(monkeypatch, service)
    assert raised.value.code == block.refusal
    assert calls == [(engine, "bot", "turn")]


def test_admitted_escalation_continues_to_the_normal_checks(monkeypatch):
    monkeypatch.setattr(gates, "check_steer", lambda *_a, **_k: None)
    service, _ = _steer_service()
    with pytest.raises(HTTPException) as raised:
        _escalate(monkeypatch, service)
    assert raised.value.detail == "Active bridge run is not ready"


def test_a_maintenance_time_message_is_an_enqueue_not_a_gated_steer(monkeypatch):
    """A message during /compact is queued and asks the gate at enqueue instead."""
    monkeypatch.setattr(gates, "check_steer", lambda *_a, **_k: pytest.fail("not a steer"))
    queued = []

    async def enqueue_user(self, request, turn):
        queued.append(turn.id)
        return {"ok": True, "detail": "queued", "turn_id": turn.id, "message_id": request.message_id}

    monkeypatch.setattr("llm_bawt.service.maintenance_queue.MaintenanceQueue.enqueue_user", enqueue_user)
    service, _ = _steer_service(prompt="/compact")
    assert _steer(service).detail == "queued"
    assert queued == ["turn"]


# --- dispatcher ----------------------------------------------------------


@pytest.mark.parametrize("accepted", [False, True])
def test_held_steer_requeues_same_target_without_spending_an_attempt(monkeypatch, accepted):
    from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher

    async def refused(*_args, **_kwargs):
        raise gates.SteerGateRefused(QUIESCE)

    monkeypatch.setattr("llm_bawt.service.routes.chat.steer_active_turn", refused)
    monkeypatch.setattr("llm_bawt.inter_bot_reply_channel.delivery_reply_channel_note", lambda _r: None)
    # The last attempt (and maybe already transport-accepted): any other
    # classification would dead-letter it or replay and burn it.
    record = SimpleNamespace(
        id="delivery-1", claim_token="claim", target_turn_id="turn-1", target_bot_id="byte",
        user_message_id="uuid", message="hi", author=None, attempt_count=5, max_attempts=5,
        transport_accepted_at=datetime.now(timezone.utc) if accepted else None, metadata={},
    )
    requeued, failed = [], []
    dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
    dispatcher.store = SimpleNamespace(
        payload=lambda _id: {"messages": [{"content": "hi"}], "user": "nick"},
        get=lambda _id: record,
        requeue=lambda *args, **kw: requeued.append((args, kw)) or SimpleNamespace(status="QUEUED"),
        fail_claim=lambda *args: failed.append(args),
    )
    dispatcher.service = SimpleNamespace(config=SimpleNamespace(DEFAULT_USER="nick"))

    async def heartbeat(*_args):
        await asyncio.sleep(3600)

    async def emit(_value):
        return None

    dispatcher._heartbeat = heartbeat
    dispatcher._emit = emit
    asyncio.run(dispatcher._steer(record))

    assert not failed
    (args, kwargs), = requeued
    assert args[:2] == ("delivery-1", "claim")
    assert args[2].startswith("system_held: restart app")
    assert kwargs == {"delay_seconds": InterBotDeliveryDispatcher.TARGET_BUSY_REQUEUE_SECONDS,
                      "refund_attempt": True}


# --- producer dispositions -------------------------------------------------


def test_retry_policies_declare_their_gate_disposition():
    from llm_bawt.inter_bot_delivery_retry import MANUAL_RETRY, RESTART_RETRY

    assert MANUAL_RETRY.on_gate == "deny"
    assert RESTART_RETRY.on_gate == "hold"


@pytest.fixture
def capability(monkeypatch):
    from llm_bawt import task_turn_context as codec

    fernet = Fernet(Fernet.generate_key())
    monkeypatch.setattr(codec, "_get_fernet", lambda: fernet)

    def mint(bot_id="byte"):
        return codec.mint_task_turn_context(
            session_id=str(uuid.uuid4()), turn_id="turn-" + "a" * 32,
            trigger_message_id=str(uuid.uuid4()), bot_id=bot_id, user_id="nick",
        )

    return mint


def _route_request(header=None):
    from llm_bawt.task_turn_context import TASK_TURN_CONTEXT_HEADER

    return SimpleNamespace(headers={TASK_TURN_CONTEXT_HEADER: header} if header else {})


def _create(kind="MODEL_SWITCH_CONTINUE"):
    return SimpleNamespace(message_kind=kind)


def test_only_a_signed_self_model_continuation_is_held(capability):
    from llm_bawt.service.routes.inter_bot_deliveries import _gate_disposition

    token = capability("byte")
    assert _gate_disposition(_route_request(token), _create(), "byte", "byte") == "hold"
    assert _gate_disposition(_route_request(token), _create(), "loopy", "loopy") == "deny"
    assert _gate_disposition(_route_request(token), _create(), "byte", "loopy") == "deny"
    assert _gate_disposition(_route_request(token), _create("TASK_CALLBACK"), "byte", "byte") == "deny"
    assert _gate_disposition(_route_request(token), _create(None), "byte", "byte") == "deny"
    assert _gate_disposition(_route_request("not-a-capability"), _create(), "byte", "byte") == "deny"
    assert _gate_disposition(_route_request(), _create(), "byte", "byte") == "deny"
    assert _gate_disposition(None, _create(), "byte", "byte") == "deny"


@pytest.mark.parametrize(("signed", "sender", "status", "disposition"), [
    (True, "byte", 202, "hold"),   # the bot's own signed continuation waits out a freeze
    (False, "byte", 503, "deny"),  # the same body with no capability is refused
    (True, "loopy", 503, "deny"),  # a capability never lends hold to another sender
])
def test_the_public_route_reads_the_capability_header(
    monkeypatch, gate, capability, signed, sender, status, disposition,
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from llm_bawt.inter_bot_delivery_record import DeliveryRecord
    from llm_bawt.service.routes import inter_bot_deliveries as route
    from llm_bawt.task_turn_context import TASK_TURN_CONTEXT_HEADER

    gate.block = FREEZE
    now = datetime.now(timezone.utc)
    record = DeliveryRecord.from_mapping({
        "id": "dlv-1", "sender_bot_id": sender, "target_bot_id": sender, "message": "go on",
        "user_message_id": "u1", "turn_id": "t1", "status": "QUEUED", "attempt_count": 0,
        "max_attempts": 5, "created_at": now, "updated_at": now, "available_at": now,
        "session_policy": "continue", "author_entity_type": "bot", "author_entity_id": sender,
    })
    seen = []

    def enqueue(**kwargs):
        seen.append(kwargs["on_gate"])
        gates.admit_enqueue(None, kwargs["target_bot_id"], on_gate=kwargs["on_gate"])
        return record, False

    async def emit(_record):
        return None

    bot = SimpleNamespace(slug=sender, agent_backend="claude-code", harness="claude-code")
    monkeypatch.setattr(route, "BotManager", lambda _cfg: SimpleNamespace(get_bot=lambda _slug: bot))
    monkeypatch.setattr(route, "get_service", lambda: SimpleNamespace(config=SimpleNamespace(DEFAULT_USER="nick")))
    store = SimpleNamespace(enqueue=enqueue, get=lambda _id: record, payload=lambda _id: {})
    monkeypatch.setattr(route, "_dispatcher", lambda: SimpleNamespace(store=store, _emit=emit, wake=lambda _t: None))
    app = FastAPI()
    app.include_router(route.router)
    headers = {TASK_TURN_CONTEXT_HEADER: capability("byte")} if signed else {}
    response = TestClient(app).post("/v1/inter-bot-deliveries", headers=headers, json={
        "sender_bot_id": sender, "target_bot_id": sender, "message": "go on",
        "message_kind": "MODEL_SWITCH_CONTINUE", "session_policy": "continue",
    })
    assert response.status_code == status, response.text
    assert seen == [disposition]
    if status == 503:
        assert response.json()["detail"].startswith("system_frozen: db migration")


def test_self_model_continue_forwards_the_turn_capability(monkeypatch):
    import llm_bawt.mcp_server.server  # noqa: F401  registers tools; import first
    from llm_bawt.mcp_server import inter_bot_tools, self_model_tools, task_association
    from llm_bawt.task_turn_context import TASK_TURN_CONTEXT_HEADER

    sent = {}

    async def enqueue(**kwargs):
        sent.update(kwargs)
        return {"success": True}

    monkeypatch.setattr(inter_bot_tools, "_enqueue_durable", enqueue)
    monkeypatch.setattr(task_association, "current_task_turn_capability", lambda: "cap-token")
    asyncio.run(self_model_tools._enqueue_continue(target_bot_id="byte"))
    assert sent["headers"] == {TASK_TURN_CONTEXT_HEADER: "cap-token"}
    assert sent["prefer_steer"] is False

    monkeypatch.setattr(task_association, "current_task_turn_capability", lambda: None)
    asyncio.run(self_model_tools._enqueue_continue(target_bot_id="byte"))
    assert sent["headers"] is None


def test_bots_send_message_never_forwards_a_capability(monkeypatch):
    import llm_bawt.mcp_server.server  # noqa: F401  registers tools; import first
    from llm_bawt.mcp_server import inter_bot_tools

    posted = {}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def post(self, url, *, json, headers=None, timeout):
            posted["headers"] = headers
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"success": True})

    monkeypatch.setattr(inter_bot_tools.httpx, "AsyncClient", Client)
    asyncio.run(inter_bot_tools._enqueue_durable(
        prefer_steer=True, target_bot_id="byte", message="hi", sender_bot_id="al",
        max_tokens=None, temperature=0.7, timeout_seconds=60.0, idempotency_key=None,
        project_id=None, task_id=None, message_kind=None, metadata=None,
        session_policy=None, reset_session_before_delivery=None, retain_history=None,
        reset_reason=None,
    ))
    assert posted["headers"] is None


def test_message_route_reports_a_gate_refusal_with_its_status(monkeypatch):
    from llm_bawt.service.routes import message_routing as route

    class Router:
        def route(self, *_args, **_kwargs):
            raise SystemGateRefused(FREEZE)

    monkeypatch.setattr(route, "_router", lambda: Router())
    body = SimpleNamespace(request=lambda: None, message="hi", idempotency_key=None,
                           project_id=None, task_id=None, user_id="nick")
    with pytest.raises(HTTPException) as raised:
        route.route(body, SimpleNamespace(headers={}))
    assert raised.value.status_code == 503
    assert str(raised.value.detail).startswith("system_frozen: db migration")
