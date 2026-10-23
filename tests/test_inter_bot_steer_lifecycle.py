"""The app side of a bot steer's lifecycle, without PostgreSQL (TASK-1300).

Row behaviour is in test_inter_bot_steer_lifecycle_pg.py. Here: which events
reach the fence, how the RPC id names its row, the agent backend's consumer,
and the registry-rendered steer framing.
"""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from agent_bridge.steer_contract import STEER_ITEMS_SLOT
from llm_bawt import steer_prompts
from llm_bawt.inter_bot_delivery_steer_fence import IGNORED, PENDING, REOPENED
from llm_bawt.service import inter_bot_steer_lifecycle as lifecycle
from llm_bawt.service.inter_bot_dispatcher_steer import SteerGeneration

HEX = "0123456789abcdef0123456789abcdef"
DELIVERY = f"delivery-{HEX}"
RPC = f"steer_delivery_{HEX}"
TURN = "turn-target"


# --- the RPC id names its row and generation ---------------------------------


@pytest.mark.parametrize("generation", [0, 1, 7, 12])
def test_the_request_id_parses_back_to_its_delivery_and_generation(generation):
    record = SimpleNamespace(id=DELIVERY, metadata={"steer_generation": generation})
    assert SteerGeneration.parse_request_id(SteerGeneration.of(record).request_id) == (DELIVERY, generation)


@pytest.mark.parametrize("request_id", [
    None, "", "steer_abc", f"steer_user_{HEX}", f"steer_delivery_{HEX[:-1]}",
    f"steer_delivery_{HEX.upper()}", f"steer_delivery_{HEX}_g0", f"steer_delivery_{HEX}_g01",
    f"steer_delivery_{HEX}_g", f"steer_delivery_{HEX}_g2x", f"xsteer_delivery_{HEX}",
])
def test_any_other_request_id_names_no_row(request_id):
    assert SteerGeneration.parse_request_id(request_id) is None


# --- which events reach the fence ---------------------------------------------


class _Fence:
    calls: list[tuple] = []
    reopen: tuple = (REOPENED, SimpleNamespace(id=DELIVERY))
    started = True
    uncertain = SimpleNamespace(id=DELIVERY)

    def __init__(self, store):
        self.store = store

    def note_steer_started(self, *args):
        self.calls.append(("started", *args))
        return self.started

    def reopen_rejected_steer(self, *args):
        self.calls.append(("reopen", *args))
        return self.reopen

    def mark_steer_uncertain(self, *args):
        self.calls.append(("uncertain", *args))
        return self.uncertain


@pytest.fixture
def fence(monkeypatch):
    _Fence.calls = []
    _Fence.reopen, _Fence.started = (REOPENED, SimpleNamespace(id=DELIVERY)), True
    _Fence.uncertain = SimpleNamespace(id=DELIVERY)
    monkeypatch.setattr(lifecycle, "SteerLifecycleFence", _Fence)
    return _Fence


@pytest.fixture
def dispatcher():
    announced, lifecycles = [], []
    return SimpleNamespace(store=object(), announce_admitted=announced.append, announced=announced,
                           announce_lifecycle=lifecycles.append, lifecycles=lifecycles)


def _payload(state, *, proof=None, origin="bot", rpc=f"{RPC}_g2"):
    return {"message_id": "cmd", "state": state, "origin": origin, "steer_rpc_id": rpc,
            "run_request_id": "req", **({"proof": proof} if proof else {})}


def _apply(dispatcher, payload, turn=TURN):
    return lifecycle.apply_bot_steer_lifecycle(dispatcher, payload, target_turn_id=turn)


@pytest.mark.parametrize("state", ["started", "completed"])
def test_a_start_is_recorded_for_its_own_generation(fence, dispatcher, state):
    assert _apply(dispatcher, _payload(state)) == lifecycle.STARTED
    assert fence.calls == [("started", DELIVERY, 2, TURN)]
    fence.started = False
    assert _apply(dispatcher, _payload(state)) == IGNORED


@pytest.mark.parametrize(("state", "proof"), [
    ("cancelled", "sweep_receipt"), ("discarded", "cli_terminal"), ("refused", "cli_terminal"),
])
def test_a_proven_rejection_reopens_and_wakes_the_dispatcher(fence, dispatcher, state, proof, caplog):
    with caplog.at_level(logging.INFO, logger=lifecycle.__name__):
        assert _apply(dispatcher, _payload(state, proof=proof)) == REOPENED
    assert fence.calls == [("reopen", DELIVERY, 2, TURN, proof)]
    assert [r.id for r in dispatcher.announced] == [DELIVERY]
    assert any(DELIVERY in r.getMessage() and proof in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("outcome", [PENDING, IGNORED])
def test_only_a_reopen_is_announced(fence, dispatcher, outcome):
    fence.reopen = (outcome, SimpleNamespace(id=DELIVERY) if outcome == PENDING else None)
    assert _apply(dispatcher, _payload("cancelled", proof="sweep_receipt")) == outcome
    assert dispatcher.announced == []


@pytest.mark.parametrize("proof", [None, "replayed", "teardown"])
def test_a_rejection_without_a_not_run_proof_changes_nothing(fence, dispatcher, proof):
    assert _apply(dispatcher, _payload("cancelled", proof=proof)) == IGNORED
    assert fence.calls == [] and dispatcher.announced == []


def test_an_uncertain_outcome_is_fenced_and_published_to_the_sender(fence, dispatcher):
    assert _apply(dispatcher, _payload("uncertain", proof="barrier_timeout")) == lifecycle.UNCERTAIN
    assert fence.calls == [("uncertain", DELIVERY, 2, TURN, "barrier_timeout")]
    assert [r.id for r in dispatcher.lifecycles] == [DELIVERY]  # published, not woken
    assert dispatcher.announced == []
    fence.uncertain = None
    assert _apply(dispatcher, _payload("uncertain")) == IGNORED
    assert fence.calls[-1] == ("uncertain", DELIVERY, 2, TURN, "uncertain")
    assert len(dispatcher.lifecycles) == 1


def test_a_published_delivery_event_carries_the_rows_revision():
    """Codex r2: publishing is async, so the event must carry the committed
    row revision for consumers to drop a stale snapshot."""
    from datetime import datetime, timezone
    from unittest.mock import AsyncMock

    from llm_bawt.inter_bot_delivery import DeliveryRecord
    from tests.test_inter_bot_dispatcher_supervision import _make_dispatcher

    now = datetime.now(timezone.utc)
    record = DeliveryRecord.from_mapping({
        "id": DELIVERY, "sender_bot_id": "byte", "target_bot_id": "codex", "message": "hi",
        "user_message_id": "m", "turn_id": "t", "status": "DELIVERED", "attempt_count": 1,
        "max_attempts": 5, "created_at": now, "updated_at": now, "available_at": now,
        "delivery_mode": "steer", "revision": 7,
        "metadata_json": {"steer_uncertain": {"generation": 0, "reason": "teardown"}},
    })
    assert record.revision == 7 and record.to_api()["revision"] == 7
    real = _make_dispatcher()
    real.service._redis_subscriber.publish_tool_event = AsyncMock()
    asyncio.run(real._emit(record))
    event = real.service._redis_subscriber.publish_tool_event.await_args.args[2]
    assert event["revision"] == 7
    assert event["steer_uncertain"] == {"generation": 0, "reason": "teardown"}


@pytest.mark.parametrize("payload", [
    _payload("cancelled", proof="sweep_receipt", origin="user"),
    _payload("cancelled", proof="sweep_receipt", origin="system"),
    _payload("cancelled", proof="sweep_receipt", origin=None),
    _payload("queued"),
])
def test_non_bot_and_non_terminal_events_never_touch_a_row(fence, dispatcher, payload):
    assert _apply(dispatcher, payload) == IGNORED
    assert fence.calls == [] and dispatcher.announced == []


@pytest.mark.parametrize(("rpc", "turn"), [(f"steer_user_{HEX}", TURN), (None, TURN), (RPC, None), (RPC, "")])
def test_an_event_without_a_delivery_identity_is_logged_and_ignored(fence, dispatcher, rpc, turn, caplog):
    with caplog.at_level(logging.WARNING, logger=lifecycle.__name__):
        assert _apply(dispatcher, _payload("cancelled", proof="sweep_receipt", rpc=rpc), turn) == IGNORED
    assert fence.calls == []
    assert any("without a delivery identity" in r.getMessage() for r in caplog.records)


# --- the agent backend's consumer ---------------------------------------------


def test_the_consumer_settles_on_the_runs_own_turn(monkeypatch):
    from llm_bawt.agent_backends import agent_bridge
    from llm_bawt.service import dependencies

    seen = []
    dispatcher = SimpleNamespace()
    monkeypatch.setattr(dependencies, "get_service", lambda: SimpleNamespace(_inter_bot_dispatcher=dispatcher))
    monkeypatch.setattr(lifecycle, "apply_bot_steer_lifecycle",
                        lambda d, p, *, target_turn_id: seen.append((d, p, target_turn_id)) or REOPENED)
    payload = _payload("cancelled", proof="sweep_receipt")
    asyncio.run(agent_bridge._settle_bot_steer(payload, SimpleNamespace(turn_id=TURN)))
    assert seen == [(dispatcher, payload, TURN)]


@pytest.mark.parametrize("failure", ["no_dispatcher", "raises"])
def test_the_consumer_never_breaks_the_event_stream(monkeypatch, failure):
    from llm_bawt.agent_backends import agent_bridge
    from llm_bawt.service import dependencies

    def boom(*_a, **_k):
        raise RuntimeError("db down")

    monkeypatch.setattr(dependencies, "get_service", lambda: SimpleNamespace(
        _inter_bot_dispatcher=None if failure == "no_dispatcher" else SimpleNamespace()))
    monkeypatch.setattr(lifecycle, "apply_bot_steer_lifecycle", boom)
    asyncio.run(agent_bridge._settle_bot_steer(_payload("started"), SimpleNamespace(turn_id=TURN)))


def test_the_run_stream_settles_only_bot_steers_and_publishes_every_one(monkeypatch):
    """AgentBridgeBackend.stream_raw: a bot-origin steer_lifecycle settles its
    delivery; a user one is only published. Both carry origin and proof."""
    from agent_bridge.events import AgentEvent, AgentEventKind
    from llm_bawt.agent_backends import agent_bridge as module

    settled, published = [], []

    async def settle(payload, execution):
        settled.append(payload)

    def lifecycle_event(origin, proof):
        return AgentEvent(event_id=f"e-{origin}", session_key="byte:nick", run_id="req",
                          kind=AgentEventKind.STEER_LIFECYCLE, origin="system",
                          raw=_payload("cancelled", proof=proof, origin=origin))

    class Subscriber:
        def __init__(self, _url):
            pass

        async def connect(self):
            return None

        async def send_command(self, **_kwargs):
            return None

        async def subscribe_run(self, *_args, **_kwargs):
            yield lifecycle_event("user", None)
            yield lifecycle_event("bot", "sweep_receipt")

        async def publish_tool_event(self, bot_id, user_id, event):
            published.append(event)

        async def close(self):
            return None

    root = SimpleNamespace(_redis=SimpleNamespace(connection_pool=SimpleNamespace(
        connection_kwargs={"url": "redis://localhost:6379/0"})))
    monkeypatch.setattr(module, "get_agent_subscriber", lambda: root)
    monkeypatch.setattr("agent_bridge.subscriber.RedisSubscriber", Subscriber)
    monkeypatch.setattr(module, "_settle_bot_steer", settle)
    list(module.AgentBridgeBackend().stream_raw(
        "hello", {"bot_id": "byte", "user_id": "nick", "model": "claude-opus-5-5", "timeout_seconds": 1},
    ))
    assert [p["origin"] for p in settled] == ["bot"]
    assert [(e["origin"], e["proof"], e["state"]) for e in published] == [
        ("user", None, "cancelled"), ("bot", "sweep_receipt", "cancelled")]


# --- registry-rendered steer framing ------------------------------------------


@pytest.fixture
def registry(monkeypatch):
    from llm_bawt import prompt_registry

    bodies: dict[str, str] = {}
    seen: list[tuple] = []

    def resolve(key, **kwargs):
        seen.append((key, kwargs))
        return SimpleNamespace(body=bodies.get(key, ""))

    monkeypatch.setattr(prompt_registry, "get_prompt_resolver", lambda: SimpleNamespace(resolve=resolve))
    return SimpleNamespace(bodies=bodies, seen=seen)


@pytest.fixture
def registered(monkeypatch):
    """Swap a steer key's registered default body, as the registry sees it."""
    import dataclasses

    from llm_bawt.prompt_definitions import DEFAULT_PROMPT_DEFINITIONS

    def register(key, body):
        """``body`` is the default text, or an exception the loader raises."""
        def loader():
            if isinstance(body, Exception):
                raise body
            return body

        monkeypatch.setitem(DEFAULT_PROMPT_DEFINITIONS, key, dataclasses.replace(
            DEFAULT_PROMPT_DEFINITIONS[key], loader=loader))

    return register


def test_the_steer_prompts_are_registered_with_their_seeds():
    from llm_bawt.prompt_definitions import DEFAULT_PROMPT_DEFINITIONS

    for key, slot, seed in ((steer_prompts.NEXT_ASIDE_KEY, "message", steer_prompts.DEFAULT_NEXT_ASIDE),
                            (steer_prompts.INTERRUPT_REPLACEMENT_KEY, "messages",
                             steer_prompts.DEFAULT_INTERRUPT_REPLACEMENT)):
        definition = DEFAULT_PROMPT_DEFINITIONS[key]
        assert definition.default_body() == seed
        assert definition.required_vars == (slot,)
        assert "{" + slot + "}" in seed


def test_a_live_body_is_resolved_per_bot_and_braces_in_the_message_survive(registry):
    registry.bodies[steer_prompts.NEXT_ASIDE_KEY] = "[aside] {message} [go on]"
    assert steer_prompts.render_next("use {x} and {}", bot_id="byte") == "[aside] use {x} and {} [go on]"
    assert registry.seen == [(steer_prompts.NEXT_ASIDE_KEY, {"scope_type": "bot", "scope_id": "byte"})]


@pytest.mark.parametrize("body", ["", "   ", "no slot here", "{bogus}", "{messages", "{0}",
                                  "{messages.x}", "{messages:d}"])
def test_a_body_that_cannot_carry_the_message_falls_back_to_the_registered_default(
        registry, registered, body):
    # The fallback is read from the registry's definition, not this module's text.
    registered(steer_prompts.INTERRUPT_REPLACEMENT_KEY, "[registered] {messages}")
    registry.bodies[steer_prompts.INTERRUPT_REPLACEMENT_KEY] = body
    assert steer_prompts.render_now("hello", bot_id="byte") == "[registered] hello"


def test_a_registry_failure_falls_back_to_the_registered_default(monkeypatch, registered):
    from llm_bawt import prompt_registry

    def broken():
        raise RuntimeError("registry down")

    monkeypatch.setattr(prompt_registry, "get_prompt_resolver", broken)
    registered(steer_prompts.NEXT_ASIDE_KEY, "[registered] {message}")
    assert steer_prompts.render_next("hi", bot_id="byte") == "[registered] hi"


@pytest.mark.parametrize("default", ["", "no slot here", "{bogus}", "{messages.x}",
                                     RuntimeError("seed loader failed")])
def test_a_steer_no_registered_body_can_frame_is_refused(registry, registered, default, caplog):
    registered(steer_prompts.INTERRUPT_REPLACEMENT_KEY, default)
    registry.bodies[steer_prompts.INTERRUPT_REPLACEMENT_KEY] = "still {nothing}"
    with caplog.at_level(logging.ERROR, logger="llm_bawt.steer_prompts"), \
            pytest.raises(steer_prompts.SteerPromptUnavailable) as caught:
        steer_prompts.render_interrupt_frame("byte")
    assert str(caught.value).startswith(f"steer_prompt_unavailable: {steer_prompts.INTERRUPT_REPLACEMENT_KEY}")
    assert any("registered default" in r.getMessage() for r in caplog.records)


def test_the_interrupt_frame_leaves_the_bridges_item_slot(registry):
    registry.bodies[steer_prompts.INTERRUPT_REPLACEMENT_KEY] = "[stop] {messages} [resume]"
    assert steer_prompts.render_interrupt_frame("byte") == f"[stop] {STEER_ITEMS_SLOT} [resume]"


def test_steer_framing_by_origin_and_priority(registry):
    from llm_bawt.service.routes.chat import _steer_framing

    registry.bodies[steer_prompts.NEXT_ASIDE_KEY] = "<next>{message}</next>"
    registry.bodies[steer_prompts.INTERRUPT_REPLACEMENT_KEY] = "<now>{messages}</now>"
    kw = {"bot_id": "byte", "user_id": "nick"}
    assert _steer_framing("quiesce", origin="system", priority="now", **kw) == {}
    assert _steer_framing("hi", origin="user", priority="next", **kw) == {
        "framed_message": "<next>hi</next>", "interrupt_item": "Message from user 'nick': hi"}
    assert _steer_framing("hi", origin="user", priority="now", **kw) == {
        "framed_message": "<now>Message from user 'nick': hi</now>"}
    bot = "Message from bot 'snark': hi"
    assert _steer_framing(bot, origin="bot", priority="next", **kw) == {
        "framed_message": f"<next>{bot}</next>", "interrupt_item": bot}


@pytest.mark.parametrize("live", ["", "[live] {message}"])
def test_an_unregistered_steer_key_is_refused_and_logged(monkeypatch, registry, caplog, live):
    """Even a stale DB override that still renders can't frame an unregistered key."""
    from llm_bawt.prompt_definitions import DEFAULT_PROMPT_DEFINITIONS

    registry.bodies[steer_prompts.NEXT_ASIDE_KEY] = live
    monkeypatch.delitem(DEFAULT_PROMPT_DEFINITIONS, steer_prompts.NEXT_ASIDE_KEY)
    with caplog.at_level(logging.ERROR, logger="llm_bawt.steer_prompts"), \
            pytest.raises(steer_prompts.SteerPromptUnavailable) as caught:
        steer_prompts.render_next("hi", bot_id="byte")
    assert caught.value.key == steer_prompts.NEXT_ASIDE_KEY
    assert any(r.levelno == logging.ERROR and "not registered" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("default, live", [
    ("no slot", "no slot either"),
    (RuntimeError("seed loader failed"), "no slot either"),
    (None, "[stale] {message}"),  # unregistered key with a still-renderable DB override
])
def test_a_steer_that_cannot_be_framed_is_refused_before_the_bridge(
        monkeypatch, registry, registered, default, live):
    from llm_bawt.prompt_definitions import DEFAULT_PROMPT_DEFINITIONS
    from fastapi import HTTPException

    from llm_bawt import bots
    from llm_bawt.agent_backends import agent_bridge
    from llm_bawt.service.routes.chat import ChatSteerRequest, steer_active_turn

    sends: list[dict] = []

    async def send_steer(**kwargs):
        sends.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(agent_bridge, "get_agent_subscriber", lambda: SimpleNamespace(send_steer=send_steer))
    monkeypatch.setattr(bots, "BotManager", lambda _config: SimpleNamespace(
        get_bot=lambda _bot_id: SimpleNamespace(agent_backend="claude-code")))
    if default is None:
        monkeypatch.delitem(DEFAULT_PROMPT_DEFINITIONS, steer_prompts.NEXT_ASIDE_KEY)
    else:
        registered(steer_prompts.NEXT_ASIDE_KEY, default)
    registry.bodies[steer_prompts.NEXT_ASIDE_KEY] = live
    turn = SimpleNamespace(
        id="turn-live", bot_id="byte", user_id="nick", status="streaming", ended_at=None,
        agent_session_key="byte:nick", agent_request_id="req-live", user_prompt="work",
    )
    service = SimpleNamespace(config=None, _turn_log_store=SimpleNamespace(get_turn=lambda _id: turn))
    with pytest.raises(HTTPException) as caught:
        asyncio.run(steer_active_turn(service, ChatSteerRequest(
            turn_id=turn.id, message="hi", message_id="m-1", bot_id="byte", user_id="nick",
            priority="next")))
    assert caught.value.status_code == 503
    assert caught.value.detail.startswith(f"steer_prompt_unavailable: {steer_prompts.NEXT_ASIDE_KEY}")
    assert sends == []
