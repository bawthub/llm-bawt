"""TASK-1243: a steer is decided by the addressed run's own pinned capabilities.

The codec and decision helpers (``steer_capabilities``), the steer routes'
refusals and the dispatcher's classification of the two refusal codes. The
PostgreSQL halves (pin at bind, the delivery claim) are in
test_steer_caps_pg.py.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from agent_bridge.steer_contract import (
    STEER_CAPABILITY_REFUSALS,
    STEER_PRIORITY_UNSUPPORTED,
    STEER_UNSUPPORTED_TRANSPORT,
)
from llm_bawt.inter_bot_control_lane import pin_matches, pinned_steer_caps
from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher
from llm_bawt.steer_capabilities import (
    ALL_STEER_CAPS,
    CANCEL_QUEUED,
    ESCALATE,
    NEXT,
    NO_STEER_CAPS,
    NOW,
    decode_caps,
    encode_caps,
    run_steer_caps,
    steer_refusal,
    turn_steer_caps,
)

# --- codec --------------------------------------------------------------------


def test_encode_is_sorted_known_names_and_decode_round_trips():
    assert encode_caps({NOW, NEXT, "bogus"}) == "next,now"
    assert encode_caps(NO_STEER_CAPS) == ""
    assert encode_caps(ALL_STEER_CAPS) == "cancel_queued,escalate,next,now"
    assert decode_caps(encode_caps(ALL_STEER_CAPS)) == ALL_STEER_CAPS
    assert decode_caps(" next , bogus ,") == frozenset({NEXT})
    assert decode_caps("") == decode_caps(None) == NO_STEER_CAPS


@pytest.mark.parametrize("backend, caps, provisional, expected", [
    # A recorded value decides on any backend, '' included.
    ("codex", "next", None, {NEXT}),
    ("claude-code", "", ALL_STEER_CAPS, set()),
    # NULL on a bound run is a pre-TASK-1243 row: only claude-code steered.
    ("claude-code", None, NO_STEER_CAPS, ALL_STEER_CAPS),
    (" Claude-Code ", None, None, ALL_STEER_CAPS),
    ("codex", None, ALL_STEER_CAPS, set()),
    # No binding yet: the bot's provisional answer, else none.
    (None, None, frozenset({NEXT}), {NEXT}),
    ("", None, None, set()),
])
def test_run_steer_caps_reads_the_runs_own_pin(backend, caps, provisional, expected):
    assert run_steer_caps(backend=backend, caps=caps, provisional=provisional) == frozenset(expected)


def test_steer_refusal_names_transport_then_priority():
    assert steer_refusal(NO_STEER_CAPS, NEXT) == STEER_UNSUPPORTED_TRANSPORT
    assert steer_refusal(NO_STEER_CAPS, ESCALATE) == STEER_UNSUPPORTED_TRANSPORT
    assert steer_refusal(frozenset({NEXT}), NOW) == STEER_PRIORITY_UNSUPPORTED
    assert steer_refusal(frozenset({NEXT}), CANCEL_QUEUED) == STEER_PRIORITY_UNSUPPORTED
    assert steer_refusal(frozenset({NEXT}), NEXT) is None
    assert steer_refusal(ALL_STEER_CAPS, NOW) is None


def test_turn_steer_caps_prefers_the_rows_backend_over_the_bots():
    codex_row = SimpleNamespace(agent_backend="codex", agent_steer_caps=None)
    assert turn_steer_caps(codex_row, bot_backend="claude-code") == ("codex", NO_STEER_CAPS)
    unrecorded = SimpleNamespace(agent_backend=None)  # bound before TASK-1232 recorded it
    assert turn_steer_caps(unrecorded, bot_backend="claude-code") == ("claude-code", ALL_STEER_CAPS)
    pinned = SimpleNamespace(agent_backend="codex", agent_steer_caps="next")
    assert turn_steer_caps(pinned, bot_backend="codex") == ("codex", frozenset({NEXT}))


def test_a_legacy_control_pin_still_matches_and_decodes_from_its_backend():
    legacy = {"request_id": "req", "backend": "claude-code", "session_key": "s"}
    current = {**legacy, "steer_caps": "next"}
    assert pin_matches(legacy, current)  # stored before the key existed
    assert not pin_matches({**legacy, "steer_caps": ""}, current)
    assert not pin_matches({**legacy, "backend": "codex"}, current)
    assert pinned_steer_caps(legacy) == ALL_STEER_CAPS
    assert pinned_steer_caps({**legacy, "backend": "codex"}) == NO_STEER_CAPS
    assert pinned_steer_caps({**legacy, "backend": "codex", "steer_caps": "next"}) == frozenset({NEXT})


# --- dispatcher classification -------------------------------------------------


@pytest.mark.parametrize("error", sorted(STEER_CAPABILITY_REFUSALS))
def test_a_capability_refusal_falls_back_before_acceptance_only(error):
    classify = InterBotDeliveryDispatcher._steer_failure_action
    assert classify(error, accepted=False, target_open=True) == "fallback"
    assert classify(error, accepted=True, target_open=True) == "retry_same_steer"


# --- the steer routes ----------------------------------------------------------


def _turn(**fields):
    return SimpleNamespace(**{
        "id": "turn-live", "bot_id": "target", "user_id": "nick", "ended_at": None,
        "status": "streaming", "agent_session_key": "session-key", "agent_request_id": "req",
        "trigger_message_id": "trigger", **fields,
    })


def _harness(monkeypatch, turn, *, bot_backend=None, result=None):
    """A steer service around ``turn``. ``bot_backend`` None: BotManager must not be read."""
    from llm_bawt.agent_backends import agent_bridge
    from llm_bawt.bots import BotManager
    from llm_bawt.service.routes import chat

    service = SimpleNamespace(config=object(), _turn_log_store=SimpleNamespace(get_turn=lambda _id: turn),
                              get_memory_client=lambda *args: None)
    subscriber = SimpleNamespace(send_steer=AsyncMock(return_value=result or {"ok": True}))
    monkeypatch.setattr(agent_bridge, "get_agent_subscriber", lambda: subscriber)
    monkeypatch.setattr(chat, "get_service", lambda: service)

    def manager(*args):
        if bot_backend is None:
            raise AssertionError("the bot's current backend was consulted for a bound run")

    monkeypatch.setattr(BotManager, "__init__", manager)
    monkeypatch.setattr(BotManager, "get_bot", lambda *args: SimpleNamespace(agent_backend=bot_backend))
    monkeypatch.setattr("llm_bawt.steer_prompts.render_interrupt_frame", lambda bot_id: "<<steer_items>>")
    return chat, service, subscriber


def _steer(chat, service, priority="next"):
    return asyncio.run(chat.steer_active_turn(service, chat.ChatSteerRequest(
        turn_id="turn-live", bot_id="target", user_id="nick", message_id="message-steer",
        message="hello", priority=priority,
    )))


@pytest.mark.parametrize("fields, priority, code", [
    # A codex run bound before caps were recorded cannot steer, even though
    # the bot is claude-code now (a mid-run flip never changes the live run).
    ({"agent_backend": "codex", "agent_steer_caps": None}, "next", STEER_UNSUPPORTED_TRANSPORT),
    ({"agent_backend": "claude-code", "agent_steer_caps": ""}, "next", STEER_UNSUPPORTED_TRANSPORT),
    ({"agent_backend": "codex", "agent_steer_caps": "next"}, "now", STEER_PRIORITY_UNSUPPORTED),
])
def test_steer_refuses_from_the_runs_pin_before_any_bridge_rpc(monkeypatch, fields, priority, code):
    chat, service, subscriber = _harness(monkeypatch, _turn(**fields))
    with pytest.raises(HTTPException) as raised:
        _steer(chat, service, priority)
    assert (raised.value.status_code, raised.value.detail) == (409, code)
    subscriber.send_steer.assert_not_awaited()


@pytest.mark.parametrize("caps, offers", [
    ("next", (False, False)),
    ("cancel_queued,escalate,next", (True, True)),
])
def test_a_pinned_run_steers_on_its_own_backend_and_offers_only_its_caps(monkeypatch, caps, offers):
    turn = _turn(agent_backend="codex", agent_steer_caps=caps)
    chat, service, subscriber = _harness(
        monkeypatch, turn, result={"ok": True, "can_escalate": True, "can_cancel": True})
    result = _steer(chat, service)  # no memory client: accepted, persistence logged as failed
    assert subscriber.send_steer.await_args.kwargs["backend"] == "codex"
    assert (result.can_escalate, result.can_cancel) == offers


def test_a_claude_run_still_steers_after_the_bot_flips_to_codex(monkeypatch):
    turn = _turn(agent_backend="claude-code", agent_steer_caps=None)  # legacy row, bound as claude-code
    chat, service, subscriber = _harness(monkeypatch, turn)
    _steer(chat, service, "now")
    assert subscriber.send_steer.await_args.kwargs["backend"] == "claude-code"


def test_an_unrecorded_backend_falls_back_to_the_bots(monkeypatch):
    chat, service, subscriber = _harness(monkeypatch, _turn(), bot_backend="codex")
    with pytest.raises(HTTPException) as raised:
        _steer(chat, service)
    assert raised.value.detail == STEER_UNSUPPORTED_TRANSPORT
    subscriber.send_steer.assert_not_awaited()


def test_a_bridge_capability_refusal_is_a_definite_409(monkeypatch):
    turn = _turn(agent_backend="codex", agent_steer_caps="next")
    chat, service, subscriber = _harness(
        monkeypatch, turn, result={"ok": False, "error": STEER_UNSUPPORTED_TRANSPORT})
    with pytest.raises(HTTPException) as raised:
        _steer(chat, service)
    assert (raised.value.status_code, raised.value.detail) == (409, STEER_UNSUPPORTED_TRANSPORT)
    assert subscriber.send_steer.await_count == 1  # not the composer's no_active_run retry loop


@pytest.mark.parametrize("route, caps, code", [
    ("chat_steer_escalate", "next", STEER_PRIORITY_UNSUPPORTED),
    ("chat_steer_cancel", "escalate,next", STEER_PRIORITY_UNSUPPORTED),
    ("chat_steer_escalate", None, STEER_UNSUPPORTED_TRANSPORT),
    ("chat_steer_cancel", None, STEER_UNSUPPORTED_TRANSPORT),
])
def test_escalate_and_cancel_refuse_without_their_capability(monkeypatch, route, caps, code):
    chat, _service, subscriber = _harness(monkeypatch, _turn(agent_backend="codex", agent_steer_caps=caps))
    with pytest.raises(HTTPException) as raised:
        asyncio.run(getattr(chat, route)(chat.ChatSteerEscalateRequest(
            turn_id="turn-live", message_id="message-1", bot_id="target", user_id="nick")))
    assert (raised.value.status_code, raised.value.detail) == (409, code)
    subscriber.send_steer.assert_not_awaited()


@pytest.mark.parametrize("route", ["chat_steer_escalate", "chat_steer_cancel"])
def test_escalate_and_cancel_address_the_pinned_runs_backend(monkeypatch, route):
    turn = _turn(agent_backend="codex", agent_steer_caps="cancel_queued,escalate,next")
    chat, _service, subscriber = _harness(monkeypatch, turn, result={"ok": True, "detail": "done"})
    asyncio.run(getattr(chat, route)(chat.ChatSteerEscalateRequest(
        turn_id="turn-live", message_id="message-1", bot_id="target", user_id="nick")))
    assert subscriber.send_steer.await_args.kwargs["backend"] == "codex"
