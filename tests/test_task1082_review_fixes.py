"""TASK-1082 review fixes: shared admission branch, in-band stream refusal,
and botchat waiting on its exact admitted turn. No live DB, bridge or bot."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.bot_types import durable_agent_backend
from llm_bawt.service.delivery_reply import (
    ACCEPTED_RECEIPTS, IN_FLIGHT_RECEIPT, QUEUED_RECEIPT, is_accepted_receipt,
    wait_for_receipt_reply,
)
from llm_bawt.service.ordinary_turn_queue import OrdinaryTurnDecision, admit_ordinary_agent_turn
from llm_bawt.service.schemas import ChatCompletionRequest, ChatMessage
from llm_bawt.service.turn_admission import (
    ADMISSION_KEY, AdmissionClaim, AdmissionUnavailable, TurnAdmission, turn_start_extensions,
)
from tests.test_nonstream_turn_execution import _service

MSG = "00000000-0000-4000-8000-000000000001"
QUEUED = {
    "object": QUEUED_RECEIPT, "status": "QUEUED", "delivery_id": "delivery-r",
    "turn_id": "turn-delivery-r", "user_message_id": MSG,
    "assistant_message_id": "00000000-0000-4000-8000-000000000002",
}


def _request(**update) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model", bot_id="caid", user="nick", user_message_id=MSG,
        messages=[ChatMessage(role="user", content="hello")], **update,
    )


# --- one durable-harness definition ----------------------------------------

@pytest.mark.parametrize("bot,expected", [
    (SimpleNamespace(agent_backend="claude-code", harness=None), "claude-code"),
    (SimpleNamespace(agent_backend=None, harness="codex"), "codex"),
    (SimpleNamespace(agent_backend="", harness="OpenClaw"), "openclaw"),
    (SimpleNamespace(agent_backend="local", harness="chat"), None),
    (SimpleNamespace(), None),
])
def test_durable_agent_backend_checks_backend_and_harness(bot, expected):
    assert durable_agent_backend(bot) == expected


def test_receipt_objects_have_one_definition():
    assert ACCEPTED_RECEIPTS == {QUEUED_RECEIPT, IN_FLIGHT_RECEIPT}
    assert is_accepted_receipt(dict(QUEUED))
    assert not is_accepted_receipt({"object": "chat.completion"})
    assert not is_accepted_receipt("chat.completion.queued")


# --- shared admission branch -------------------------------------------------

def _decision():
    return OrdinaryTurnDecision("queued", "turn-x", MSG, "assistant", "delivery-x", "QUEUED")


@pytest.mark.parametrize("is_agent,claim,bot", [
    (False, None, SimpleNamespace(agent_backend="claude-code")),
    (True, AdmissionClaim(owner="o", fence=lambda _c: True), SimpleNamespace(agent_backend="claude-code")),
    (True, None, SimpleNamespace(agent_backend="local", harness="chat")),
])
def test_direct_starts_bypass_the_ordinary_queue(monkeypatch, is_agent, claim, bot):
    admit = Mock(side_effect=AssertionError("must not enqueue"))
    monkeypatch.setattr("llm_bawt.service.ordinary_turn_queue.admit_agent_request", admit)
    assert admit_ordinary_agent_turn(
        SimpleNamespace(), bot, _request(), is_agent_backend=is_agent, admission_claim=claim,
        bot_id="caid", user_id="nick", user_prompt="hello", trigger_message_id=MSG,
    ) is None


@pytest.mark.parametrize("row_status,announced", [("QUEUED", True), ("DISPATCHING", False), (None, False)])
def test_harness_only_durable_bot_is_queued_and_announced(monkeypatch, row_status, announced):
    """Every tab learns of a fresh QUEUED row; anything else just wakes."""
    admit = Mock(return_value=_decision())
    monkeypatch.setattr("llm_bawt.service.ordinary_turn_queue.admit_agent_request", admit)
    row = SimpleNamespace(id="delivery-x", status=row_status) if row_status else None
    dispatcher = SimpleNamespace(wake=Mock(), announce_admitted=Mock(),
                                 store=SimpleNamespace(get=Mock(return_value=row)))
    decision = admit_ordinary_agent_turn(
        SimpleNamespace(_inter_bot_dispatcher=dispatcher),
        SimpleNamespace(agent_backend=None, harness="codex"), _request(),
        is_agent_backend=True, admission_claim=None, bot_id="caid", user_id="nick",
        user_prompt="hello", trigger_message_id=MSG,
    )
    assert decision.delivery_id == "delivery-x"
    assert admit.call_args.args[1].user_message_id == MSG
    if announced:
        dispatcher.announce_admitted.assert_called_once_with(row)
        dispatcher.wake.assert_not_called()
    else:
        dispatcher.announce_admitted.assert_not_called()
        dispatcher.wake.assert_called_once_with("caid")


def test_announce_publishes_queued_before_waking_the_dispatcher():
    from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher

    order: list[str] = []
    dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
    dispatcher._announce_tasks = set()

    async def emit(record):
        order.append(f"emit:{record.status}")

    class Wake:
        def set(self):
            order.append("wake")

    dispatcher._emit = emit
    dispatcher._wake_event = Wake()

    async def run():
        dispatcher._loop = asyncio.get_running_loop()
        dispatcher.announce_admitted(SimpleNamespace(id="d", status="QUEUED", target_bot_id="caid"))
        for _ in range(5):
            await asyncio.sleep(0)
        assert not dispatcher._announce_tasks, "announce task must not leak"

    asyncio.run(run())
    assert order == ["emit:QUEUED", "wake"]


def test_turn_start_extensions_marks_only_unclaimed_agent_starts_idle_only():
    claim = AdmissionClaim(owner="o", fence=lambda _c: True)
    idle = TurnAdmission.IDLE_ONLY.value
    assert turn_start_extensions(_request(), is_agent_backend=True, admission_claim=None)[ADMISSION_KEY] == idle
    assert ADMISSION_KEY not in turn_start_extensions(_request(), is_agent_backend=True, admission_claim=claim)
    assert ADMISSION_KEY not in turn_start_extensions(_request(), is_agent_backend=False, admission_claim=None)


@pytest.mark.parametrize("stream", [True, False])
def test_both_paths_return_the_shared_receipt_without_starting(monkeypatch, stream):
    service = _service(Mock(side_effect=AssertionError("no execution")))
    monkeypatch.setattr("llm_bawt.service.dependencies.get_service", lambda: service)
    monkeypatch.setattr(
        "llm_bawt.service.chat_streaming.prepare_stream_request_attachments",
        AsyncMock(return_value=("hello", [], [], None)),
    )
    module = "chat_streaming" if stream else "chat_nonstream"
    monkeypatch.setattr(f"llm_bawt.service.{module}.admit_ordinary_agent_turn", Mock(return_value=_decision()))

    async def run():
        if not stream:
            return await service.chat_completion(_request())
        return [event async for event in service.chat_completion_stream(_request(stream=True))
                if event.startswith("data: ")]

    result = asyncio.run(run())
    receipt = json.loads(result[0][6:]) if stream else result
    assert receipt["object"] == QUEUED_RECEIPT and receipt["delivery_id"] == "delivery-x"
    service._persist_turn_log.assert_not_called()
    service._start_generation.assert_not_called()


def test_stream_admission_refusal_is_reported_in_band(monkeypatch):
    """Headers are already sent: refusal must be an explicit event, not a cut."""
    service = _service(Mock(side_effect=AssertionError("no execution")))
    monkeypatch.setattr("llm_bawt.service.dependencies.get_service", lambda: service)
    monkeypatch.setattr(
        "llm_bawt.service.chat_streaming.prepare_stream_request_attachments",
        AsyncMock(return_value=("hello", [], [], None)),
    )
    monkeypatch.setattr(
        "llm_bawt.service.chat_streaming.admit_ordinary_agent_turn",
        Mock(side_effect=AdmissionUnavailable("delivery database unavailable")),
    )

    async def run():
        return [event async for event in service.chat_completion_stream(_request(stream=True))
                if event.startswith("data: ")]

    events = asyncio.run(run())
    warning = json.loads(events[0][6:])
    assert warning["object"] == "service.warning"
    assert "delivery database unavailable" in warning["warnings"][0]
    assert json.loads(events[1][6:])["choices"][0]["finish_reason"] == "stop"
    assert events[-1] == "data: [DONE]\n\n"
    service._persist_turn_log.assert_not_called()


# --- botchat waits on the exact admitted turn ---------------------------------

def _stores(*, status="DELIVERED", turn_status="ok", text="the answer", error=None):
    record = SimpleNamespace(
        id="delivery-r", turn_id="turn-delivery-r", status=status, last_error=error,
        sender_bot_id="review-bot", target_bot_id="review-bot", author_entity_type="user", author_entity_id="review-user",
        user_message_id=MSG, metadata={"ordinary_chat": {"digest": "d"}},
    )
    turn = SimpleNamespace(
        ended_at=datetime.now(timezone.utc), status=turn_status, error_text=None,
        bot_id="review-bot", user_id="review-user", trigger_message_id=MSG, response_text=text,
    )
    # ``list`` backs the FAILED-original retry-child lookup (TASK-1231B); none here.
    deliveries = SimpleNamespace(get=lambda _id: record, list=lambda **_kw: [])
    return deliveries, SimpleNamespace(get_turn=lambda _id: turn)


def _botchat(monkeypatch, *, stream=False, **store_kw):
    from llm_bawt.service.routes import botchat

    deliveries, turns = _stores(**store_kw)

    async def stream_receipt(_request):
        yield f"data: {json.dumps(QUEUED)}\n\n"
        yield "data: [DONE]\n\n"

    service = SimpleNamespace(
        chat_completion=AsyncMock(return_value=dict(QUEUED)),
        chat_completion_stream=stream_receipt,
        _inter_bot_dispatcher=SimpleNamespace(store=deliveries), _turn_log_store=turns,
    )
    monkeypatch.setattr(botchat, "get_service", lambda: service)
    app = FastAPI()
    app.include_router(botchat.router)
    with TestClient(app) as client:
        return client.post("/v1/botchat/review-bot/review-user/chat/completions", json={
            "model": "test-model", "stream": stream,
            "messages": [{"role": "user", "content": "hello"}],
        })


def test_botchat_returns_the_admitted_turns_reply(monkeypatch):
    result = _botchat(monkeypatch)
    assert result.status_code == 200
    assert result.json()["choices"][0]["message"]["content"] == "the answer"


def test_botchat_failed_delivery_is_an_error_not_an_empty_reply(monkeypatch):
    result = _botchat(monkeypatch, status="FAILED", error="bridge rejected")
    assert result.status_code == 502
    assert "bridge rejected" in result.json()["detail"]


def test_botchat_stream_replaces_receipt_with_reply(monkeypatch):
    result = _botchat(monkeypatch, stream=True)
    events = [line[6:] for line in result.text.split("\n\n") if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert all(not is_accepted_receipt(c) for c in chunks), "raw receipt leaked to OpenAI client"
    assert chunks[0]["choices"][0]["delta"]["content"] == "the answer"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_receipt_waiter_follows_in_flight_turn_identity():
    _deliveries, turns = _stores()
    receipt = {"object": IN_FLIGHT_RECEIPT, "turn_id": "turn-delivery-r", "user_message_id": MSG}
    reply = asyncio.run(wait_for_receipt_reply(
        None, turns, receipt, bot_id="review-bot", user_id="review-user", timeout_seconds=1,
    ))
    assert reply == {"turn_id": "turn-delivery-r", "status": "DELIVERED", "response_text": "the answer"}
