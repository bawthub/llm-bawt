"""TASK-935: user escalation is scoped to its owning active turn."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from llm_bawt.service.routes import chat


@pytest.mark.anyio
async def test_escalate_uses_same_message_id_without_resending_text(monkeypatch):
    turn = SimpleNamespace(
        id="turn-1", bot_id="caid", user_id="nick", status="streaming",
        ended_at=None, agent_session_key="caid:nick", agent_request_id="req-1",
    )
    service = SimpleNamespace(config=object(), _turn_log_store=SimpleNamespace(get_turn=lambda _id: turn))
    subscriber = SimpleNamespace(send_steer=AsyncMock(return_value={"ok": True, "detail": "escalated"}))
    monkeypatch.setattr(chat, "get_service", lambda: service)
    monkeypatch.setattr("llm_bawt.bots.BotManager", lambda config: SimpleNamespace(
        get_bot=lambda bot: SimpleNamespace(agent_backend="claude-code")))
    monkeypatch.setattr("llm_bawt.agent_backends.agent_bridge.get_agent_subscriber", lambda: subscriber)
    request = chat.ChatSteerEscalateRequest(
        turn_id="turn-1", message_id="message-1", bot_id="caid", user_id="nick",
    )
    assert (await chat.chat_steer_escalate(request))["ok"] is True
    assert (await chat.chat_steer_escalate(request))["ok"] is True
    subscriber.send_steer.assert_awaited_with(
        session_key="caid:nick", message="", message_id="message-1",
        backend="claude-code", target_request_id="req-1",
        request_id="escalate_turn-1_message-1", origin="user", escalate=True,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("change", [{"status": "ok"}, {"bot_id": "snark"}, {"user_id": "other"}])
async def test_escalate_rejects_inactive_or_mismatched_turn_before_bridge(monkeypatch, change):
    turn = SimpleNamespace(id="turn-1", bot_id="caid", user_id="nick", status="streaming",
                           ended_at=None, agent_session_key="caid:nick", agent_request_id="req-1")
    for key, value in change.items():
        setattr(turn, key, value)
    monkeypatch.setattr(chat, "get_service", lambda: SimpleNamespace(
        _turn_log_store=SimpleNamespace(get_turn=lambda _id: turn)))
    with pytest.raises(HTTPException) as caught:
        await chat.chat_steer_escalate(chat.ChatSteerEscalateRequest(
            turn_id="turn-1", message_id="message-1", bot_id="caid", user_id="nick"))
    assert caught.value.status_code == 409


@pytest.mark.anyio
async def test_cancel_targets_original_turn_and_message_without_resending_text(monkeypatch):
    turn = SimpleNamespace(
        id="turn-1", bot_id="caid", user_id="nick", status="streaming",
        ended_at=None, agent_session_key="caid:nick", agent_request_id="req-1",
    )
    service = SimpleNamespace(config=object(), _turn_log_store=SimpleNamespace(get_turn=lambda _id: turn))
    subscriber = SimpleNamespace(send_steer=AsyncMock(return_value={"ok": True, "detail": "cancelled"}))
    monkeypatch.setattr(chat, "get_service", lambda: service)
    monkeypatch.setattr("llm_bawt.bots.BotManager", lambda config: SimpleNamespace(
        get_bot=lambda bot: SimpleNamespace(agent_backend="claude-code")))
    monkeypatch.setattr("llm_bawt.agent_backends.agent_bridge.get_agent_subscriber", lambda: subscriber)
    request = chat.ChatSteerEscalateRequest(
        turn_id="turn-1", message_id="message-1", bot_id="caid", user_id="nick",
    )
    assert (await chat.chat_steer_cancel(request))["detail"] == "cancelled"
    subscriber.send_steer.assert_awaited_once_with(
        session_key="caid:nick", message="", message_id="message-1",
        backend="claude-code", target_request_id="req-1",
        request_id="cancel_turn-1_message-1", origin="user", cancel=True,
    )


@pytest.mark.anyio
async def test_cancel_rejects_other_owner_before_bridge(monkeypatch):
    turn = SimpleNamespace(id="turn-1", bot_id="caid", user_id="other", status="streaming",
                           ended_at=None, agent_session_key="caid:other", agent_request_id="req-1")
    monkeypatch.setattr(chat, "get_service", lambda: SimpleNamespace(
        _turn_log_store=SimpleNamespace(get_turn=lambda _id: turn)))
    with pytest.raises(HTTPException) as caught:
        await chat.chat_steer_cancel(chat.ChatSteerEscalateRequest(
            turn_id="turn-1", message_id="message-1", bot_id="caid", user_id="nick"))
    assert caught.value.status_code == 409


def test_user_steer_defaults_to_next():
    assert chat.ChatSteerRequest(
        turn_id="turn-1", bot_id="caid", message_id="message-1", message="hello"
    ).priority == "next"
