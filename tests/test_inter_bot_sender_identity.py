"""Inter-bot sends are attributed to the signed caller, never "unknown" (TASK-1241)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

import llm_bawt.mcp_server.server as server  # registers tools; import first
import llm_bawt.mcp_server.inter_bot_tools as tools  # noqa: E402
from llm_bawt.mcp_server import approval_interceptor, task_association
from llm_bawt.mcp_server.inter_bot_sender import (
    SenderIdentityError,
    resolve_sender_bot_id,
)
from llm_bawt import inter_bot_send as send
from llm_bawt.service.routes import inter_bot_deliveries as route
from llm_bawt.task_turn_context import TaskTurnContextError


def _signed(monkeypatch: pytest.MonkeyPatch, bot_id: str | None) -> None:
    def context():
        if bot_id is None:
            raise TaskTurnContextError("no capability")
        return SimpleNamespace(bot_id=bot_id, turn_id="turn-" + "c" * 32,
                               trigger_message_id="m")

    monkeypatch.setattr(task_association, "current_task_turn_context", context)


# ---- resolver --------------------------------------------------------------

@pytest.mark.parametrize("claimed", [None, "", "  ", "unknown", "AL", " al "])
def test_signed_caller_fills_or_confirms_sender(monkeypatch, claimed) -> None:
    _signed(monkeypatch, "al")
    assert resolve_sender_bot_id(claimed) == "al"


def test_signed_caller_rejects_spoofed_sender(monkeypatch) -> None:
    _signed(monkeypatch, "al")
    with pytest.raises(SenderIdentityError, match="does not match the signed caller"):
        resolve_sender_bot_id("gavel")


@pytest.mark.parametrize("claimed", [None, "", "unknown", " Unknown "])
def test_unsigned_caller_must_name_itself(monkeypatch, claimed) -> None:
    _signed(monkeypatch, None)
    with pytest.raises(SenderIdentityError, match="sender_bot_id is required"):
        resolve_sender_bot_id(claimed)


def test_unsigned_caller_with_explicit_sender_is_accepted(monkeypatch) -> None:
    _signed(monkeypatch, None)
    assert resolve_sender_bot_id(" Codex ") == "codex"


def test_approved_replay_uses_original_caller(monkeypatch) -> None:
    _signed(monkeypatch, None)
    approved = approval_interceptor.ApprovedCallerContext(bot_id="byte")
    token = approval_interceptor._approved_caller_context.set(approved)
    try:
        assert resolve_sender_bot_id(None) == "byte"
        with pytest.raises(SenderIdentityError):
            resolve_sender_bot_id("snark")
    finally:
        approval_interceptor._approved_caller_context.reset(token)


# ---- tools -----------------------------------------------------------------

async def _no_refusal(_target_bot_id: str) -> None:
    """Every fixture target is active; tag refusals are covered in test_bot_tag_discovery."""
    return None


def _capture_enqueue(monkeypatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def enqueue(**kwargs):
        calls.append(kwargs)
        return {"delivery_id": "delivery-1", "status": "QUEUED"}

    monkeypatch.setattr(server, "_enqueue_durable", enqueue)
    monkeypatch.setattr(tools, "_target_refusal", _no_refusal)
    return calls


def test_send_without_sender_is_stamped_with_signed_bot(monkeypatch) -> None:
    _signed(monkeypatch, "al")
    calls = _capture_enqueue(monkeypatch)
    result = asyncio.run(server.send_message_to_bot(target_bot_id="snark", message="hi"))
    assert result["dispatched"] is True
    assert calls[0]["sender_bot_id"] == "al"


def test_send_with_spoofed_sender_has_no_side_effect(monkeypatch) -> None:
    _signed(monkeypatch, "al")
    calls = _capture_enqueue(monkeypatch)
    result = asyncio.run(server.send_message_to_bot(
        target_bot_id="snark", message="hi", sender_bot_id="gavel",
    ))
    assert result["dispatched"] is False and "signed caller" in result["error"]
    assert calls == []


def test_unattributed_send_without_context_is_refused(monkeypatch) -> None:
    _signed(monkeypatch, None)
    calls = _capture_enqueue(monkeypatch)
    result = asyncio.run(server.send_message_to_bot(target_bot_id="snark", message="hi"))
    assert result["dispatched"] is False and "sender_bot_id is required" in result["error"]
    assert calls == []


def test_compact_without_sender_uses_signed_bot(monkeypatch) -> None:
    _signed(monkeypatch, "snark")
    monkeypatch.setattr(tools, "agent_context_health", AsyncMock(return_value={
        "capabilities": {"compact": True},
    }))
    enqueue = AsyncMock(return_value={"delivery_id": "delivery-1"})
    monkeypatch.setattr(tools, "_enqueue_durable", enqueue)
    asyncio.run(tools.agent_context_compact("snark", "compact-1"))
    assert enqueue.call_args.kwargs["sender_bot_id"] == "snark"


def test_compact_without_any_identity_is_refused(monkeypatch) -> None:
    _signed(monkeypatch, None)
    monkeypatch.setattr(tools, "agent_context_health", AsyncMock(return_value={
        "capabilities": {"compact": True},
    }))
    enqueue = AsyncMock()
    monkeypatch.setattr(tools, "_enqueue_durable", enqueue)
    result = asyncio.run(tools.agent_context_compact("snark", "compact-1"))
    assert result["success"] is False
    enqueue.assert_not_called()


# ---- route -----------------------------------------------------------------

@pytest.fixture
def delivery_route(monkeypatch):
    record = SimpleNamespace(id="fixture-delivery")
    store = Mock(engine=True)
    store.enqueue.return_value = (record, False)
    store.get.return_value = record
    store.payload.return_value = {}
    dispatcher = SimpleNamespace(store=store, _emit=AsyncMock(), wake=Mock())
    service = SimpleNamespace(config=SimpleNamespace(DEFAULT_USER="nick"),
                              _inter_bot_dispatcher=dispatcher)
    bot = SimpleNamespace(agent_backend="claude-code", harness="claude-code")
    known = {"al", "snark"}
    monkeypatch.setattr(route, "get_service", lambda: service)
    monkeypatch.setattr(send, "BotManager", lambda config: SimpleNamespace(
        get_bot=lambda slug: bot if slug.strip().lower() in known else None,
    ))
    monkeypatch.setattr(send, "submission_result", lambda *a, **kw: (202, {"success": True}))
    return store


def _create(**kwargs):
    body = route.DeliveryCreate(target_bot_id="snark", message="hello",
                                session_policy="continue", **kwargs)
    return asyncio.run(route.create_delivery(body))


@pytest.mark.parametrize("fields", [
    {},
    {"sender_bot_id": "unknown"},
    {"sender_bot_id": "ghost"},
    {"sender_bot_id": "al", "author_entity_type": "bot", "author_entity_id": "unknown"},
])
def test_route_rejects_bot_author_that_is_not_a_real_bot(delivery_route, fields) -> None:
    with pytest.raises(HTTPException, match="real sender bot") as error:
        _create(**fields)
    assert error.value.status_code == 422
    delivery_route.enqueue.assert_not_called()


def test_route_accepts_real_bot_sender(delivery_route) -> None:
    _create(sender_bot_id="al")
    assert delivery_route.enqueue.call_args.kwargs["sender_bot_id"] == "al"


def test_route_keeps_user_authored_workspace_delivery(delivery_route) -> None:
    """BawtHub's repos workspace posts author=nick with sender unknown."""
    _create(sender_bot_id="unknown", author_entity_type="user", author_entity_id="nick")
    kwargs = delivery_route.enqueue.call_args.kwargs
    assert kwargs["sender_bot_id"] == "unknown"
    assert kwargs["author"].entity_type == "user"
