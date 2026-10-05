"""DB-backed cross-bot synchronous wait regression coverage (TASK-618)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import llm_bawt.mcp_server.server as server


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _stub_idle_and_dispatch(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Waited sends are durable enqueue + bounded wait; record the wait budget."""
    observed: list[float] = []

    async def fake_check(_target_bot_id: str) -> None:
        return None

    async def fake_find(*_args) -> None:
        return None

    async def fake_enqueue(**_kwargs) -> dict:
        return {"delivery_id": "delivery-1", "status": "QUEUED"}

    async def fake_await(receipt: dict, wait_seconds: float) -> dict:
        observed.append(wait_seconds)
        return {**receipt, "status": "DELIVERED", "success": True, "content": "done"}

    monkeypatch.setattr(server, "_check_bot_in_turn", fake_check)
    monkeypatch.setattr(server, "_find_delivery_by_key", fake_find)
    monkeypatch.setattr(server, "_enqueue_durable", fake_enqueue)
    monkeypatch.setattr(server, "_await_delivery", fake_await)
    return observed


def test_setting_is_db_backed_with_five_minute_default() -> None:
    from llm_bawt.setting_definitions import (
        SETTING_DEFINITIONS,
        STORAGE_RUNTIME_SETTING,
    )

    definition = SETTING_DEFINITIONS["bot_send_wait_seconds"]
    assert definition.default == 300
    assert definition.storage == STORAGE_RUNTIME_SETTING
    assert definition.applies_to == ()


def test_default_synchronous_wait_is_five_minutes(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = _stub_idle_and_dispatch(monkeypatch)
    monkeypatch.setattr(server, "_bot_send_wait_ceiling_seconds", lambda: 300.0)

    result = _run(server.send_message_to_bot(
        target_bot_id="gavel",
        message="audit this",
        sender_bot_id="snark",
        wait_for_reply=True,
    ))

    assert result["success"] is True
    assert observed == [300.0]
    assert "timeout_clamped" not in result


def test_db_ceiling_clamps_wait_with_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = _stub_idle_and_dispatch(monkeypatch)
    monkeypatch.setattr(server, "_bot_send_wait_ceiling_seconds", lambda: 240.0)

    result = _run(server.send_message_to_bot(
        target_bot_id="gavel",
        message="audit this",
        sender_bot_id="snark",
        wait_for_reply=True,
        timeout_seconds=900.0,
    ))

    assert result["success"] is True
    assert observed == [240.0]
    assert result["timeout_clamped"] is True
    assert result["requested_timeout_seconds"] == 900.0
    assert result["effective_timeout_seconds"] == 240.0


def test_redis_command_carries_db_derived_claude_timeout() -> None:
    from agent_bridge.subscriber import RedisSubscriber

    subscriber = RedisSubscriber("redis://localhost:6379/0")
    subscriber._pub_redis = MagicMock()
    subscriber._pub_redis.eval = None
    captured: dict[str, Any] = {}

    async def fake_xadd(_stream, fields, **_kwargs):
        captured.update(fields)
        return b"1"

    subscriber._pub_redis.xadd = fake_xadd  # type: ignore[method-assign]
    _run(subscriber.send_command(
        session_key="snark:nick",
        message="hello",
        request_id="req-1",
        mcp_tool_timeout_ms=330_000,
    ))

    assert captured["mcp_tool_timeout_ms"] == "330000"


def test_claude_request_parses_mcp_timeout() -> None:
    from claude_code_bridge.send_request import SendRequest

    request = SendRequest.from_fields({
        "request_id": "req-1",
        "session_key": "snark:nick",
        "bot_id": "snark",
        "message": "hello",
        "model": "openai_chatgpt/gpt-5.6-sol",
        "mcp_tool_timeout_ms": "330000",
    })

    assert request.mcp_tool_timeout_ms == 330_000


def test_claude_request_normalizes_catalog_transport_policy() -> None:
    from claude_code_bridge.send_request import SendRequest

    fields = {
        "request_id": "req-1",
        "session_key": "snark:nick",
        "bot_id": "snark",
        "message": "hello",
        "model": "openai_chatgpt/gpt-6-astra",
    }

    assert SendRequest.from_fields({
        **fields, "responses_transport": " SSE ",
    }).responses_transport == "sse"
    assert SendRequest.from_fields({
        **fields, "responses_transport": "unknown",
    }).responses_transport is None


def test_claude_timeout_is_not_configured_by_environment() -> None:
    compose = Path(__file__).parents[1].joinpath("docker-compose.yml").read_text()
    assert "MCP_TOOL_TIMEOUT" not in compose
