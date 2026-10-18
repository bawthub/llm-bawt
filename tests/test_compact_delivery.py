"""Exercise MCP -> durable HTTP payload -> bridge policy without live sessions."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from llm_bawt.chat_prompt_defaults import COMPACT_SUMMARY_DEFAULT
from claude_code_bridge.send_preflight import prepare_send
from llm_bawt.service.routes import inter_bot_deliveries as route


@pytest.fixture
def delivery_route(monkeypatch):
    record = SimpleNamespace(id="fixture-delivery")
    store = Mock(engine=True)
    store.enqueue.return_value = (record, False)
    store.get.return_value = record
    store.payload.return_value = {}
    dispatcher = SimpleNamespace(store=store, _emit=AsyncMock(), wake=Mock())
    service = SimpleNamespace(config=SimpleNamespace(DEFAULT_USER="fixture-user"),
                              _inter_bot_dispatcher=dispatcher)
    bot = SimpleNamespace(agent_backend="claude-code", harness="claude-code")
    monkeypatch.setattr(route, "get_service", lambda: service)
    monkeypatch.setattr(route, "BotManager", lambda config: SimpleNamespace(get_bot=lambda slug: bot))
    monkeypatch.setattr(route, "submission_result", lambda *a, **kw: (202, {"success": True}))
    return dispatcher, bot


def _body(message="/compact", **kwargs):
    return route.DeliveryCreate(
        sender_bot_id="fixture-sender", target_bot_id="fixture-target", message=message,
        session_policy="continue", **kwargs,
    )


@pytest.mark.parametrize("message", ["/compact", "/compact\n保留 details\n  ", "/compact Write 15000 tokens"])
def test_tagged_maintenance_preserves_entire_command(delivery_route, message):
    dispatcher, _ = delivery_route
    asyncio.run(route.create_delivery(_body(
        message, prefer_steer=False, message_kind="CONTEXT_MAINTENANCE",
        metadata={"context_action": "compact"},
    )))
    fields = dispatcher.store.enqueue.call_args.kwargs
    assert fields["payload"]["messages"][0]["content"] == message
    assert fields["message"] == message
    assert fields["author"].entity_id == "fixture-sender"
    assert fields["payload"]["prefer_steer"] is False
    assert fields["payload"]["max_tokens"] is None


@pytest.mark.parametrize("message", ["/compact", "Please /compact", "/compact details"])
def test_ordinary_callbacks_still_keep_authorship_wrapper(delivery_route, message):
    dispatcher, _ = delivery_route
    asyncio.run(route.create_delivery(_body(message)))
    fields = dispatcher.store.enqueue.call_args.kwargs
    assert fields["payload"]["messages"][0]["content"] == f"Message from bot 'fixture-sender': {message}"


@pytest.mark.parametrize("kind,metadata,message", [
    ("CONTEXT_MAINTENANCE", {"context_action": "health"}, "inspect context"),
    ("CONTEXT_MAINTENANCE", {}, "/compact"),
    ("READY", {"context_action": "compact"}, "/compact"),
    (None, {"context_action": "compact"}, "ordinary callback"),
])
def test_non_compact_maintenance_and_partial_tags_keep_wrapper(delivery_route, kind, metadata, message):
    dispatcher, _ = delivery_route
    asyncio.run(route.create_delivery(_body(
        message, prefer_steer=False, message_kind=kind, metadata=metadata,
    )))
    fields = dispatcher.store.enqueue.call_args.kwargs
    assert fields["payload"]["messages"][0]["content"] == f"Message from bot 'fixture-sender': {message}"


@pytest.mark.parametrize("message,backend,steer", [
    ("mention /compact", "claude-code", False),
    ("/compactness", "claude-code", False),
    ("/compact", "codex", False),
    ("/compact", "claude-code", True),
])
def test_invalid_maintenance_rejected_before_enqueue(delivery_route, message, backend, steer):
    dispatcher, bot = delivery_route
    bot.agent_backend = backend
    with pytest.raises(HTTPException) as error:
        asyncio.run(route.create_delivery(_body(
            message, prefer_steer=steer, message_kind="CONTEXT_MAINTENANCE",
            metadata={"context_action": "compact"},
        )))
    assert error.value.status_code == 422
    dispatcher.store.enqueue.assert_not_called()


@pytest.mark.parametrize("default_user", [None, "", "   "])
def test_bot_delivery_without_owner_rejected_before_enqueue(delivery_route, monkeypatch, default_user):
    dispatcher, _bot = delivery_route
    route.get_service().config.DEFAULT_USER = default_user
    with pytest.raises(HTTPException, match="Delivery user is required") as error:
        asyncio.run(route.create_delivery(_body("hello")))
    assert error.value.status_code == 422
    dispatcher.store.enqueue.assert_not_called()


def test_user_authored_delivery_uses_explicit_owner_without_default(delivery_route):
    dispatcher, _bot = delivery_route
    route.get_service().config.DEFAULT_USER = None
    asyncio.run(route.create_delivery(_body(
        "hello", author_entity_type="user", author_entity_id="  Casey  ",
    )))
    assert dispatcher.store.enqueue.call_args.kwargs["payload"]["user"] == "casey"


def test_bot_authored_delivery_uses_configured_default(delivery_route):
    dispatcher, _bot = delivery_route
    route.get_service().config.DEFAULT_USER = "casey"
    asyncio.run(route.create_delivery(_body("hello")))
    assert dispatcher.store.enqueue.call_args.kwargs["payload"]["user"] == "casey"


def test_preventive_health_uses_explicit_owner_not_empty_default(delivery_route, monkeypatch):
    from llm_bawt import runtime_settings

    dispatcher, _bot = delivery_route
    route.get_service().config.DEFAULT_USER = None
    health = Mock(return_value={"state": "healthy"})
    monkeypatch.setattr(route, "AgentContextStore", lambda config: SimpleNamespace(health=health))
    monkeypatch.setattr(route, "_target_context_window", lambda config, bot: None)
    monkeypatch.setattr(runtime_settings, "RuntimeSettingsResolver", lambda *a, **kw: SimpleNamespace(
        resolve=lambda key, fallback: fallback,
    ))

    asyncio.run(route.create_delivery(route.DeliveryCreate(
        target_bot_id="fixture-target", message="hello", sender_bot_id="fixture-sender",
        author_entity_type="user", author_entity_id="  Casey  ",
    )))
    assert health.call_args.kwargs["user_id"] == "casey"
    assert dispatcher.store.enqueue.call_args.kwargs["payload"]["user"] == "casey"


def test_mcp_through_real_route_and_preflight(delivery_route, monkeypatch):
    from llm_bawt.mcp_server import server  # noqa: F401 — initialize registration first
    from llm_bawt.mcp_server import inter_bot_tools

    dispatcher, _ = delivery_route
    monkeypatch.setattr(inter_bot_tools, "agent_context_health", AsyncMock(return_value={
        "capabilities": {"compact": True},
    }))

    async def enqueue(**kwargs):
        return await route.create_delivery(route.DeliveryCreate(**kwargs))

    monkeypatch.setattr(inter_bot_tools, "_enqueue_durable", enqueue)

    monkeypatch.setattr("claude_code_bridge.send_preflight.fetch_compact_prompt",
                        lambda url: COMPACT_SUMMARY_DEFAULT)

    async def run():
        await inter_bot_tools.agent_context_compact("fixture-target", "fixture-key", "fixture-sender")
        enqueued = dispatcher.store.enqueue.call_args.kwargs
        message = enqueued["payload"]["messages"][0]["content"]
        bridge = SimpleNamespace(
            _trigger_message_ids={},
            _preprocess_new_command=AsyncMock(side_effect=lambda text, **kw: text),
        )
        prepared = await prepare_send(bridge, {
            "message": message, "model": "fixture-model", "request_id": "fixture-request",
        }, "fixture-redis", SimpleNamespace(xack=AsyncMock()))
        assert prepared.message == f"/compact {COMPACT_SUMMARY_DEFAULT}"
        assert enqueued["idempotency_key"] == "fixture-key"

    asyncio.run(run())
