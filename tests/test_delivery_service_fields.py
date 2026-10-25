"""TASK-1144: service-only delivery fields for task dispatch, and their fence."""

import asyncio
import inspect
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from llm_bawt import inter_bot_send as send
from llm_bawt.service.routes import inter_bot_deliveries as route
from llm_bawt.service.schemas import ChatCompletionRequest
from llm_bawt.task_turn_context import TASK_TURN_CONTEXT_HEADER

U = str(uuid.UUID(int=1144))
S = "11440000-0000-4000-8000-000000000001"


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
    bots = {"fixture-target": bot, "fixture-sender": bot}
    owned = {(S, "fixture-target", "fixture-user")}
    monkeypatch.setattr(route, "get_service", lambda: service)
    monkeypatch.setattr(send, "BotManager", lambda config: SimpleNamespace(get_bot=bots.get))
    monkeypatch.setattr(send, "submission_result", lambda *a, **kw: (202, {"success": True}))

    def require_owned(_sender, session_id, bot_id, user_id):
        if (session_id, bot_id, user_id) not in owned:
            raise HTTPException(status_code=404, detail={"error": "session_not_found"})

    monkeypatch.setattr(send.DeliverySender, "_require_owned_session", require_owned)
    return SimpleNamespace(dispatcher=dispatcher, bot=bot, bots=bots, owned=owned)


def _body(**kwargs):
    fields = {"sender_bot_id": "fixture-sender", "target_bot_id": "fixture-target",
              "message": "Please work on TASK-1", "prefer_steer": False,
              "author_entity_type": "user", "author_entity_id": "fixture-user", **kwargs}
    return route.DeliveryCreate(**fields)


def _request(headers=None):
    return SimpleNamespace(headers=headers or {})


def _send(body, request=None):
    return asyncio.run(route.create_delivery(body, request or _request()))


def _enqueued(env):
    return env.dispatcher.store.enqueue.call_args.kwargs


def test_dispatch_fields_reach_the_stored_request(delivery_route):
    _send(_body(user_message_id=U, session_id=S, augment_memory=False, raw_message=True))
    fields = _enqueued(delivery_route)
    assert fields["user_message_id"] == U
    assert fields["idempotency_key"] == U
    payload = fields["payload"]
    assert payload["session_id"] == S
    assert payload["augment_memory"] is False
    assert payload["messages"] == [{"role": "user", "content": "Please work on TASK-1"}]


def test_caller_key_wins_over_the_trigger_uuid(delivery_route):
    _send(_body(user_message_id=U, idempotency_key="caller-key"))
    assert _enqueued(delivery_route)["idempotency_key"] == "caller-key"


def test_defaults_are_unchanged_without_the_new_fields(delivery_route):
    _send(_body(sender_bot_id="fixture-sender", author_entity_type=None, author_entity_id=None,
                session_policy="continue"))
    fields = _enqueued(delivery_route)
    payload = fields["payload"]
    assert fields["user_message_id"] is None and fields["idempotency_key"] is None
    assert payload["augment_memory"] is True and "session_id" not in payload
    assert payload["messages"][0]["content"] == "Message from bot 'fixture-sender': Please work on TASK-1"


def test_raw_message_skips_the_bot_attribution_wrapper(delivery_route):
    _send(_body(author_entity_type=None, author_entity_id=None, raw_message=True,
                session_policy="continue"))
    assert _enqueued(delivery_route)["payload"]["messages"][0]["content"] == "Please work on TASK-1"


@pytest.mark.parametrize("field,value", [
    ("session_id", S), ("augment_memory", False), ("augment_memory", True), ("raw_message", False),
])
def test_bot_turn_capability_cannot_set_a_privileged_field(delivery_route, field, value):
    # Presence of the header is enough: an expired or forged capability is refused too.
    with pytest.raises(HTTPException) as error:
        _send(_body(**{field: value}), _request({TASK_TURN_CONTEXT_HEADER: "not-a-token"}))
    assert error.value.status_code == 403
    assert error.value.detail["error"] == "privileged_field"
    assert error.value.detail["fields"] == [field]
    delivery_route.dispatcher.store.enqueue.assert_not_called()


def test_bot_turn_capability_still_sends_ordinary_messages(delivery_route):
    _send(_body(author_entity_type=None, author_entity_id=None, session_policy="continue"),
          _request({TASK_TURN_CONTEXT_HEADER: "not-a-token"}))
    delivery_route.dispatcher.store.enqueue.assert_called_once()


def test_session_must_belong_to_the_target_and_user(delivery_route):
    other = "11440000-0000-4000-8000-000000000002"
    with pytest.raises(HTTPException) as error:
        _send(_body(session_id=other))
    assert error.value.status_code == 404
    delivery_route.dispatcher.store.enqueue.assert_not_called()


@pytest.mark.parametrize("status,owned", [("active", True), ("archived", True), ("deleted", False)])
def test_ownership_query_admits_archived_task_threads_but_not_deleted(monkeypatch, status, owned):
    """Born-archived task sessions are the dispatch target; deleted ones never are."""
    rows = [{"id": S, "bot_id": "fixture-target", "user_id": "fixture-user", "status": status}]

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, sql, params):
            query = str(sql)
            assert "status <> 'deleted'" in query
            match = [r for r in rows if r["id"] == params["id"] and r["bot_id"] == params["bot"]
                     and r["user_id"] == params["user"] and r["status"] != "deleted"]
            return SimpleNamespace(first=lambda: match[0] if match else None)

    engine = SimpleNamespace(connect=Conn)
    sender = send.DeliverySender(
        lambda: None, dispatcher=lambda: SimpleNamespace(store=SimpleNamespace(engine=engine)))
    if owned:
        sender._require_owned_session(S, "fixture-target", "fixture-user")
    else:
        with pytest.raises(HTTPException) as error:
            sender._require_owned_session(S, "fixture-target", "fixture-user")
        assert error.value.status_code == 404


@pytest.mark.parametrize("kwargs", [
    {"session_id": S, "prefer_steer": True},
    {"user_message_id": U, "prefer_steer": True},
    {"session_id": S, "session_policy": "reset_retain_history"},
    {"session_id": S, "reset_session_before_delivery": True},
    {"user_message_id": U, "session_policy": "reset_without_history"},
])
def test_pinned_fields_refuse_steering_and_resets(delivery_route, kwargs):
    with pytest.raises(HTTPException) as error:
        _send(_body(**kwargs))
    assert error.value.status_code == 422
    delivery_route.dispatcher.store.enqueue.assert_not_called()


def test_explicit_continue_is_allowed_with_a_pinned_session(delivery_route):
    _send(_body(session_id=S, session_policy="continue"))
    assert _enqueued(delivery_route)["payload"]["session_id"] == S


def test_dispatch_to_a_chat_bot_is_a_typed_error_not_a_failed_row(delivery_route):
    delivery_route.bot.agent_backend = None
    delivery_route.bot.harness = None
    with pytest.raises(HTTPException) as error:
        _send(_body(session_id=S))
    assert error.value.status_code == 422
    assert error.value.detail["error"] == "target_not_durable"
    delivery_route.dispatcher.store.enqueue.assert_not_called()


@pytest.mark.parametrize("pinned", [{"augment_memory": False}, {"user_message_id": U}])
def test_dispatch_to_an_unknown_bot_is_a_typed_error(delivery_route, pinned):
    with pytest.raises(HTTPException) as error:
        _send(_body(target_bot_id="nobody", **pinned))
    assert error.value.status_code == 404
    assert error.value.detail["error"] == "unknown_target"
    delivery_route.dispatcher.store.enqueue.assert_not_called()


def test_pinned_trigger_alone_to_a_chat_bot_does_not_burn_the_uuid(delivery_route):
    delivery_route.bot.agent_backend = None
    delivery_route.bot.harness = None
    with pytest.raises(HTTPException) as error:
        _send(_body(user_message_id=U))
    assert error.value.detail["error"] == "target_not_durable"
    delivery_route.dispatcher.store.enqueue.assert_not_called()


def test_pinned_trigger_skips_the_automatic_context_reset(delivery_route, monkeypatch):
    """Active-thread health must not rotate a thread or drop the caller's U."""
    health = Mock(side_effect=AssertionError("context health consulted"))
    monkeypatch.setattr(send, "AgentContextStore", lambda config: SimpleNamespace(health=health))
    _send(_body(user_message_id=U))
    _send(_body(session_id=S))
    assert delivery_route.dispatcher.store.enqueue.call_count == 2


def test_dispatched_request_carries_the_thread_and_trigger_its_capability_signs(
        delivery_route, monkeypatch):
    """The stored payload is what the dispatcher validates and runs. The turn's
    capability is minted from its session and trigger, which is how a worker's
    tasks_associate_current resolves the DISPATCH link BawtHub wrote for U."""
    from cryptography.fernet import Fernet

    from llm_bawt import request_identity, task_turn_context as codec

    fernet = Fernet(Fernet.generate_key())
    monkeypatch.setattr(codec, "_get_fernet", lambda: fernet)
    _send(_body(user_message_id=U, session_id=S))
    _, message_id, turn_id = request_identity.stable_ids(uuid.UUID(U))
    # admit() stamps the trigger onto the stored payload.
    payload = {**_enqueued(delivery_route)["payload"], "user_message_id": message_id}
    request = ChatCompletionRequest.model_validate({**payload, "stream": True})
    assert (request.session_id, request.user_message_id) == (S, U)
    token = codec.mint_for_agent_turn(
        backend="claude-code", session_id=request.session_id, turn_id=turn_id,
        trigger_message_id=request.user_message_id, bot_id="fixture-target", user_id="fixture-user",
    )
    opened = codec.open_task_turn_context(token)
    assert (opened.session_id, opened.trigger_message_id, opened.turn_id) == (S, U, turn_id)


# The primary fence: bots_send_message has no way to express these fields.

SERVICE_FIELDS = (*route.PRIVILEGED_FIELDS, "user_message_id")


def _tools():
    from llm_bawt.mcp_server import server  # registers tools; import first
    from llm_bawt.mcp_server import inter_bot_tools
    return server, inter_bot_tools


@pytest.mark.parametrize("name", SERVICE_FIELDS)
def test_bot_send_tool_schema_has_no_service_field(name):
    server, tools = _tools()
    for function in (tools.send_message_to_bot, tools._enqueue_durable):
        parameters = inspect.signature(function).parameters
        assert name not in parameters
        assert not any(p.kind is p.VAR_KEYWORD for p in parameters.values())
    schema = asyncio.run(server.mcp.list_tools())
    send = next(tool for tool in schema if tool.name == "bots_send_message")
    assert name not in send.inputSchema["properties"]


def test_extra_service_arguments_to_the_tool_are_never_forwarded(monkeypatch):
    server, tools = _tools()
    captured = {}

    async def enqueue(**kwargs):
        captured.update(kwargs)
        return {"delivery_id": "fixture-delivery"}

    async def nothing(*args, **kwargs):
        return None

    monkeypatch.setattr(server, "_enqueue_durable", enqueue)
    monkeypatch.setattr(tools, "_target_refusal", nothing)
    monkeypatch.setattr(tools, "_waited_reply_origin", nothing)
    from mcp.server.fastmcp import FastMCP

    # The FastMCP argument model, minus the approval interceptor (tested elsewhere).
    result = asyncio.run(FastMCP.call_tool(server.mcp, "bots_send_message", {
        "target_bot_id": "fixture-target", "message": "hi", "sender_bot_id": "fixture-sender",
        "session_id": S, "augment_memory": False, "raw_message": True, "user_message_id": U,
    }))
    assert captured.get("message") == "hi", result
    assert not set(captured) & set(SERVICE_FIELDS)


def test_bot_send_http_body_never_carries_a_service_field(monkeypatch):
    _, tools = _tools()
    sent = {}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json, headers=None, timeout=None):
            sent.update(json)
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"delivery_id": "d"})

    monkeypatch.setattr(tools.httpx, "AsyncClient", Client)
    asyncio.run(tools._enqueue_durable(
        prefer_steer=False, target_bot_id="t", message="m", sender_bot_id="s",
        max_tokens=None, temperature=0.7, timeout_seconds=10, idempotency_key=None,
        project_id=None, task_id=None, message_kind=None, metadata=None,
        session_policy=None, reset_session_before_delivery=None, retain_history=None,
        reset_reason=None,
    ))
    assert sent and not set(sent) & set(SERVICE_FIELDS)
