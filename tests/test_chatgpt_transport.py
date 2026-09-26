"""Hermetic Codex-compatible WS/Lite protocol and lifecycle regressions."""
import asyncio
import copy
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
import pytest
from websockets.exceptions import InvalidStatus

from claude_code_bridge.proxy.chatgpt_transport import (
    ChatGPTEventTimeout, ChatGPTResponsesTransport, LITE_HEADER, TURN_HEADER,
    lite_request,
)
from claude_code_bridge.proxy.stream import (
    IncompleteResponseStreamError, responses_to_anthropic_sse,
)

BODY = {"model": "gpt-6-astra", "instructions": "Be helpful", "input": [
    {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
], "tools": [{"type": "function", "name": "check", "parameters": {"type": "object"}}],
    "stream": True, "store": False, "reasoning": {"effort": "low"}}
DONE = {"type": "response.completed", "response": {"usage": None, "status": "completed"}}


class Socket:
    def __init__(self):
        self.response = NS(headers={TURN_HEADER: "sticky"})
        self.sent = []
        self.events = asyncio.Queue()
        self.close = AsyncMock()

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    async def recv(self):
        return json.dumps(await self.events.get())


def kwargs(**changes):
    values = dict(body=BODY, headers={"session_id": "conversation", "chatgpt-account-id": "account"},
                  bearer="token", base_url="https://example.test/backend-api/codex",
                  context=NS(conversation_id="conversation", request_id="turn"), http_client=AsyncMock())
    values.update(changes)
    return values


def test_lite_payload_is_stable_scoped_and_does_not_mutate():
    from claude_code_bridge.proxy.adapters.openai_chatgpt import OpenAIChatGPTAdapter

    prepared = OpenAIChatGPTAdapter().prepare_request(copy.deepcopy(BODY))
    original = copy.deepcopy(prepared)
    a = lite_request(prepared, "one")
    assert prepared == original
    assert a == lite_request(prepared, "one")
    assert a["input"][0]["id"] != lite_request(prepared, "two")["input"][0]["id"]
    assert "tools" not in a and "instructions" not in a
    assert a["input"][0]["type"] == "additional_tools"
    assert a["input"][1]["role"] == "developer"
    assert a["reasoning"]["context"] == "all_turns"
    assert a["parallel_tool_calls"] is False


def test_lite_strips_image_details_in_messages_and_tool_results():
    body = {**BODY, "input": [{"role": "user", "content": [
        {"type": "input_image", "image_url": "data:abc", "detail": "high"}]},
        {"type": "function_call_output", "call_id": "c", "output": [
            {"type": "input_image", "image_url": "data:xyz", "detail": "auto"}]}]}
    result = lite_request(body, "one")["input"]
    assert "detail" not in result[-2]["content"][0]
    assert "detail" not in result[-1]["output"][0]


def test_ws_success_reuse_and_turn_isolation():
    async def run():
        sockets = []
        async def connect(*args, **kw):
            assert args[0] == "wss://example.test/backend-api/codex/responses"
            assert kw["additional_headers"][LITE_HEADER] == "true"
            socket = Socket()
            sockets.append(socket)
            return socket
        client = ChatGPTResponsesTransport(connector=connect)
        first = await client.open(**kwargs())
        socket = sockets[0]
        await socket.events.put(DONE)
        assert (await anext(first)).type == "response.completed"
        await first.close()
        second = await client.open(**kwargs())
        assert len(sockets) == 1
        assert socket.sent[1]["type"] == "response.create"
        assert "previous_response_id" not in socket.sent[1]
        assert socket.sent[1]["client_metadata"]["x-codex-turn-state"] == "sticky"
        await socket.events.put(DONE)
        await anext(second)
        await second.close()
        third = await client.open(**kwargs(context=NS(conversation_id="conversation", request_id="other")))
        assert len(sockets) == 2
        await third.close()
        await client.close()
        assert all(s.close.await_count == 1 for s in sockets)
    asyncio.run(run())


def test_parallel_calls_never_share_socket_and_cancel_discards():
    async def run():
        connector = AsyncMock(side_effect=[Socket(), Socket(), Socket()])
        client = ChatGPTResponsesTransport(connector=connector)
        a = await client.open(**kwargs())
        b = await client.open(**kwargs())
        assert a.session is not b.session
        task = asyncio.create_task(anext(a))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await a.close()
        assert a.session.socket is None
        c = await client.open(**kwargs())
        assert connector.await_count == 3
        await b.close()
        await c.close()
        await client.close()
    asyncio.run(run())


def test_first_event_timeout_then_idle_timeout_and_keepalive_expiry():
    async def run():
        socket = Socket()
        client = ChatGPTResponsesTransport(
            connector=AsyncMock(return_value=socket),
            first_event_timeout=.01, idle_timeout=.08, keepalive=.01,
        )
        stream = await client.open(**kwargs())
        started = asyncio.get_running_loop().time()
        with pytest.raises(ChatGPTEventTimeout, match="first"):
            await anext(stream)
        assert asyncio.get_running_loop().time() - started < .06
        await stream.close()
        assert socket.close.await_count == 1
        await client.close()

        socket = Socket()
        client = ChatGPTResponsesTransport(
            connector=AsyncMock(return_value=socket),
            first_event_timeout=.01, idle_timeout=.08, keepalive=.01,
        )
        stream = await client.open(**kwargs())
        await socket.events.put({"type": "response.created"})
        assert (await anext(stream)).type == "response.created"
        started = asyncio.get_running_loop().time()
        with pytest.raises(ChatGPTEventTimeout, match="next"):
            await anext(stream)
        assert asyncio.get_running_loop().time() - started >= .06
        await stream.close()
        assert socket.close.await_count == 1
        await client.close()

        socket = Socket()
        client = ChatGPTResponsesTransport(
            connector=AsyncMock(return_value=socket),
            first_event_timeout=.01, idle_timeout=.08, keepalive=.01,
        )
        stream = await client.open(**kwargs())
        await socket.events.put(DONE)
        await anext(stream)
        await stream.close()
        await asyncio.sleep(.03)
        assert not client._idle
        assert socket.close.await_count == 1
        await client.close()
    asyncio.run(run())


@pytest.mark.parametrize(
    "kind",
    ["response.created", "response.in_progress", "response.metadata", "ping", "empty_delta"],
)
def test_nonproductive_chatter_hits_productive_progress_deadline(kind):
    async def run():
        socket = Socket()
        client = ChatGPTResponsesTransport(
            connector=AsyncMock(return_value=socket),
            first_event_timeout=.1,
            idle_timeout=.2,
            productive_idle_timeout=.025,
            attempt_timeout=.2,
        )
        stream = await client.open(**kwargs())

        async def chatter():
            while True:
                payload = {
                    "type": "response.output_text.delta" if kind == "empty_delta" else kind,
                }
                if kind == "empty_delta":
                    payload["delta"] = ""
                if kind == "response.metadata":
                    payload["headers"] = {"x-test": "noise"}
                await socket.events.put(payload)
                await asyncio.sleep(.002)

        producer = asyncio.create_task(chatter())
        try:
            with pytest.raises(ChatGPTEventTimeout) as error:
                while True:
                    await anext(stream)
            assert error.value.phase == "productive"
            assert error.value.transport == "websocket"
            assert error.value.fallback_transport == "sse"
            assert error.value.productive_idle_seconds >= .02
            assert stream.session.http_only is True
        finally:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
            await stream.close()
            await client.close()
        assert socket.close.await_count == 1

    asyncio.run(run())


def test_productive_stream_still_hits_absolute_attempt_deadline():
    async def run():
        socket = Socket()
        client = ChatGPTResponsesTransport(
            connector=AsyncMock(return_value=socket),
            first_event_timeout=.1,
            idle_timeout=.2,
            productive_idle_timeout=.2,
            attempt_timeout=.025,
        )
        stream = await client.open(**kwargs())

        async def chatter():
            while True:
                await socket.events.put({
                    "type": "response.output_text.delta",
                    "delta": "x",
                })
                await asyncio.sleep(.002)

        producer = asyncio.create_task(chatter())
        try:
            with pytest.raises(ChatGPTEventTimeout) as error:
                while True:
                    await anext(stream)
            assert error.value.phase == "absolute"
            assert error.value.elapsed_seconds >= .02
            assert error.value.productive_idle_seconds < .02
        finally:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
            await stream.close()
            await client.close()

    asyncio.run(run())


def test_failed_fallback_terminal_does_not_report_recovered():
    async def run():
        statuses = []
        context = NS(
            conversation_id="test",
            request_id="turn",
            attempt=2,
            report_status=lambda status: statuses.append(status),
        )
        socket = Socket()
        client = ChatGPTResponsesTransport(connector=AsyncMock(return_value=socket))
        stream = await client.open(**{**kwargs(), "context": context})
        await socket.events.put({
            "type": "response.failed",
            "response": {"error": {"code": "server_error", "message": "failed"}},
        })
        assert (await anext(stream)).type == "response.failed"
        assert statuses == []
        await stream.close()
        await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_upgrade_errors_do_not_fallback(status):
    async def run():
        connector = AsyncMock(side_effect=InvalidStatus(NS(status_code=status, headers={})))
        client = ChatGPTResponsesTransport(connector=connector)
        args = kwargs()
        with pytest.raises(httpx.HTTPStatusError) as error:
            await client.open(**args)
        assert error.value.response.status_code == status
        args["http_client"].post.assert_not_called()
        assert not client._active
        await client.close()
    asyncio.run(run())


def test_426_falls_back_with_lite_body_and_sticky_http_headers():
    async def run():
        connector = AsyncMock(side_effect=InvalidStatus(NS(status_code=426, headers={})))
        client = ChatGPTResponsesTransport(connector=connector)
        args = kwargs()
        async def events():
            yield NS(type="response.completed")
        http_stream = NS(response=NS(headers={TURN_HEADER: "http-sticky"}), close=AsyncMock())
        class Stream:
            response = http_stream.response
            close = http_stream.close
            def __aiter__(self):
                return events()
        args["http_client"].post.return_value = Stream()
        first = await client.open(**args)
        await anext(first)
        await first.close()
        second = await client.open(**args)
        assert connector.await_count == 1
        sent = args["http_client"].post.call_args.kwargs
        assert "instructions" not in sent["body"] and "tools" not in sent["body"]
        assert sent["options"]["headers"][TURN_HEADER] == "http-sticky"
        await second.close()
        await client.close()
    asyncio.run(run())


@pytest.mark.parametrize("state", [None, "provided"])
def test_missing_terminal_is_never_success(state):
    from claude_code_bridge.proxy.stream import TranslatorState
    async def run():
        async def events():
            if False:
                yield None
        if state:
            with pytest.raises(IncompleteResponseStreamError):
                _ = [chunk async for chunk in responses_to_anthropic_sse(
                    events(), anthropic_model="astra", state=TranslatorState())]
        else:
            chunks = [chunk async for chunk in responses_to_anthropic_sse(events(), anthropic_model="astra")]
            assert b'event: error\n' in chunks[-1]
            assert not any(b"message_stop" in chunk for chunk in chunks)
    asyncio.run(run())


def test_metadata_token_survives_failed_socket_for_safe_retry():
    async def run():
        sockets = []
        calls = []
        async def connect(*args, **kw):
            calls.append(kw)
            socket = Socket()
            socket.response.headers = {}
            sockets.append(socket)
            return socket
        client = ChatGPTResponsesTransport(connector=connect)
        first = await client.open(**kwargs())
        await sockets[0].events.put({"type": "response.metadata", "headers": {"X-Codex-Turn-State": "from-event"}})
        await anext(first)
        await first.close()
        second = await client.open(**kwargs())
        assert calls[1]["additional_headers"][TURN_HEADER] == "from-event"
        assert sockets[0].close.await_count == 1
        await second.close()
        await client.close()
    asyncio.run(run())


@pytest.mark.parametrize("field", ["account", "bearer", "conversation", "model", "anonymous"])
def test_pool_isolates_identity_dimensions(field):
    async def run():
        connector = AsyncMock(side_effect=[Socket(), Socket()])
        client = ChatGPTResponsesTransport(connector=connector)
        first = await client.open(**kwargs())
        await first.session.socket.events.put(DONE)
        await anext(first)
        await first.close()
        args = kwargs()
        if field == "account":
            args["headers"] = {**args["headers"], "chatgpt-account-id": "other"}
        elif field == "bearer":
            args["bearer"] = "rotated"
        elif field == "model":
            args["body"] = {**BODY, "model": "different"}
        elif field == "conversation":
            args["context"] = NS(conversation_id="different", request_id="turn")
        else:
            args["context"] = None
        second = await client.open(**args)
        assert connector.await_count == 2
        await second.close()
        await client.close()
    asyncio.run(run())


def test_adapter_transport_policy_is_catalog_driven(monkeypatch):
    from claude_code_bridge.proxy.adapters.openai_chatgpt import OpenAIChatGPTAdapter
    from claude_code_bridge.proxy.request_context import ProxyRequestContext

    async def run():
        adapter = OpenAIChatGPTAdapter()
        ordinary = AsyncMock(return_value="ordinary-sse")
        lite = AsyncMock(return_value="lite-ws")
        monkeypatch.setattr(
            "claude_code_bridge.proxy.adapters.base.ProviderAdapter.open_stream",
            ordinary,
        )
        adapter._chatgpt_transport = NS(open=lite, close=AsyncMock())
        common = dict(
            client=NS(),
            body={"model": "gpt-6-astra"},
            headers={},
            bearer="token",
            base_url="https://example.test",
        )

        sse_context = ProxyRequestContext(
            request_id="sse", provider="openai_chatgpt",
            responses_transport="sse",
        )
        assert await adapter.open_stream(**common, context=sse_context) == "ordinary-sse"
        ordinary.assert_awaited_once()
        lite.assert_not_awaited()
        assert adapter.retry_policy("gpt-6-astra", sse_context).max_attempts == 3

        lite_context = ProxyRequestContext(
            request_id="lite", provider="openai_chatgpt",
            responses_transport="lite_ws",
        )
        assert await adapter.open_stream(**common, context=lite_context) == "lite-ws"
        lite.assert_awaited_once()
        assert adapter.retry_policy("some-other-model", lite_context).max_attempts == 2

        await adapter.close()

    asyncio.run(run())


def test_ordinary_sse_keeps_reasoning_continuity_and_parallel_tools():
    from claude_code_bridge.proxy.adapters.openai_chatgpt import OpenAIChatGPTAdapter

    body = OpenAIChatGPTAdapter().prepare_request(copy.deepcopy(BODY))

    # Native Codex standard Responses leaves context at the server default.
    assert "context" not in body["reasoning"]
    assert "reasoning.encrypted_content" in body["include"]
    assert body.get("parallel_tool_calls") is not False


@pytest.mark.parametrize("committed", [False, True])
def test_adapter_retry_never_replays_after_tool_use(monkeypatch, committed):
    from claude_code_bridge.proxy.adapters.openai_chatgpt import OpenAIChatGPTAdapter
    from claude_code_bridge.proxy import retry
    from claude_code_bridge.proxy.request_context import ProxyRequestContext
    monkeypatch.setattr(retry, "compute_backoff", lambda *args, **kwargs: 0)

    async def run():
        sockets = []
        async def connect(*args, **kw):
            socket = Socket()
            sockets.append(socket)
            if len(sockets) == 1:
                if committed:
                    await socket.events.put({"type": "response.output_item.added", "item": {
                        "type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "check"}})
                await socket.events.put({"type": "response.failed", "response": {
                    "error": {"code": "server_error", "message": "temporary test failure"}}})
            else:
                await socket.events.put(DONE)
            return socket
        adapter = OpenAIChatGPTAdapter()
        adapter.authorize = AsyncMock(return_value=("token", "https://example.test"))
        adapter._chatgpt_transport = ChatGPTResponsesTransport(connector=connect)
        body = {"model": "openai_chatgpt/gpt-6-astra", "max_tokens": 128,
                "messages": [{"role": "user", "content": "test"}]}
        context = ProxyRequestContext(
            request_id="turn", provider="openai_chatgpt", conversation_id="test",
            responses_transport="lite_ws",
        )
        try:
            chunks = [chunk async for chunk in adapter.call(body, "gpt-6-astra", context)]
            assert len(sockets) == (1 if committed else 2)
            assert bool(b"event: error" in b"".join(chunks)) == committed
            assert bool(b"message_stop" in b"".join(chunks)) != committed
            assert sockets[0].close.await_count == 1
        finally:
            await adapter.close()
    asyncio.run(run())


def test_silent_websocket_recovers_over_http_without_outer_cli_retry(monkeypatch):
    from claude_code_bridge.proxy.adapters.openai_chatgpt import OpenAIChatGPTAdapter
    from claude_code_bridge.proxy import retry
    from claude_code_bridge.proxy.request_context import ProxyRequestContext
    monkeypatch.setattr(retry, "compute_backoff", lambda *args, **kwargs: 0)

    async def run():
        socket = Socket()
        adapter = OpenAIChatGPTAdapter()
        adapter.authorize = AsyncMock(return_value=("token", "https://example.test"))
        adapter._chatgpt_transport = ChatGPTResponsesTransport(
            connector=AsyncMock(return_value=socket),
            first_event_timeout=.01, idle_timeout=.08,
        )

        async def events():
            yield NS(type="response.completed", response=NS(
                usage=None, status="completed",
            ))

        http_stream = NS(response=NS(headers={}), close=AsyncMock())

        class HTTPStream:
            response = http_stream.response
            close = http_stream.close

            def __aiter__(self):
                return events()

        body = {
            "model": "openai_chatgpt/gpt-6-astra", "max_tokens": 128,
            "messages": [{"role": "user", "content": "test"}],
        }
        statuses = []
        context = ProxyRequestContext(
            request_id="turn", provider="openai_chatgpt",
            conversation_id="test",
            responses_transport="lite_ws",
            status_callback=lambda request_id, status: statuses.append(
                (request_id, status)
            ),
        )
        request_client = NS(post=AsyncMock(return_value=HTTPStream()))
        response_client = NS(
            with_options=lambda **kwargs: request_client,
            close=AsyncMock(),
        )
        adapter._http_client = NS()
        adapter._responses_client = response_client
        try:
            chunks = [chunk async for chunk in adapter.call(
                body, "gpt-6-astra", context,
            )]
            joined = b"".join(chunks)
            assert adapter._chatgpt_transport.connector.await_count == 1
            assert socket.close.await_count == 1
            assert b"message_stop" in joined
            assert b'"type":"api_error"' not in joined
            assert [status[1]["state"] for status in statuses] == [
                "reconnecting", "recovered",
            ]
            assert statuses[0][1]["fallback_transport"] == "sse"
            assert statuses[1][1]["transport"] == "sse"
        finally:
            await adapter.close()
    asyncio.run(run())


@pytest.mark.parametrize("committed_kind", ["reasoning", "text", "tool"])
def test_productive_stall_after_committed_output_never_replays(monkeypatch, committed_kind):
    from claude_code_bridge.proxy.adapters.openai_chatgpt import OpenAIChatGPTAdapter
    from claude_code_bridge.proxy import retry
    from claude_code_bridge.proxy.request_context import ProxyRequestContext
    monkeypatch.setattr(retry, "compute_backoff", lambda *args, **kwargs: 0)

    async def run():
        socket = Socket()
        producer = None

        async def connect(*args, **kw):
            nonlocal producer
            if committed_kind == "reasoning":
                await socket.events.put({
                    "type": "response.output_item.added",
                    "item": {"type": "reasoning", "id": "rs_1", "content": []},
                })
                await socket.events.put({
                    "type": "response.reasoning_text.delta", "delta": "thinking",
                    "item_id": "rs_1", "output_index": 0, "content_index": 0,
                })
            elif committed_kind == "text":
                await socket.events.put({
                    "type": "response.output_item.added",
                    "item": {"type": "message", "id": "msg_1", "content": []},
                })
                await socket.events.put({
                    "type": "response.output_text.delta", "delta": "hello",
                    "item_id": "msg_1", "output_index": 0, "content_index": 0,
                })
            else:
                await socket.events.put({
                    "type": "response.output_item.added",
                    "item": {
                        "type": "function_call", "id": "fc_1",
                        "call_id": "call_1", "name": "check",
                    },
                })

            async def chatter():
                while True:
                    await socket.events.put({"type": "response.in_progress"})
                    await asyncio.sleep(.002)

            producer = asyncio.create_task(chatter())
            return socket

        statuses = []
        adapter = OpenAIChatGPTAdapter()
        adapter.authorize = AsyncMock(return_value=("token", "https://example.test"))
        adapter._chatgpt_transport = ChatGPTResponsesTransport(
            connector=AsyncMock(side_effect=connect),
            first_event_timeout=.1,
            idle_timeout=.2,
            productive_idle_timeout=.025,
            attempt_timeout=.2,
        )
        body = {
            "model": "openai_chatgpt/gpt-6-astra", "max_tokens": 128,
            "messages": [{"role": "user", "content": "test"}],
        }
        context = ProxyRequestContext(
            request_id="turn", provider="openai_chatgpt",
            conversation_id="test",
            responses_transport="lite_ws",
            status_callback=lambda request_id, status: statuses.append(status),
        )
        try:
            chunks = [chunk async for chunk in adapter.call(
                body, "gpt-6-astra", context,
            )]
            joined = b"".join(chunks)
            assert adapter._chatgpt_transport.connector.await_count == 1
            assert b'"type":"api_error"' in joined
            assert not statuses
            assert socket.close.await_count == 1
            assert not adapter._chatgpt_transport._idle
        finally:
            if producer is not None:
                producer.cancel()
                await asyncio.gather(producer, return_exceptions=True)
            await adapter.close()

    asyncio.run(run())


def test_fallback_stall_fails_promptly_and_discards_lease(monkeypatch):
    from claude_code_bridge.proxy.adapters.openai_chatgpt import OpenAIChatGPTAdapter
    from claude_code_bridge.proxy import retry
    from claude_code_bridge.proxy.request_context import ProxyRequestContext
    monkeypatch.setattr(retry, "compute_backoff", lambda *args, **kwargs: 0)

    async def run():
        socket = Socket()
        ws_producer = None
        sse_closed = AsyncMock()

        async def connect(*args, **kw):
            nonlocal ws_producer

            async def chatter():
                while True:
                    await socket.events.put({"type": "response.in_progress"})
                    await asyncio.sleep(.002)

            ws_producer = asyncio.create_task(chatter())
            return socket

        async def sse_events():
            while True:
                yield NS(type="response.in_progress")
                await asyncio.sleep(.002)

        class HTTPStream:
            response = NS(headers={})
            close = sse_closed

            def __aiter__(self):
                return sse_events()

        statuses = []
        adapter = OpenAIChatGPTAdapter()
        adapter.authorize = AsyncMock(return_value=("token", "https://example.test"))
        adapter._chatgpt_transport = ChatGPTResponsesTransport(
            connector=AsyncMock(side_effect=connect),
            first_event_timeout=.1,
            idle_timeout=.2,
            productive_idle_timeout=.025,
            attempt_timeout=.2,
        )
        request_client = NS(post=AsyncMock(return_value=HTTPStream()))
        adapter._http_client = NS()
        adapter._responses_client = NS(
            with_options=lambda **kwargs: request_client,
            close=AsyncMock(),
        )
        context = ProxyRequestContext(
            request_id="turn", provider="openai_chatgpt",
            conversation_id="test",
            responses_transport="lite_ws",
            status_callback=lambda request_id, status: statuses.append(status),
        )
        body = {
            "model": "openai_chatgpt/gpt-6-astra", "max_tokens": 128,
            "messages": [{"role": "user", "content": "test"}],
        }
        try:
            started = asyncio.get_running_loop().time()
            chunks = [chunk async for chunk in adapter.call(
                body, "gpt-6-astra", context,
            )]
            assert asyncio.get_running_loop().time() - started < .15
            assert b'"type":"api_error"' in b"".join(chunks)
            assert [status["state"] for status in statuses] == ["reconnecting"]
            assert socket.close.await_count == 1
            assert sse_closed.await_count == 1
            assert not adapter._chatgpt_transport._idle
        finally:
            if ws_producer is not None:
                ws_producer.cancel()
                await asyncio.gather(ws_producer, return_exceptions=True)
            await adapter.close()

    asyncio.run(run())


def test_terminal_does_not_wait_for_transport_eof():
    async def run():
        async def events():
            yield NS(type="response.completed", response=NS(usage=None, status="completed"))
            raise AssertionError("must stop at terminal")
        chunks = [chunk async for chunk in responses_to_anthropic_sse(events(), anthropic_model="astra")]
        assert b"message_stop" in chunks[-1]
    asyncio.run(run())
