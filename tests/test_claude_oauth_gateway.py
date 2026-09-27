"""Request-level rotation recovery never replays the SDK's agent loop."""
from __future__ import annotations

import asyncio
import hashlib
import json

import httpx
import pytest
from starlette.requests import Request

from claude_code_bridge.proxy import claude_oauth as gateway
from claude_code_bridge.proxy.app import create_app
from claude_code_bridge.send_stream import ClaudeStreamMixin


class BytesStream(httpx.AsyncByteStream):
    def __init__(self, data=b"ok", *, fail=False):
        self.data = data
        self.fail = fail
        self.closed = False

    async def __aiter__(self):
        yield self.data
        if self.fail:
            raise httpx.ReadError("connection lost after tool_use")

    async def aclose(self):
        self.closed = True


def request(body=b'{"model":"claude-test","stream":true}', path="messages"):
    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request({
        "type": "http", "method": "POST", "scheme": "http",
        "server": ("localhost", 80), "path": f"/claude-oauth/v1/{path}",
        "query_string": b"beta=true", "headers": [
            (b"authorization", b"Bearer startup-token"), (b"x-api-key", b"not-forwarded"),
            (b"anthropic-beta", b"oauth-2025-04-20,thinking-test"),
            (b"anthropic-version", b"2023-06-01"),
            (b"content-type", b"application/json"),
            (b"connection", b"keep-alive, x-local-only"),
            (b"x-local-only", b"no"),
        ],
    }, receive)


def setup(monkeypatch, statuses, *, tokens=("a", "b"), data=b"ok", fail=False):
    sent = []
    streams = []
    broker_calls = []
    token_iter = iter(tokens)

    def broker(**kwargs):
        broker_calls.append(kwargs)
        return next(token_iter), None

    monkeypatch.setattr(gateway, "_fetch_broker_token", broker)
    codes = iter(statuses)

    async def upstream(req):
        sent.append(req)
        stream = BytesStream(data, fail=fail)
        streams.append(stream)
        return httpx.Response(next(codes), stream=stream, headers={
            "content-type": "text/event-stream", "request-id": "req-upstream",
        })

    transport = httpx.MockTransport(upstream)
    return gateway.ClaudeOAuthGateway(httpx.AsyncClient(transport=transport)), sent, streams, broker_calls


def test_rotation_between_requests_preserves_completed_tool_history(monkeypatch):
    proxy, sent, streams, broker = setup(monkeypatch, [200, 200], tokens=("a", "b"))

    async def run():
        # The SDK performs the first tool just once. Its next request carries
        # that result and must use B despite the unchanged bootstrap bearer.
        first = await proxy.forward(request(), "messages")
        assert b"".join([part async for part in first.body_iterator]) == b"ok"
        body = json.dumps({"model": "claude-test", "messages": [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "once", "name": "Write", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "once", "content": "done"}]},
        ]}).encode()
        second = await proxy.forward(request(body), "messages")
        assert b"".join([part async for part in second.body_iterator]) == b"ok"
        assert sent[1].content == body
        await proxy.close()

    asyncio.run(run())
    assert [r.headers["authorization"] for r in sent] == ["Bearer a", "Bearer b"]
    assert broker == [{}, {}]
    assert all(s.closed for s in streams)


def test_401_retry_is_exact_request_and_invisible_to_sdk(monkeypatch):
    data = b'event: content_block_delta\ndata: {"type":"thinking_delta","thinking":"signed"}\n\n'
    proxy, sent, streams, broker = setup(monkeypatch, [401, 200], data=data)

    async def run():
        result = await proxy.forward(request(), "messages")
        assert result.status_code == 200
        assert result.headers["request-id"] == "req-upstream"
        assert b"".join([part async for part in result.body_iterator]) == data
        await proxy.close()

    asyncio.run(run())
    assert len(sent) == 2
    assert sent[0].content == sent[1].content
    assert str(sent[1].url) == "https://api.anthropic.com/v1/messages?beta=true"
    assert sent[1].headers["authorization"] == "Bearer b"
    assert sent[1].headers["anthropic-beta"] == "oauth-2025-04-20,thinking-test"
    assert "x-api-key" not in sent[1].headers
    assert "x-local-only" not in sent[1].headers
    assert broker == [{}, {"force": True, "rejected_token_sha256": hashlib.sha256(b"a").hexdigest()}]
    assert all(s.closed for s in streams)


@pytest.mark.parametrize("statuses,tokens,expected,calls", [
    ([401, 401], ("a", "b"), 401, 2),
    ([401], ("a", "a"), 401, 1),
    ([401], ("a", None), 401, 1),
    ([429], ("a",), 429, 1),
    ([503], ("a",), 503, 1),
])
def test_failures_are_bounded_and_keep_real_status(monkeypatch, statuses, tokens, expected, calls):
    proxy, sent, streams, _ = setup(monkeypatch, statuses, tokens=tokens)

    async def run():
        result = await proxy.forward(request(), "messages")
        assert result.status_code == expected
        assert b"".join([part async for part in result.body_iterator]) == b"ok"
        await proxy.close()

    asyncio.run(run())
    assert len(sent) == calls
    assert all(s.closed for s in streams)


def test_partial_tool_stream_is_never_retried(monkeypatch):
    tool = b'event: content_block_start\ndata: {"type":"tool_use"}\n\n'
    proxy, sent, streams, broker = setup(monkeypatch, [200], data=tool, fail=True)

    async def run():
        result = await proxy.forward(request(), "messages")
        iterator = result.body_iterator
        assert await anext(iterator) == tool
        with pytest.raises(httpx.ReadError):
            await anext(iterator)
        await proxy.close()

    asyncio.run(run())
    assert len(sent) == len(broker) == 1
    assert streams[0].closed


def test_disconnect_closes_upstream_without_retry(monkeypatch):
    proxy, sent, streams, _ = setup(monkeypatch, [200])

    async def run():
        result = await proxy.forward(request(), "messages")
        assert await anext(result.body_iterator) == b"ok"
        await result.body_iterator.aclose()
        await proxy.close()

    asyncio.run(run())
    assert len(sent) == 1
    assert streams[0].closed


def test_missing_broker_never_uses_bootstrap_token(monkeypatch):
    proxy, sent, _, _ = setup(monkeypatch, [], tokens=(None,))

    async def run():
        result = await proxy.forward(request(), "messages")
        assert result.status_code == 503
        await proxy.close()

    asyncio.run(run())
    assert sent == []


@pytest.mark.parametrize("path", ["messages", "messages/count_tokens"])
def test_gateway_routes_preserve_nonstreaming_json(monkeypatch, path):
    data = b'{"type":"message","content":[],"usage":{"input_tokens":1}}'
    proxy, sent, _, _ = setup(monkeypatch, [200], tokens=("a",), data=data)

    async def run():
        app = create_app()
        app.state.claude_oauth_gateway = proxy
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(f"/claude-oauth/v1/{path}?beta=true", json={"model": "claude-test", "stream": False})
            assert resp.status_code == 200
            assert resp.content == data
        await proxy.close()

    asyncio.run(run())
    assert sent[0].url.path == f"/v1/{path}"
    assert json.loads(sent[0].content)["stream"] is False


@pytest.mark.parametrize("base", ["http://127.0.0.1:12345", None])
@pytest.mark.parametrize("broker_configured", [True, False])
def test_native_sdk_env_retains_oauth_and_model_semantics(monkeypatch, base, broker_configured):
    if broker_configured:
        monkeypatch.setenv("LLM_BAWT_API_URL", "http://app-fixture")
    else:
        monkeypatch.delenv("LLM_BAWT_API_URL", raising=False)
    monkeypatch.setattr("claude_code_bridge.send_stream._get_fresh_oauth_token", lambda **_: "a")
    harness = ClaudeStreamMixin()
    harness._proxy_base_url = base
    env = harness._build_sdk_env(
        use_proxy=False, model="claude-test", subagent_model=None, force_refresh=False,
        bot_id="snark", session_key="snark:test", thread_session_id="test", request_id="test",
    )
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "a"
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    assert "CLAUDE_CODE_SUBAGENT_MODEL" not in env
    assert env.get("ANTHROPIC_BASE_URL") == (
        f"{base}/claude-oauth" if base and broker_configured else None
    )
