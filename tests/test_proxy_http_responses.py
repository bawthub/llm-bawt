"""Offline ASGI regressions for proxy error envelopes and buffered usage.

ASGITransport deliberately does not run the app lifespan: no broker, DB,
service startup, credentials or sockets are involved.
"""

from __future__ import annotations

import asyncio
import gzip
import json

import httpx
import pytest

from claude_code_bridge.proxy import claude_oauth, routes
from claude_code_bridge.proxy.app import create_app
from claude_code_bridge.proxy.errors import anthropic_error
from claude_code_bridge.proxy.request_context import REQUEST_HEADER


def _post(app, *, content=None, body=None, headers=None):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test",
        ) as client:
            return await client.post(
                "/v1/messages", content=content, json=body, headers=headers,
            )

    return asyncio.run(run())


@pytest.mark.parametrize("content,fragment", [
    (b"{", "Body is not valid JSON"),
    (b"[]", "Body must be a JSON object"),
    (b"{}", "Missing required field"),
    (b'{"model":"bare-model"}', "Proxy expects model"),
    (b'{"model":"/missing-provider"}', "Invalid namespaced model"),
    (b'{"model":"unknown-offline-provider/model"}', "Unknown proxy provider"),
])
def test_pre_stream_validation_errors_are_top_level_anthropic(content, fragment):
    response = _post(create_app(), content=content)
    assert response.status_code == 400
    assert response.headers["content-type"] == "application/json"
    payload = response.json()
    assert set(payload) == {"type", "error"}
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "invalid_request_error"
    assert fragment in payload["error"]["message"]


@pytest.mark.parametrize("status,error_type", [
    (401, "authentication_error"),
    (403, "permission_error"),
    (429, "rate_limit_error"),
    (503, "overloaded_error"),
])
def test_http_exception_status_and_headers_survive_asgi(monkeypatch, status, error_type):
    def reject(_provider):
        exc = anthropic_error(status, "offline rejection", error_type)
        exc.headers = {"Retry-After": "17"}
        raise exc

    monkeypatch.setattr(routes, "lookup", reject)
    response = _post(create_app(), body={"model": "offline/model", "stream": True})
    assert response.status_code == status
    assert response.headers["retry-after"] == "17"
    assert response.json() == {
        "type": "error", "error": {"type": error_type, "message": "offline rejection"},
    }


def test_cancelled_request_keeps_499_and_anthropic_body(monkeypatch):
    app = create_app()
    app.state.proxy_cancellations.cancel("cancelled-offline")
    monkeypatch.setattr(routes, "lookup", lambda _provider: object())
    response = _post(
        app, body={"model": "offline/model", "stream": True},
        headers={REQUEST_HEADER: "cancelled-offline"},
    )
    assert response.status_code == 499
    assert response.json() == {
        "type": "error", "error": {"type": "api_error", "message": "Request cancelled"},
    }


def test_unrelated_framework_errors_keep_default_handler():
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app()), base_url="http://proxy.test",
        ) as client:
            return await client.get("/not-a-proxy-route")

    response = asyncio.run(run())
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


@pytest.mark.parametrize("initial,deltas,expected", [
    (
        {"input_tokens": 50, "cache_read_input_tokens": 150,
         "cache_creation_input_tokens": 20, "output_tokens": 1},
        [{"output_tokens": 7}],
        {"input_tokens": 50, "cache_read_input_tokens": 150,
         "cache_creation_input_tokens": 20, "output_tokens": 7},
    ),
    (
        {"input_tokens": 50, "cache_read_input_tokens": 150,
         "cache_creation_input_tokens": 20, "output_tokens": 1},
        [{"input_tokens": 60, "cache_read_input_tokens": 0, "output_tokens": 4},
         {"cache_creation_input_tokens": 30, "output_tokens": 7}],
        {"input_tokens": 60, "cache_read_input_tokens": 0,
         "cache_creation_input_tokens": 30, "output_tokens": 7},
    ),
    (None, [{"input_tokens": 12, "output_tokens": 7}],
     {"input_tokens": 12, "output_tokens": 7}),
    ({"input_tokens": 12, "output_tokens": 1}, [{}],
     {"input_tokens": 12, "output_tokens": 1}),
])
def test_nonstream_endpoint_merges_cumulative_usage(monkeypatch, initial, deltas, expected):
    message = {"id": "m", "type": "message", "role": "assistant",
               "model": "offline/model", "content": []}
    if initial is not None:
        message["usage"] = initial
    events = [{"type": "message_start", "message": message}]
    events.extend({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                   "usage": usage} for usage in deltas)
    events.append({"type": "message_stop"})
    wire = b"".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
        for event in events
    )

    class Adapter:
        def account_hash(self):
            return "offline"

        async def call(self, body, model, context):
            assert body["stream"] is True
            for offset in range(0, len(wire), 7):
                yield wire[offset:offset + 7]

    monkeypatch.setattr(routes, "lookup", lambda _provider: Adapter())
    monkeypatch.setattr(routes, "_PING_INTERVAL", 0)
    response = _post(create_app(), body={"model": "offline/model", "stream": False})
    assert response.status_code == 200
    assert response.json()["usage"] == expected
    assert response.json()["stop_reason"] == "end_turn"


@pytest.mark.parametrize("status", [200, 429])
def test_native_oauth_still_forwards_compressed_bytes_and_headers(monkeypatch, status):
    """The transforming adapter fix must not turn native OAuth into a parser."""
    payload = (b'event: message_start\ndata: {"type":"message_start",'
               b'"message":{"model":"native-model"}}\n\n')
    wire = gzip.compress(payload)
    incoming_body = b'{ "model": "native-model", "stream": true }'

    class RawChunks(httpx.AsyncByteStream):
        async def __aiter__(self):
            for offset in range(0, len(wire), 7):
                yield wire[offset:offset + 7]

    def respond(request):
        assert request.content == incoming_body
        assert request.headers["anthropic-beta"] == "offline-beta"
        assert request.headers["authorization"] == "Bearer offline-token"
        return httpx.Response(status, stream=RawChunks(), headers={
            "content-type": "text/event-stream", "content-encoding": "gzip",
            "content-length": str(len(wire)), "request-id": "native-request",
        })

    monkeypatch.setattr(claude_oauth, "_fetch_broker_token", lambda: ("offline-token", None))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as upstream:
            app = create_app()
            app.state.claude_oauth_gateway = claude_oauth.ClaudeOAuthGateway(upstream)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://proxy.test",
            ) as client:
                async with client.stream(
                    "POST", "/claude-oauth/v1/messages", content=incoming_body,
                    headers={"anthropic-beta": "offline-beta"},
                ) as response:
                    raw = b"".join([chunk async for chunk in response.aiter_raw()])
                    return response, raw

    response, raw = asyncio.run(run())
    assert response.status_code == status
    assert response.headers["content-encoding"] == "gzip"
    assert response.headers["content-length"] == str(len(wire))
    assert response.headers["request-id"] == "native-request"
    assert raw == wire
