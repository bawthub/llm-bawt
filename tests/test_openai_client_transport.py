"""TASK-2339: the direct OpenAI client through the real SDK and real transport.

Other OpenAI client tests stub ``_chat_create_with_fallback`` or the SDK, so
they passed while openai 3 broke every custom-endpoint call: its
``DefaultHttpxClient`` is an httpx2 client, and our httpx connect-reporting
transport inside it failed with a blank AssertionError. These tests send real
HTTP to a loopback server through ``OpenAIClient`` unchanged.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from llm_bawt.clients import openai_client
from llm_bawt.clients.openai_client import OpenAIClient
from llm_bawt.models.message import Message
from llm_bawt.utils.config import Config

COMPLETION = {
    "id": "c1", "object": "chat.completion", "created": 1, "model": "fake",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "TRANSPORT_OK"}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}


def _chunk(content: str | None, finish: str | None = None) -> bytes:
    body = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "fake",
            "choices": [{"index": 0, "delta": {"content": content} if content else {},
                         "finish_reason": finish}]}
    return f"data: {json.dumps(body)}\n\n".encode()


class _Handler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def do_POST(self):  # noqa: N802 - http.server API
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        type(self).requests.append({"path": self.path, "body": body,
                                    "auth": self.headers.get("authorization")})
        if body.get("stream"):
            raw = _chunk("TRANSPORT") + _chunk("_STREAM") + _chunk(None, "stop") + b"data: [DONE]\n\n"
            ctype = "text/event-stream"
        else:
            raw = json.dumps(COMPLETION).encode()
            ctype = "application/json"
        self.send_response(200)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *_args):
        pass


@pytest.fixture
def server() -> Iterator[str]:
    _Handler.requests = []
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{srv.server_port}/v1"
    finally:
        srv.shutdown()
        srv.server_close()


def _client(base_url: str) -> OpenAIClient:
    return OpenAIClient("fake", Config(), base_url=base_url, api_key="test-key")


def test_custom_endpoint_uses_connect_reporting_transport(server):
    client = _client(server)
    transport = client.client._client._transport
    assert isinstance(transport, openai_client._ConnectReportingTransport)


def test_non_streaming_query_round_trips_through_sdk(server):
    reply = _client(server).query([Message(role="user", content="hi")],
                                  plaintext_output=True, stream=False)
    assert reply == "TRANSPORT_OK"
    (sent,) = _Handler.requests
    assert sent["path"] == "/v1/chat/completions"
    assert sent["auth"] == "Bearer test-key"
    assert sent["body"]["messages"][-1] == {"role": "user", "content": "hi"}


def test_streaming_query_round_trips_through_sdk(server):
    reply = _client(server).query([Message(role="user", content="hi")],
                                  plaintext_output=True, stream=True)
    assert reply == "TRANSPORT_STREAM"
    assert _Handler.requests[0]["body"]["stream"] is True


def test_unreachable_endpoint_is_reported(monkeypatch):
    from llm_bawt.service.providers import ollama

    seen: list[str] = []
    monkeypatch.setattr(ollama, "report_unreachable", lambda url, err: seen.append(url))
    base = "http://127.0.0.1:9/v1"
    client = _client(base)
    client.client = client.client.with_options(max_retries=0)
    reply = client.query([Message(role="user", content="hi")], plaintext_output=True, stream=False)
    assert reply.startswith("ERROR")
    assert seen == [base]


def test_unexpected_error_logs_type_and_traceback(server, monkeypatch, caplog):
    client = _client(server)

    def boom(_payload):
        raise AssertionError  # blank str(), as in the openai 3 break

    monkeypatch.setattr(client, "_chat_create_with_fallback", boom)
    with caplog.at_level(logging.ERROR, logger=openai_client.logger.name):
        reply = client.query([Message(role="user", content="hi")], plaintext_output=True, stream=False)
    assert reply == "ERROR: Unexpected error - AssertionError: "
    record = caplog.records[-1]
    assert "AssertionError" in record.getMessage()
    assert record.exc_info is not None


@pytest.mark.parametrize(("content", "expected"), [
    (b'{"error": {"message": "upstream exploded"}}', "upstream exploded"),
    (b'{"detail": "Cannot accept session"}', "Cannot accept session"),
    (b"Bad Gateway from the edge", "Bad Gateway from the edge"),
])
def test_api_error_text_keeps_status_and_body_reason(content, expected):
    import httpx
    import openai

    from llm_bawt.clients.base import api_error_text

    response = httpx.Response(502, content=content, request=httpx.Request("POST", "http://x/v1"))
    error = openai.OpenAI(api_key="k", base_url="http://x/v1")._make_status_error_from_response(response)
    text = api_error_text(error)
    assert text.startswith("Error code: 502")
    assert expected in text
    assert text.count(expected) == 1


class _RejectingHandler(_Handler):
    """Refuses the first token-parameter name (or temperature) it sees with a
    non-OpenAI-shaped 400 body, as gateways do, then accepts the retry."""

    mode = "token"
    plain = False
    refused_key: str | None = None

    def do_POST(self):  # noqa: N802 - http.server API
        raw = self.rfile.read(int(self.headers["content-length"]))
        body = json.loads(raw)
        cls = type(self)
        cls.requests.append({"path": self.path, "body": body, "auth": None})
        if cls.mode == "token":
            key = next((k for k in ("max_completion_tokens", "max_tokens") if k in body), None)
            if cls.refused_key is None:
                cls.refused_key = key
            reject = key is not None and key == cls.refused_key
            reason = f"unsupported parameter {key}"
        else:
            reject = "temperature" in body
            reason = "temperature is unsupported for this model"
        if reject:
            out = reason.encode() if cls.plain else json.dumps({"detail": reason}).encode()
            self.send_response(400)
            self.send_header("content-type", "text/plain" if cls.plain else "application/json")
        else:
            out = json.dumps(COMPLETION).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.mark.parametrize("plain", [False, True])
@pytest.mark.parametrize("mode", ["token", "temperature"])
def test_parameter_fallback_reads_non_openai_error_bodies(mode, plain):
    _RejectingHandler.requests = []
    _RejectingHandler.mode, _RejectingHandler.plain, _RejectingHandler.refused_key = mode, plain, None
    srv = HTTPServer(("127.0.0.1", 0), _RejectingHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        client = _client(f"http://127.0.0.1:{srv.server_port}/v1")
        client.client = client.client.with_options(max_retries=0)
        payload = {"model": "fake", "messages": [{"role": "user", "content": "hi"}],
                   "stream": False, "temperature": 0.2}
        completion = client._chat_create_with_fallback(payload)
    finally:
        srv.shutdown()
        srv.server_close()
    assert completion.choices[0].message.content == "TRANSPORT_OK"
    sent = [r["body"] for r in _RejectingHandler.requests]
    assert len(sent) == 2
    if mode == "temperature":
        assert "temperature" in sent[0] and "temperature" not in sent[1]
    else:
        first = _RejectingHandler.refused_key
        assert first in sent[0] and first not in sent[1]
