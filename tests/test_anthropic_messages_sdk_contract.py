"""AnthropicMessagesClient against the installed anthropic SDK (TASK-2298).

No mock of ``messages.create``: the real SDK builds and sends the request
through an in-memory transport, so a typed kwarg the installed SDK no longer
declares fails here (TypeError) instead of in production.
"""

from __future__ import annotations

import inspect
import json

import pytest

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")

from llm_bawt.clients.anthropic_messages_client import AnthropicMessagesClient  # noqa: E402
from llm_bawt.models.message import Message  # noqa: E402
from llm_bawt.utils.config import Config  # noqa: E402

_OK = {
    "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
    "content": [{"type": "text", "text": "hello"}],
    "stop_reason": "end_turn", "stop_sequence": None,
    "usage": {"input_tokens": 3, "output_tokens": 1},
}


def _client(handler) -> tuple[AnthropicMessagesClient, list[dict]]:
    sent: list[dict] = []

    def record(request):
        body = json.loads(request.content)
        sent.append(body)
        return handler(body)

    client = AnthropicMessagesClient("m", Config(), base_url="https://a.example", api_key="k")
    client.client = anthropic.Anthropic(
        api_key="k", base_url="https://a.example", max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(record)),
    )
    return client, sent


def _messages() -> list[Message]:
    return [Message(role="system", content="be brief"), Message(role="user", content="hi")]


def test_query_sends_temperature_and_thinking_through_the_real_sdk():
    client, sent = _client(lambda body: httpx2.Response(200, json=_OK))
    out = client.query(_messages(), temperature=0.3, thinking={"type": "disabled"})
    assert out == "hello"
    assert sent[0]["temperature"] == 0.3
    assert sent[0]["thinking"] == {"type": "disabled"}
    assert sent[0]["system"] == "be brief"


def test_query_bounded_sends_through_the_real_sdk():
    client, sent = _client(lambda body: httpx2.Response(200, json=_OK))
    assert client.query_bounded(_messages(), timeout_seconds=5, max_tokens=16, temperature=0.0) == "hello"
    assert sent[0]["max_tokens"] == 16 and sent[0]["temperature"] == 0.0


def test_rejected_temperature_is_retried_without_it_and_remembered():
    def handler(body):
        if "temperature" in body:
            return httpx2.Response(400, json={"type": "error", "error": {
                "type": "invalid_request_error", "message": "temperature is not supported"}})
        return httpx2.Response(200, json=_OK)

    client, sent = _client(handler)
    assert client.query(_messages()) == "hello"
    assert client.query(_messages()) == "hello"
    assert ["temperature" in body for body in sent] == [True, False, False]


def test_every_typed_kwarg_is_declared_by_the_installed_sdk():
    client, _ = _client(lambda body: httpx2.Response(200, json=_OK))
    create = client.client.messages.create
    payload = client._payload(_messages(), max_tokens=8, optional={
        "temperature": 0.1, "thinking": {"type": "disabled"}})
    kwargs = client._sdk_kwargs(create, payload)
    inspect.signature(create).bind(**kwargs)  # TypeError on an undeclared field
