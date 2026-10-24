"""DB-free public completion adapter tests: observation is not submission."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.service import chat_response_adapter as adapter
from llm_bawt.service.routes import botchat, chat

RECEIPT = {
    "object": "chat.completion.queued", "status": "QUEUED",
    "delivery_id": "delivery-1", "user_message_id": "00000000-0000-4000-8000-000000000001",
}
BODY = {"model": "test-model", "messages": [{"role": "user", "content": "hello"}]}


def _client(monkeypatch, *, botchat_route=False, stream_events=None):
    async def stream(_request):
        for event in stream_events or []:
            yield event

    service = SimpleNamespace(chat_completion=AsyncMock(return_value=dict(RECEIPT)),
                              chat_completion_stream=stream)
    route = botchat if botchat_route else chat
    monkeypatch.setattr(route, "get_service", lambda: service)
    app = FastAPI()
    app.include_router(route.router)
    path = ("/v1/botchat/test-bot/nick/chat/completions" if botchat_route
            else "/v1/chat/completions")
    return TestClient(app), service, path


def test_nonstream_receipt_waits_on_exact_original(monkeypatch):
    observer = AsyncMock(return_value={"status": "DELIVERED", "response_text": "hello back"})
    monkeypatch.setattr(adapter, "observe_receipt", observer)
    client, service, path = _client(monkeypatch)
    with client:
        response = client.post(path, json={**BODY, "stream": False},
                               headers={"X-LLM-Bawt-Response-Mode": "openai"})
    assert response.status_code == 200
    assert response.json()["object"] == "chat.completion"
    assert response.json()["choices"][0]["message"]["content"] == "hello back"
    assert response.json()["usage"] is None  # Unknown is not zero measured tokens.
    observer.assert_awaited_once()
    assert observer.await_args.args[0] is service
    assert observer.await_args.args[2] == RECEIPT
    service.chat_completion.assert_awaited_once()


@pytest.mark.parametrize("headers", [
    {},
    {"X-LLM-Bawt-Response-Mode": "unknown"},
    {"X-LLM-Bawt-Admission-Events": "comment"},
])
@pytest.mark.parametrize("stream", [False, True])
def test_unmigrated_caller_keeps_original_receipt_contract(monkeypatch, headers, stream):
    events = [f"data: {json.dumps(RECEIPT)}\n\n", "data: [DONE]\n\n"]
    observer = AsyncMock(side_effect=AssertionError("legacy request unexpectedly waited"))
    monkeypatch.setattr(adapter, "observe_receipt", observer)
    client, service, path = _client(monkeypatch, stream_events=events)
    with client:
        response = client.post(path, json={**BODY, "stream": stream}, headers=headers)
    if stream:
        assert response.status_code == 200
        assert response.text == "".join(events)
        service.chat_completion.assert_not_awaited()
    else:
        assert response.status_code == 202
        assert response.json() == RECEIPT
        service.chat_completion.assert_awaited_once()
    observer.assert_not_awaited()


def test_botchat_stream_preserves_legacy_pending_receipt_failure_shape(monkeypatch):
    events = [f"data: {json.dumps(RECEIPT)}\n\n", "data: [DONE]\n\n"]
    client, service, path = _client(monkeypatch, botchat_route=True, stream_events=events)
    with client:
        response = client.post(path, json={**BODY, "stream": True})
    chunks = [line[6:] for line in response.text.split("\n\n") if line.startswith("data: ")]
    assert chunks[-1] == "[DONE]"
    assert json.loads(chunks[0])["object"] == "service.warning"
    assert json.loads(chunks[-2])["choices"][0]["finish_reason"] == "stop"
    service.chat_completion.assert_not_awaited()


@pytest.mark.parametrize("reply,code", [
    ({"status": "FAILED", "error": "bridge rejected"}, 502),
    ({"status": "UNRESOLVED"}, 502),
    ({"status": "IN_FLIGHT"}, 504),
])
def test_nonstream_failure_never_returns_receipt(monkeypatch, reply, code):
    monkeypatch.setattr(adapter, "observe_receipt", AsyncMock(return_value=reply))
    client, service, path = _client(monkeypatch)
    with client:
        response = client.post(path, json=BODY,
                               headers={"X-LLM-Bawt-Response-Mode": "openai"})
    assert response.status_code == code
    assert "choices" not in response.json()
    service.chat_completion.assert_awaited_once()


@pytest.mark.parametrize("reply,success", [
    ({"status": "DELIVERED", "response_text": "the answer"}, True),
    ({"status": "IN_FLIGHT"}, False),
    ({"status": "FAILED", "error": "bridge rejected"}, False),
])
def test_stream_receipt_exact_reply_or_error(monkeypatch, reply, success):
    events = [": connected\n\n", f"data: {json.dumps(RECEIPT)}\n\n", "data: [DONE]\n\n"]
    observer = AsyncMock(return_value=reply)
    monkeypatch.setattr(adapter, "observe_receipt", observer)
    client, service, path = _client(monkeypatch, stream_events=events)
    with client:
        response = client.post(path, json={**BODY, "stream": True},
                               headers={"X-LLM-Bawt-Response-Mode": "openai"})
    chunks = [line[6:] for line in response.text.split("\n\n") if line.startswith("data: ")]
    assert all("chat.completion.queued" not in item for item in chunks)
    assert (chunks[-1] == "[DONE]") is success
    if success:
        assert json.loads(chunks[0])["choices"][0]["delta"]["content"] == "the answer"
        assert json.loads(chunks[1])["choices"][0]["finish_reason"] == "stop"
    else:
        assert len(chunks) == 1
        assert "error" in json.loads(chunks[0])
    observer.assert_awaited_once()
    service.chat_completion.assert_not_awaited()


@pytest.mark.parametrize("warning", ["request_failed: denied", "stream_interrupted: dropped"])
def test_stream_failure_never_emits_success_trailer(monkeypatch, warning):
    events = ["data: " + json.dumps({"object": "service.warning", "warnings": [warning]}) + "\n\n",
              'data: {"object":"chat.completion.chunk","choices":[{"finish_reason":"stop"}]}\n\n',
              "data: [DONE]\n\n"]
    client, service, path = _client(monkeypatch, stream_events=events)
    with client:
        response = client.post(path, json={**BODY, "stream": True},
                               headers={"X-LLM-Bawt-Response-Mode": "openai"})
    assert "[DONE]" not in response.text
    assert "finish_reason" not in response.text
    assert json.loads(response.text.strip()[6:])["error"]["message"] == warning
    service.chat_completion.assert_not_awaited()


def test_negotiated_admission_comment_is_not_an_openai_data_frame(monkeypatch):
    events = [f"data: {json.dumps(RECEIPT)}\n\n", "data: [DONE]\n\n"]
    monkeypatch.setattr(adapter, "observe_receipt", AsyncMock(
        return_value={"status": "DELIVERED", "response_text": "answer"},
    ))
    client, _service, path = _client(monkeypatch, stream_events=events)
    with client:
        response = client.post(
            path, json={**BODY, "stream": True},
            headers={"X-LLM-Bawt-Response-Mode": "openai",
                     "X-LLM-Bawt-Admission-Events": "comment"},
        )
    assert f": llm-bawt-admission {json.dumps(RECEIPT)}\n\n" in response.text
    assert f"data: {json.dumps(RECEIPT)}\n\n" not in response.text
    assert response.text.endswith("data: [DONE]\n\n")


def test_comment_consumer_can_disconnect_before_observer_starts(monkeypatch):
    from llm_bawt.service.schemas import ChatCompletionRequest, ChatMessage

    observer = AsyncMock(side_effect=AssertionError("observer started after consumer left"))
    monkeypatch.setattr(adapter, "observe_receipt", observer)

    async def stream(_request):
        yield f"data: {json.dumps(RECEIPT)}\n\n"
        yield "data: [DONE]\n\n"

    request = ChatCompletionRequest(
        model="test-model", bot_id="test-bot", user="nick", stream=True,
        messages=[ChatMessage(role="user", content="hello")],
    )
    service = SimpleNamespace(chat_completion_stream=stream)

    async def consume_comment_only():
        events = adapter.completion_stream(service, request, admission_comment=True)
        first = await anext(events)
        assert first.startswith(": llm-bawt-admission ")
        await events.aclose()

    asyncio.run(consume_comment_only())
    observer.assert_not_awaited()


def test_delivery_in_flight_is_observed_until_terminal(monkeypatch):
    from llm_bawt.service.delivery_reply import wait_for_delivery_reply

    record = SimpleNamespace(id="delivery-1", status="IN_FLIGHT", last_error=None,
                             target_bot_id="test-bot", author_entity_type="user",
                             author_entity_id="nick", user_message_id=RECEIPT["user_message_id"],
                             turn_id="turn-1", metadata={"ordinary_chat": {"digest": "test"}})
    delivery = SimpleNamespace(get=lambda _id: record)
    polled = asyncio.Event()
    resume = asyncio.Event()

    async def wait_for_next_poll(_delay):
        polled.set()
        await resume.wait()

    monkeypatch.setattr("llm_bawt.service.delivery_reply.asyncio.sleep", wait_for_next_poll)
    turn = SimpleNamespace(ended_at=1, status="ok", error_text=None,
                           bot_id="test-bot", user_id="nick",
                           trigger_message_id=RECEIPT["user_message_id"], response_text="done")
    turns = SimpleNamespace(get_turn=lambda _id: turn)

    async def wait():
        task = asyncio.create_task(wait_for_delivery_reply(
            delivery, turns, "delivery-1", bot_id="test-bot", user_id="nick",
            user_message_id=RECEIPT["user_message_id"], timeout_seconds=2,
        ))
        await asyncio.wait_for(polled.wait(), timeout=1)
        assert not task.done()
        record.status = "DELIVERED"
        resume.set()
        return await task

    assert asyncio.run(wait())["response_text"] == "done"


def _sse(payload):
    return f"data: {json.dumps(payload)}\n\n"


def _chunk(delta, finish_reason=None):
    return _sse({"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m",
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]})


TEXT = _chunk({"content": "hi"})
STOP = _chunk({}, "stop")
EXTENSIONS = [
    _sse({"object": "service.warning", "model": "m", "warnings": ["model fallback"]}),
    _chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "text_offset": 0,
                            "function": {"name": "time", "arguments": "{}"}}]}),
    _chunk({}, "tool_calls"),
    _sse({"object": "chat.completion.usage", "turn_id": "t", "token_usage": {}}),
    _sse({"object": "service.animation", "animation": "wave"}),
    _sse({"object": "tool.await_result", "tool_use_id": "q1"}),
]
DIRECT = [": connected\n\n", TEXT, *EXTENSIONS, STOP, "data: [DONE]\n\n"]


@pytest.mark.parametrize("headers,expected_extensions", [
    # Stock OpenAI client: server-run tools and sidecars never reach it.
    ({}, []),
    # BawtHub proxy that predates extension negotiation: frames unchanged.
    ({"X-LLM-Bawt-Admission-Events": "comment"}, EXTENSIONS),
    # Negotiated: the exact legacy JSON, carried as an SSE comment.
    ({"X-LLM-Bawt-Admission-Events": "comment", "X-LLM-Bawt-Extension-Events": "comment"},
     [": llm-bawt-event " + frame[6:] for frame in EXTENSIONS]),
    ({"X-LLM-Bawt-Extension-Events": "comment"},
     [": llm-bawt-event " + frame[6:] for frame in EXTENSIONS]),
])
def test_openai_stream_extension_frames_follow_negotiation(monkeypatch, headers, expected_extensions):
    client, service, path = _client(monkeypatch, stream_events=DIRECT)
    with client:
        response = client.post(path, json={**BODY, "stream": True},
                               headers={"X-LLM-Bawt-Response-Mode": "openai", **headers})
    assert response.text == "".join([": connected\n\n", TEXT, *expected_extensions,
                                     STOP, "data: [DONE]\n\n"])
    service.chat_completion.assert_not_awaited()


def test_legacy_stream_is_untouched_by_extension_negotiation(monkeypatch):
    client, _service, path = _client(monkeypatch, stream_events=DIRECT)
    with client:
        response = client.post(path, json={**BODY, "stream": True},
                               headers={"X-LLM-Bawt-Extension-Events": "comment"})
    assert response.text == "".join(DIRECT)
