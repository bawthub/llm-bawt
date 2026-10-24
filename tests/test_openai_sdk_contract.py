"""Stock OpenAI SDK contract for /v1/chat/completions (TASK-1144 cutover gate).

Drives the real ``openai`` client against the real route with no llm-bawt
knowledge beyond ``extra_body``. ``OPENAI`` cases use the opt-in response mode
and must pass now. ``DEFAULT`` cases are the same calls with no header. Where
today's default is not compliant they run under ``_contract`` and must FAIL the
compliance assertions (the gate forbids xfail). Once the default flips they stop
failing, ``_contract`` raises, and the cutover is done when ``PRECUTOVER`` is
deleted and every case passes with no header.
"""

import contextlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import openai
import pydantic
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from llm_bawt.service import chat_response_adapter as adapter
from llm_bawt.service.routes import chat
from llm_bawt.service.schemas import ChatCompletionChoice, ChatCompletionResponse, ChatMessage

OPENAI = {"X-LLM-Bawt-Response-Mode": "openai"}
DEFAULT: dict[str, str] = {}
PRECUTOVER = "precutover-default"
# Compliant now in openai mode; the no-header default is a documented defect.
MODES = [pytest.param(OPENAI, id="openai"), pytest.param(PRECUTOVER, id="default")]
BOTH_PASS = [pytest.param(OPENAI, id="openai"), pytest.param(DEFAULT, id="default")]


def _headers(mode) -> dict[str, str]:
    return DEFAULT if mode == PRECUTOVER else mode


def _contract(mode):
    """Compliance block: must pass in openai mode, must fail pre-cutover."""
    if mode != PRECUTOVER:
        return contextlib.nullcontext()
    # Failed = a nested pytest.raises that did not raise; ValidationError = a
    # receipt or sidecar frame that is not a valid ChatCompletion[Chunk].
    return pytest.raises((AssertionError, pytest.fail.Exception, pydantic.ValidationError))

RECEIPT = {
    "object": "chat.completion.queued", "status": "QUEUED", "delivery_id": "delivery-1",
    "turn_id": "turn-delivery-1", "user_message_id": "00000000-0000-4000-8000-000000000001",
}
MESSAGES = [{"role": "user", "content": "hello"}]
EXTRA = {"bot_id": "test-bot", "user": "test-user"}


def _frame(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _chunk(delta: dict, finish_reason: str | None = None) -> str:
    return _frame({
        "id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1,
        "model": "test-model", "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    })


# The frame kinds chat_streaming.py really emits on a direct turn: an early
# comment, a non-failure warning, reasoning, a tool the SERVER already ran
# (delta.tool_calls + finish_reason tool_calls), usage, animation, stop, DONE.
DIRECT_STREAM = [
    ": connected\n\n",
    _frame({"object": "service.warning", "model": "test-model", "warnings": ["model fallback: x -> y"]}),
    _chunk({"role": "assistant", "content": "Let me "}),
    _chunk({"reasoning_content": "thinking"}),
    _chunk({"content": "check. "}),
    _chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "text_offset": 15,
                            "function": {"name": "time", "arguments": "{}"}}]}),
    _chunk({}, "tool_calls"),
    _chunk({"content": "It is noon."}),
    _frame({"object": "chat.completion.usage", "turn_id": "turn-1", "model": "test-model",
            "token_usage": {"input_tokens": 3}}),
    _frame({"object": "service.animation", "animation": "wave"}),
    _chunk({}, "stop"),
    "data: [DONE]\n\n",
]
QUEUED_STREAM = [": connected\n\n", _frame(RECEIPT), "data: [DONE]\n\n"]


@pytest.fixture
def harness(monkeypatch):
    seen: list = []
    service = SimpleNamespace(chat_completion=AsyncMock(), stream_events=[])

    async def chat_completion_stream(request):
        seen.append(request)
        for event in service.stream_events:
            yield event

    async def chat_completion(request):
        seen.append(request)
        return service.chat_completion.return_value

    service.chat_completion_stream = chat_completion_stream
    service.chat_completion.side_effect = chat_completion
    monkeypatch.setattr(chat, "get_service", lambda: service)
    app = FastAPI()
    app.include_router(chat.router)
    http = TestClient(app)

    def sdk(headers: dict[str, str]) -> openai.OpenAI:
        return openai.OpenAI(base_url="http://testserver/v1", api_key="x",
                             http_client=http, default_headers=headers, max_retries=0)

    with http:
        yield SimpleNamespace(service=service, sdk=sdk, seen=seen)


def _delivered(monkeypatch, text="queued answer"):
    observer = AsyncMock(return_value={"status": "DELIVERED", "response_text": text})
    monkeypatch.setattr(adapter, "observe_receipt", observer)
    return observer


def _unknown_keys(raw: dict) -> list[str]:
    """Wire keys the stock SDK's own models do not define (it keeps them as extras)."""
    unknown = [k for k in raw if k not in ChatCompletionChunk.model_fields]
    for choice in raw.get("choices") or []:
        unknown += [f"choices.{k}" for k in choice if k not in ChunkChoice.model_fields]
        unknown += [f"delta.{k}" for k in choice.get("delta") or {} if k not in ChoiceDelta.model_fields]
    return unknown


def _strict_chunks(stream) -> list[ChatCompletionChunk]:
    chunks = []
    for chunk in stream:
        # The SDK builds chunks without validation; a sidecar frame would come
        # back as a "chunk" with no choices, and unknown fields (for example
        # delta.reasoning_content) survive as extras. Re-validate strictly.
        raw = chunk.to_dict()
        strict = ChatCompletionChunk.model_validate(raw)
        assert strict.object == "chat.completion.chunk"
        assert strict.choices, f"non-chunk frame reached the SDK: {raw}"
        assert not _unknown_keys(raw), f"non-standard fields reached the SDK: {_unknown_keys(raw)}"
        chunks.append(strict)
    return chunks


def _assert_one_answer(chunks: list[ChatCompletionChunk], text: str) -> None:
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == text
    finishes = [c.choices[0].finish_reason for c in chunks if c.choices[0].finish_reason]
    assert finishes == ["stop"]
    # Every tool here ran on the server. delta.tool_calls tells an OpenAI
    # client to execute the call itself, so it must never appear.
    assert all(not c.choices[0].delta.tool_calls for c in chunks)


@pytest.mark.parametrize("headers", BOTH_PASS)
def test_nonstream_direct_chat_bot(harness, headers):
    harness.service.chat_completion.return_value = ChatCompletionResponse(
        model="test-model", choices=[ChatCompletionChoice(
            index=0, message=ChatMessage(role="assistant", content="direct answer"))],
    )
    result = harness.sdk(headers).chat.completions.create(model="test-model", messages=MESSAGES, extra_body=EXTRA)
    strict = ChatCompletion.model_validate(result.to_dict())
    assert strict.choices[0].message.content == "direct answer"
    assert strict.choices[0].finish_reason == "stop"


@pytest.mark.parametrize("mode", MODES)
def test_nonstream_queued_agent_turn_returns_the_answer(harness, monkeypatch, mode):
    _delivered(monkeypatch)
    harness.service.chat_completion.return_value = dict(RECEIPT)
    result = harness.sdk(_headers(mode)).chat.completions.create(
        model="test-model", messages=MESSAGES, extra_body=EXTRA)
    with _contract(mode):  # pre-cutover: HTTP 202 chat.completion.queued receipt
        strict = ChatCompletion.model_validate(result.to_dict())
        assert strict.object == "chat.completion"
        assert strict.choices[0].message.content == "queued answer"


@pytest.mark.parametrize("mode", MODES)
def test_nonstream_queued_failure_is_an_sdk_error(harness, monkeypatch, mode):
    monkeypatch.setattr(adapter, "observe_receipt", AsyncMock(
        return_value={"status": "FAILED", "error": "bridge rejected"}))
    harness.service.chat_completion.return_value = dict(RECEIPT)
    with _contract(mode):  # pre-cutover: the receipt is a 2xx "success"
        with pytest.raises(openai.APIStatusError) as caught:
            harness.sdk(_headers(mode)).chat.completions.create(
                model="test-model", messages=MESSAGES, extra_body=EXTRA)
        assert caught.value.status_code == 502


@pytest.mark.parametrize("mode", MODES)
def test_stream_direct_turn_is_standard_chunks_only(harness, mode):
    harness.service.stream_events = DIRECT_STREAM
    stream = harness.sdk(_headers(mode)).chat.completions.create(
        model="test-model", messages=MESSAGES, stream=True, extra_body=EXTRA)
    with _contract(mode):  # pre-cutover: sidecars and server-run tool_calls leak
        _assert_one_answer(_strict_chunks(stream), "Let me check. It is noon.")


@pytest.mark.parametrize("mode", MODES)
def test_stream_queued_agent_turn_streams_the_answer(harness, monkeypatch, mode):
    _delivered(monkeypatch)
    harness.service.stream_events = QUEUED_STREAM
    stream = harness.sdk(_headers(mode)).chat.completions.create(
        model="test-model", messages=MESSAGES, stream=True, extra_body=EXTRA)
    with _contract(mode):  # pre-cutover: the receipt frame instead of the answer
        _assert_one_answer(_strict_chunks(stream), "queued answer")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("warning", ["stream_interrupted: bridge dropped", "request_failed: denied"])
def test_stream_failure_raises_instead_of_a_fake_success(harness, mode, warning):
    harness.service.stream_events = [
        _chunk({"role": "assistant", "content": "partial"}),
        _frame({"object": "service.warning", "model": "test-model", "warnings": [warning]}),
        _chunk({}, "stop"),
        "data: [DONE]\n\n",
    ]
    stream = harness.sdk(_headers(mode)).chat.completions.create(
        model="test-model", messages=MESSAGES, stream=True, extra_body=EXTRA)
    received = []
    with _contract(mode):  # pre-cutover: warning frame, then stop + [DONE]
        with pytest.raises(openai.APIError, match=warning):
            for chunk in stream:
                received.append(chunk)
        assert all(not c.choices or c.choices[0].finish_reason is None for c in received)


@pytest.mark.parametrize("headers", BOTH_PASS)
@pytest.mark.parametrize("stream", [False, True])
def test_extension_fields_ride_extra_body(harness, monkeypatch, headers, stream):
    _delivered(monkeypatch)
    harness.service.chat_completion.return_value = dict(RECEIPT)
    harness.service.stream_events = QUEUED_STREAM
    response = harness.sdk(headers).chat.completions.create(
        model="test-model", messages=MESSAGES, stream=stream,
        extra_body={**EXTRA, "augment_memory": False})
    if stream:
        list(response)
    (request,) = harness.seen
    assert (request.bot_id, request.user, request.augment_memory) == ("test-bot", "test-user", False)
