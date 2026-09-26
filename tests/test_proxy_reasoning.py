"""TASK-931: native reasoning survives the SDK wire/transcript round trip."""
import asyncio
import copy
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
import pytest
from anthropic import AsyncAnthropic

from claude_code_bridge.proxy.adapters.openai_chatgpt import OpenAIChatGPTAdapter
from claude_code_bridge.proxy.chatgpt_transport import lite_request, _event_object
from claude_code_bridge.proxy.reasoning import ReasoningCodec, SIGNATURE_PREFIX
from claude_code_bridge.proxy.request_context import ProxyRequestContext
from claude_code_bridge.proxy.retry import RetryPhase, phase_from_state
from claude_code_bridge.proxy.stream import TranslatorState, responses_to_anthropic_sse
from claude_code_bridge.proxy.translate import anthropic_to_responses

MODEL = "test-reasoning-model"
BASE = "https://provider.test/codex"
CODEC = ReasoningCodec.for_account("openai_chatgpt", MODEL, BASE, "account-a")
NATIVE = {"type": "reasoning", "id": "rs_test", "summary": [
    {"type": "summary_text", "text": "First section."},
    {"type": "summary_text", "text": "Second section."},
], "encrypted_content": "opaque+/=NOT-PLAINTEXT", "status": "completed"}
DONE = {"type": "response.completed", "response": {"status": "completed", "usage": None}}


async def events(values):
    for value in values:
        yield _event_object(value)


def native_events(item, *, added=True, summary=True):
    result = []
    if added:
        result.append({"type": "response.output_item.added", "item": {
            "type": "reasoning", "id": item["id"], "summary": [],
        }})
    if summary:
        for idx, part in enumerate(item["summary"]):
            result.extend([
                {"type": "response.reasoning_summary_part.added", "summary_index": idx},
                {"type": "response.reasoning_summary_text.delta", "delta": part["text"]},
            ])
    result.append({"type": "response.output_item.done", "item": item})
    return result


async def parse_sdk_message(frames):
    """Use the real Anthropic streaming parser, not a homemade block merger."""
    transport = httpx.MockTransport(lambda _: httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=b"".join(frames),
    ))
    async with AsyncAnthropic(api_key="test", http_client=httpx.AsyncClient(transport=transport)) as client:
        async with client.messages.stream(model="test", max_tokens=10, messages=[
            {"role": "user", "content": "hello"},
        ]) as stream:
            message = await stream.get_final_message()
            return json.loads(message.model_dump_json())["content"]


async def roundtrip_blocks(items, **kwargs):
    source = []
    for item in items:
        source.extend(native_events(item, **kwargs))
    source.extend([{"type": "response.output_text.delta", "delta": "Answer"}, DONE])
    frames = [frame async for frame in responses_to_anthropic_sse(
        events(source), "openai_chatgpt/" + MODEL, reasoning_codec=CODEC,
    )]
    return await parse_sdk_message(frames)


def payload(blocks, codec=CODEC):
    return anthropic_to_responses({"messages": [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": blocks},
    ]}, MODEL, reasoning_codec=codec)


@pytest.mark.parametrize("summary,added", [(True, True), (False, True), (False, False)])
def test_encrypted_reasoning_roundtrip_with_or_without_visible_summary(summary, added):
    item = copy.deepcopy(NATIVE)
    if not summary:
        item["summary"] = []
    blocks = asyncio.run(roundtrip_blocks([item], summary=summary, added=added))
    assert blocks[0]["signature"].startswith(SIGNATURE_PREFIX)
    assert CODEC.decode(blocks[0]["signature"]) == item
    assert payload(blocks)["input"][1] == item
    assert blocks[0]["thinking"] == ("First section.\n\nSecond section." if summary else "")


def test_multiple_native_items_order_and_text_are_preserved():
    second = {**NATIVE, "id": "rs_second", "encrypted_content": "OTHER"}
    blocks = asyncio.run(roundtrip_blocks([NATIVE, second]))
    blocks.insert(1, {"type": "text", "text": "Between reasoning items"})
    assert payload(blocks)["input"][1:] == [
        NATIVE, {"role": "assistant", "content": "Between reasoning items"},
        second, {"role": "assistant", "content": "Answer"},
    ]


def test_resume_from_disk_uses_no_process_local_state(tmp_path):
    blocks = asyncio.run(roundtrip_blocks([NATIVE]))
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(json.dumps({"role": "assistant", "content": blocks}) + "\n")
    restored = json.loads(transcript.read_text())["content"]
    fresh_codec = ReasoningCodec.for_account("openai_chatgpt", MODEL, BASE, "account-a")
    assert payload(restored, fresh_codec)["input"][1] == NATIVE
    # Compaction is owned by the SDK. Absent blocks aren't resurrected.
    assert payload([{"type": "text", "text": "Compacted summary"}], fresh_codec)["input"][1:] == [
        {"role": "assistant", "content": "Compacted summary"},
    ]


@pytest.mark.parametrize("codec", [
    ReasoningCodec.for_account("other-provider", MODEL, BASE, "account-a"),
    ReasoningCodec.for_account("openai_chatgpt", "different-model", BASE, "account-a"),
    ReasoningCodec.for_account("openai_chatgpt", MODEL, BASE, "account-b"),
    ReasoningCodec.for_account("openai_chatgpt", MODEL, "https://other.test", "account-a"),
    None,
])
def test_foreign_scope_is_omitted_from_request_not_transcript(codec):
    blocks = asyncio.run(roundtrip_blocks([NATIVE]))
    original = copy.deepcopy(blocks)
    assert payload(blocks, codec)["input"][1:] == [{"role": "assistant", "content": "Answer"}]
    assert blocks == original
    assert payload(blocks)["input"][1] == NATIVE


@pytest.mark.parametrize("signature", ["native-anthropic-signature", "reasoning:rs_old", "gAAAAlegacy-blob"])
def test_legacy_or_synthetic_signatures_are_not_guessed(signature):
    blocks = [{"type": "thinking", "thinking": "display only", "signature": signature},
              {"type": "text", "text": "Answer"}]
    assert payload(blocks)["input"][1:] == [{"role": "assistant", "content": "Answer"}]


def test_corrupt_our_own_envelope_fails_visibly():
    with pytest.raises(ValueError, match="Malformed"):
        CODEC.decode(SIGNATURE_PREFIX + "not valid base64")


def test_display_summary_without_encrypted_state_is_not_replayed():
    item = {**NATIVE, "encrypted_content": None}
    blocks = asyncio.run(roundtrip_blocks([item]))
    assert blocks[0]["signature"].startswith("reasoning:")
    assert payload(blocks)["input"][1:] == [{"role": "assistant", "content": "Answer"}]


def test_replayable_reasoning_is_a_retry_barrier():
    async def run():
        state = TranslatorState()
        frames = [frame async for frame in responses_to_anthropic_sse(
            events(native_events(NATIVE) + [DONE]), "test", state=state, reasoning_codec=CODEC,
        )]
        assert frames and state.reasoning_committed
        assert phase_from_state(state) == RetryPhase.TEXT
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["sse", "lite_ws"])
def test_adapter_final_payload_replays_native_state_and_stable_prefix(monkeypatch, mode):
    """Exercise base.call translation, adapter shaping and final transport body."""
    monkeypatch.setattr("claude_code_bridge.proxy.usage_capture.schedule_capture", lambda _: None)
    async def run():
        adapter = OpenAIChatGPTAdapter()
        adapter._cached_account_id = "account-a"
        adapter.authorize = AsyncMock(return_value=("token", BASE))
        adapter.start = AsyncMock()
        adapter._responses_client = NS(with_options=lambda **_: object())
        context = ProxyRequestContext(request_id="req", provider="openai_chatgpt",
                                      conversation_id="thread", responses_transport=mode)
        calls = []
        async def open_stream(**kwargs):
            body = kwargs["body"]
            calls.append(lite_request(body, "thread") if mode == "lite_ws" else copy.deepcopy(body))
            source = native_events(NATIVE) + [
                {"type": "response.output_item.added", "item": {
                    "type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "check"}},
                {"type": "response.output_item.done", "item": {
                    "type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "check", "arguments": "{}"}},
                DONE,
            ]
            return events(source)
        adapter.open_stream = open_stream
        body = {"model": "openai_chatgpt/" + MODEL, "system": "Stable instructions", "messages": [
            {"role": "user", "content": "Question"},
        ]}
        frames = [frame async for frame in adapter.call(body, MODEL, context)]
        blocks = await parse_sdk_message(frames)
        body["messages"].extend([
            {"role": "assistant", "content": blocks},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "OK"}]},
        ])
        _ = [frame async for frame in adapter.call(body, MODEL, context)]
        body["messages"].extend([
            {"role": "assistant", "content": "Finished"},
            {"role": "user", "content": "Next question"},
        ])
        _ = [frame async for frame in adapter.call(body, MODEL, context)]
        assert NATIVE in calls[1]["input"]
        assert calls[2]["input"][:len(calls[1]["input"])] == calls[1]["input"]
        assert calls[1]["input"][:len(calls[0]["input"])] == calls[0]["input"]
        assert calls[1]["store"] is False
        assert calls[1]["include"] == ["reasoning.encrypted_content"]
        assert calls[1]["reasoning"].get("context") == ("all_turns" if mode == "lite_ws" else None)
        assert "previous_response_id" not in calls[1]
        if mode == "lite_ws":
            assert calls[1]["parallel_tool_calls"] is False
    asyncio.run(run())


def test_real_openai_sdk_serializes_replayed_item_on_standard_http(monkeypatch):
    from openai import AsyncOpenAI

    monkeypatch.setattr("claude_code_bridge.proxy.usage_capture.schedule_capture", lambda _: None)
    async def run():
        captured = []
        def handler(request):
            captured.append(json.loads(request.content))
            wire = "data: " + json.dumps(DONE) + "\n\ndata: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=wire)
        adapter = OpenAIChatGPTAdapter()
        adapter._cached_account_id = "account-a"
        adapter.authorize = AsyncMock(return_value=("fixture", BASE))
        adapter.start = AsyncMock()
        adapter._responses_client = AsyncOpenAI(
            api_key="fixture", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        body = {"messages": [{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "summary", "signature": CODEC.encode(NATIVE)},
            {"type": "text", "text": "Previous answer"},
        ]}, {"role": "user", "content": "Next question"}]}
        try:
            frames = [frame async for frame in adapter.call(body, MODEL)]
            assert b'"type":"message_stop"' in b"".join(frames)
        finally:
            await adapter.close()
        assert len(captured) == 1
        assert captured[0]["input"][0] == NATIVE
        assert captured[0]["store"] is False
        assert "context" not in captured[0]["reasoning"]
    asyncio.run(run())


def test_missing_account_identity_disables_replay():
    assert OpenAIChatGPTAdapter().reasoning_codec(MODEL, BASE) is None


def test_malformed_encrypted_upstream_item_is_not_silently_lost():
    with pytest.raises(ValueError, match="Malformed native"):
        CODEC.encode({"type": "reasoning", "encrypted_content": "opaque"})


def test_committed_reasoning_failure_does_not_splice_another_attempt(monkeypatch):
    monkeypatch.setattr("claude_code_bridge.proxy.usage_capture.schedule_capture", lambda _: None)
    async def run():
        adapter = OpenAIChatGPTAdapter()
        adapter._cached_account_id = "account-a"
        adapter.authorize = AsyncMock(return_value=("token", BASE))
        adapter.start = AsyncMock()
        adapter._responses_client = NS(with_options=lambda **_: object())
        async def upstream():
            async for event in events(native_events(NATIVE)):
                yield event
            raise ConnectionError("fixture disconnected after native state")
        adapter.open_stream = AsyncMock(side_effect=lambda **_: upstream())
        frames = [frame async for frame in adapter.call({"messages": []}, MODEL)]
        assert adapter.open_stream.await_count == 1
        assert b'"type":"api_error"' in b"".join(frames)
    asyncio.run(run())


@pytest.mark.parametrize("changed_account", [False, True])
def test_auth_refresh_preserves_or_rejects_reasoning_scope(monkeypatch, changed_account):
    monkeypatch.setattr("claude_code_bridge.proxy.usage_capture.schedule_capture", lambda _: None)
    monkeypatch.setattr("claude_code_bridge.proxy.retry.compute_backoff", lambda *a, **k: 0)
    async def run():
        adapter = OpenAIChatGPTAdapter()
        adapter.start = AsyncMock()
        adapter._responses_client = NS(with_options=lambda **_: object())
        authorizations = 0
        async def authorize():
            nonlocal authorizations
            authorizations += 1
            adapter._cached_account_id = "account-b" if changed_account and authorizations > 1 else "account-a"
            return "fresh-token", BASE
        adapter.authorize = authorize
        count = 0
        async def open_stream(**kwargs):
            nonlocal count
            count += 1
            if count == 1:
                raise httpx.HTTPStatusError("expired", request=httpx.Request("POST", BASE),
                                            response=httpx.Response(401))
            return events([DONE])
        adapter.open_stream = open_stream
        frames = [frame async for frame in adapter.call({"messages": []}, MODEL)]
        assert authorizations == 2
        assert count == (1 if changed_account else 2)
        assert (b'Reasoning account scope changed' in b"".join(frames)) == changed_account
    asyncio.run(run())
