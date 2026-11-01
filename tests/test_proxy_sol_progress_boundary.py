"""TASK-1915: raw offline Responses bytes through SDK decode, clock and export."""
import asyncio
import json
from types import SimpleNamespace as NS

import httpx
from openai import AsyncOpenAI, AsyncStream
from openai.types.responses import ResponseStreamEvent
import pytest

from claude_code_bridge.proxy import chatgpt_transport
from claude_code_bridge.proxy.responses_supervisor import ResponsesSSEStream
from claude_code_bridge.proxy.reasoning import ReasoningCodec
from claude_code_bridge.proxy.stream import TranslatorState, responses_to_anthropic_sse
from tests import virtual_clock


class Wire(httpx.AsyncByteStream):
    """No socket or request: delays and partial UTF-8/SSE reads are synthetic."""

    def __init__(self, script):
        self.script = script
        self.closed = False

    async def __aiter__(self):
        for delay, payload in self.script:
            await asyncio.sleep(delay)
            # Comments are transport traffic, not generation. The real SDK SSE
            # decoder discards these, before the supervisor observes any event.
            yield b": fixture keepalive\n\n"
            raw = ("event: " + payload["type"] + "\ndata: "
                   + json.dumps(payload, ensure_ascii=False) + "\n\n").encode()
            for start in range(0, len(raw), 7):
                yield raw[start:start + 7]
        await asyncio.Event().wait()  # the response, not transport EOF, ends a turn

    async def aclose(self):
        self.closed = True


def offline_client():
    transport = httpx.MockTransport(lambda _: pytest.fail("fixture must not issue HTTP requests"))
    return AsyncOpenAI(api_key="fixture", http_client=httpx.AsyncClient(transport=transport))


def decoded(wire, client):
    response = httpx.Response(
        200, request=httpx.Request("POST", "https://fixture.invalid/responses"),
        headers={"content-type": "text/event-stream"}, stream=wire,
    )
    return AsyncStream(cast_to=ResponseStreamEvent, response=response, client=client)


def context(statuses):
    return NS(attempt=1, provider="fixture", first_output_ms=None,
              report_status=statuses.append)


def terminal():
    return {"type": "response.completed", "response": {
        "status": "completed", "usage": None,
    }}


def frames(chunks):
    return [json.loads(line[6:]) for line in b"".join(chunks).splitlines()
            if line.startswith(b"data: ")]


async def export(script):
    wire = Wire(script)
    statuses = []
    state = TranslatorState()
    # Construct the same SDK stream used by responses.create without making
    # any HTTP call, even to a localhost provider. Decode real raw SSE bytes.
    async with offline_client() as client:
        async def opener():
            return decoded(wire, client)
        stream = ResponsesSSEStream(opener, context=context(statuses))
        try:
            chunks = [chunk async for chunk in responses_to_anthropic_sse(
                stream, anthropic_model="fixture", state=state,
                reasoning_codec=ReasoningCodec.for_account(
                    "openai_chatgpt", "fixture", "https://fixture.invalid", "fixture-account",
                ),
            )]
            return frames(chunks), statuses, stream, state
        finally:
            await stream.close()
            assert wire.closed


@pytest.mark.parametrize("tool_kind", ["function_call", "custom_tool_call"])
def test_raw_sdk_reasoning_text_and_tool_progress_reaches_export(tool_kind):
    async def run():
        tool = {"type": tool_kind, "id": "fixture-tool", "call_id": "fixture-call",
                "name": "fixture_check"}
        argument_kind = ("function_call_arguments" if tool_kind == "function_call"
                         else "custom_tool_call_input")
        completed_tool = {**tool, "arguments": '{"value":"ok"}', "input": "ok"}
        script = [
            (0, {"type": "response.created"}),
            (0, {"type": "response.output_item.added", "item": {
                "type": "reasoning", "id": "fixture-reasoning", "summary": [],
            }}),
            (0, {"type": "response.reasoning_summary_text.delta", "delta": "Heading"}),
            (20, {"type": "response.reasoning_text.delta", "delta": " β"}),
            (20, {"type": "response.reasoning_summary_text.delta", "delta": " tail"}),
            (10, {"type": "response.output_item.done", "item": {
                "type": "reasoning", "id": "fixture-reasoning", "summary": [],
                "encrypted_content": "opaque-fixture-state", "status": "completed",
            }}),
            (10, {"type": "response.output_text.delta", "delta": "Ready"}),
            (10, {"type": "response.output_item.added", "output_index": 2, "item": tool}),
            (20, {"type": f"response.{argument_kind}.delta", "output_index": 2,
                  "item_id": "fixture-tool", "delta": '{"value":"ok"}'}),
            (20, {"type": "response.output_item.done", "output_index": 2,
                  "item": completed_tool}),
            (0, terminal()),
        ]
        output, statuses, stream, state = await export(script)
        deltas = [frame["delta"] for frame in output if frame["type"] == "content_block_delta"]
        assert "".join(d.get("thinking", "") for d in deltas) == "Heading β tail"
        assert [d["text"] for d in deltas if d["type"] == "text_delta"] == ["Ready"]
        assert any(d["type"] == "signature_delta" for d in deltas)
        assert any(d["type"] == "input_json_delta" for d in deltas)
        assert output[-1]["type"] == "message_stop"
        assert not statuses
        assert state.tool_committed
        assert stream.last_productive_activity_at == 110
        assert asyncio.get_running_loop().time() == 110

    virtual_clock.run(run(), follow=[chatgpt_transport])


@pytest.mark.parametrize("heading", [False, True])
def test_completed_native_reasoning_renews_productive_clock_before_text(heading):
    """A completed native state is real progress, not a metadata heartbeat."""
    async def run():
        script = [(0, {"type": "response.output_item.added", "item": {
            "type": "reasoning", "id": "fixture-reasoning", "summary": [],
        }})]
        if heading:
            script.append((0, {"type": "response.reasoning_summary_text.delta",
                               "delta": "Reading fixture details"}))
        script.extend([
            (80, {"type": "response.output_item.done", "item": {
                "type": "reasoning", "id": "fixture-reasoning", "summary": [],
                "encrypted_content": "opaque-fixture-state", "status": "completed",
            }}),
            (20, {"type": "response.output_text.delta", "delta": "Ready"}),
            (0, terminal()),
        ])
        output, statuses, stream, state = await export(script)
        assert [s["state"] for s in statuses] == ["stalled", "recovered"]
        assert output[-1]["type"] == "message_stop"
        assert stream.last_productive_activity_at == 100
        assert any(frame.get("delta", {}).get("type") == "signature_delta" for frame in output)
        assert state.reasoning_committed
        assert not state.tool_committed
        assert asyncio.get_running_loop().time() == 100

    virtual_clock.run(run(), follow=[chatgpt_transport])


@pytest.mark.parametrize("native", [None, "", "   "])
def test_empty_completed_reasoning_marker_is_not_productive(native):
    async def run():
        wire = Wire([
            (0, {"type": "response.reasoning_summary_text.delta", "delta": "Heading"}),
            (80, {"type": "response.output_item.done", "item": {
                "type": "reasoning", "id": "fixture-reasoning", "summary": [],
                "encrypted_content": native, "status": "completed",
            }}),
        ])
        statuses = []
        async with offline_client() as client:
            async def opener():
                return decoded(wire, client)
            stream = ResponsesSSEStream(opener, context=context(statuses))
            try:
                with pytest.raises(chatgpt_transport.ChatGPTEventTimeout) as error:
                    _ = [event async for event in stream]
                assert error.value.phase == "productive"
                assert error.value.productive_idle_seconds == 90
                assert asyncio.get_running_loop().time() == 90
                assert [s["state"] for s in statuses] == ["stalled"]
            finally:
                await stream.close()

    virtual_clock.run(run(), follow=[chatgpt_transport])


@pytest.mark.parametrize("item_mapping", [False, True])
def test_completed_native_reasoning_classification_matches_ws_and_sdk_shapes(item_mapping):
    item = {"type": "reasoning", "encrypted_content": "opaque-fixture-state"}
    event = NS(type="response.output_item.done", item=item if item_mapping else NS(**item))
    assert chatgpt_transport.ChatGPTStream._is_productive(event)
    event.type = "response.output_item.added"
    assert not chatgpt_transport.ChatGPTStream._is_productive(event)


@pytest.mark.parametrize("heading, phase, seconds", [
    (False, "first_output", 60), (True, "productive", 90),
])
def test_raw_sdk_metadata_only_keeps_incident_watchdogs(heading, phase, seconds):
    async def run():
        wire = Wire([
            (0, {"type": "response.created"}),
            (0, {"type": "response.in_progress"}),
            *([(0, {"type": "response.reasoning_summary_text.delta", "delta": "Heading"})]
              if heading else []),
            *[(10, {"type": "response.in_progress"}) for _ in range(10)],
        ])
        statuses = []
        async with offline_client() as client:
            async def opener():
                return decoded(wire, client)
            stream = ResponsesSSEStream(opener, context=context(statuses))
            try:
                with pytest.raises(chatgpt_transport.ChatGPTEventTimeout) as error:
                    _ = [event async for event in stream]
                assert error.value.phase == phase
                assert error.value.productive_idle_seconds == seconds
                assert asyncio.get_running_loop().time() == seconds
                assert [s["state"] for s in statuses] == ["stalled"]
                assert statuses[0]["productive_idle_seconds"] == 30
            finally:
                await stream.close()
        assert wire.closed

    virtual_clock.run(run(), follow=[chatgpt_transport])
