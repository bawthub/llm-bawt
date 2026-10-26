"""TASK-1079: stalls are detected early, shown honestly, and never run ``{}``."""
import asyncio
import json
from types import SimpleNamespace as NS

import pytest
from anthropic import APIError

from claude_code_bridge.proxy import chatgpt_transport
from claude_code_bridge.proxy.chatgpt_transport import (
    ChatGPTEventTimeout, DEFAULT_FIRST_OUTPUT_TIMEOUT, DEFAULT_PRODUCTIVE_IDLE_TIMEOUT,
    DEFAULT_STALL_NOTICE_AFTER,
)
from claude_code_bridge.proxy.responses_supervisor import ResponsesSSEStream
from tests.test_proxy_progress_recovery import THINKING, TOOL, Harness, parse_message, success_frames
from tests import virtual_clock
from tests.test_turn_phase import _phases, _ProxyHarness, _tracker


class HTTP:
    response = NS(headers={})

    def __init__(self, events):
        self.events = events

    def __aiter__(self):
        return self.events

    async def close(self):
        pass


def _stream(events, statuses=None, **limits):
    async def opener():
        return HTTP(events)
    context = None
    if statuses is not None:
        context = NS(attempt=1, provider="openai_chatgpt", report_status=statuses.append)
    stream = ResponsesSSEStream(opener, context=context,
                                productive_idle_timeout=limits.pop("productive", DEFAULT_PRODUCTIVE_IDLE_TIMEOUT))
    for name, value in limits.items():
        setattr(stream.owner, name, value)
    return stream


async def _drain(stream):
    try:
        return [event async for event in stream]
    finally:
        await stream.close()


async def _chatter(*prefix, item=NS(type="response.in_progress")):
    for event in prefix:
        yield event
    while True:
        await asyncio.sleep(.002)
        yield item


def test_request_with_no_output_item_times_out_as_first_output():
    async def run():
        stream = _stream(_chatter(NS(type="response.created")), first_output_timeout=.05)
        with pytest.raises(ChatGPTEventTimeout) as error:
            await _drain(stream)
        assert error.value.phase == "first_output"
    virtual_clock.run_timing(run())


def test_first_output_telemetry_measures_item_not_headers():
    async def run():
        async def events():
            yield NS(type="response.created")  # headers/created are not model output
            await asyncio.sleep(.04)
            yield NS(type="response.output_item.added", item=NS(type="reasoning"))
            yield NS(type="response.completed", response=NS(usage=None, status="completed"))

        stream = _stream(events(), [], productive=.1)
        await _drain(stream)
        assert stream.context.first_output_ms == pytest.approx(40)

    virtual_clock.run(run(), follow=[chatgpt_transport])


def test_hidden_reasoning_item_satisfies_first_output():
    async def run():
        reasoning = NS(type="response.output_item.added", item=NS(type="reasoning"))
        async def events():
            yield reasoning
            await asyncio.Event().wait()
        stream = _stream(events(), first_output_timeout=DEFAULT_FIRST_OUTPUT_TIMEOUT)
        with pytest.raises(ChatGPTEventTimeout) as error:
            await _drain(stream)
        assert error.value.phase == "productive"
    virtual_clock.run_timing(run())


def test_whitespace_argument_runaway_times_out_before_productive_idle():
    async def run():
        added = NS(type="response.output_item.added", item=NS(type="function_call"))
        blank = NS(type="response.function_call_arguments.delta", delta="  \n")
        loop = asyncio.get_running_loop()
        started = loop.time()
        stream = _stream(_chatter(added, item=blank), whitespace_runaway_timeout=.05)
        with pytest.raises(ChatGPTEventTimeout) as error:
            await _drain(stream)
        assert error.value.phase == "whitespace"
        # The runaway clock starts at the first blank delta, one chatter step in.
        assert loop.time() - started == pytest.approx(.002 + .05)
    virtual_clock.run(run(), follow=[chatgpt_transport])


def test_stall_notice_reports_once_then_recovers_on_the_same_stream():
    async def run():
        async def events():
            yield NS(type="response.created")
            await asyncio.sleep(.12)  # the notice must not cancel this wait
            yield NS(type="response.output_text.delta", delta="back")
            yield NS(type="response.completed", response=NS(usage=None, status="completed"))

        statuses = []
        before = chatgpt_transport.time.time()  # the stream's own (virtual) wall clock
        stream = _stream(events(), statuses, stall_notice_after=.03)
        kinds = [event.type for event in await _drain(stream)]
        assert kinds[-1] == "response.completed"
        assert [s["state"] for s in statuses] == ["stalled", "recovered"]
        stalled = statuses[0]
        assert stalled["message"] == "Waiting on model; no output yet"
        # The silence began when the stream opened, not when the notice fired.
        assert stalled["since"] == pytest.approx(before)
    virtual_clock.run(run(), follow=[chatgpt_transport])


def test_stall_notice_waits_for_header_timeout_rather_than_failing():
    async def run():
        statuses = []

        async def slow_opener():
            # Slow headers outlast the production notice but arrive well before
            # the production first-event deadline on either scheduler speed.
            await asyncio.sleep(DEFAULT_STALL_NOTICE_AFTER + 10)
            return HTTP(_chatter(NS(type="response.output_text.delta", delta="ok"),
                                 item=NS(type="response.completed",
                                         response=NS(usage=None, status="completed"))))

        stream = ResponsesSSEStream(
            slow_opener, context=NS(attempt=1, provider="p", report_status=statuses.append),
        )
        events = await _drain(stream)
        assert events[0].delta == "ok"
        assert [s["state"] for s in statuses] == ["stalled", "recovered"]
    virtual_clock.run_timing(run())


def _frames(raw: list[bytes]) -> list[dict]:
    frames = []
    for chunk in b"".join(raw).split(b"\n\n"):
        for line in chunk.splitlines():
            if line.startswith(b"data: "):
                frames.append(json.loads(line[6:]))
    return frames


def _tool_frames(h):
    frames = _frames(h.frames)
    starts = [f["index"] for f in frames if f.get("type") == "content_block_start"
              and f["content_block"]["type"] == "tool_use"]
    stops = {f["index"] for f in frames if f.get("type") == "content_block_stop"}
    return frames, starts, stops


@pytest.mark.parametrize("mode", ["sse", "lite_ws"])
def test_final_failure_leaves_argumentless_tool_block_open(monkeypatch, mode):
    """A closed empty block is a complete ``{}`` call the CLI would execute."""
    async def run():
        blank = {"type": "response.function_call_arguments.delta",
                 "item_id": "fc_new", "delta": "   "}
        h = Harness(monkeypatch, mode, [THINKING, {"type": "response.output_item.added",
                                                  "item": TOOL}, blank, blank],
                    success_frames())
        try:
            await h.collect()
            assert h.attempts == 1  # a forwarded tool is never replayed
            frames, starts, stops = _tool_frames(h)
            assert len(starts) == 1 and starts[0] not in stops
            assert any(f.get("type") == "error" for f in frames)
            assert not any(f.get("type") == "message_stop" for f in frames)
        finally:
            await h.adapter.close()
    virtual_clock.run_timing(run())


def test_unflushed_tool_failure_is_rejected_by_the_anthropic_sdk(monkeypatch):
    async def run():
        h = Harness(monkeypatch, "sse", [
            THINKING, {"type": "response.output_item.added", "item": TOOL},
            {"type": "response.function_call_arguments.delta", "item_id": "fc_new",
             "delta": "   "},
        ], success_frames())
        try:
            await h.collect()
            with pytest.raises(APIError, match="Proxy stream failed"):
                await parse_message(h.frames)
            frames, starts, stops = _tool_frames(h)
            assert len(starts) == 1 and starts[0] not in stops
            assert not any(f.get("type") == "message_stop" for f in frames)
        finally:
            await h.adapter.close()
    virtual_clock.run_timing(run())


@pytest.mark.parametrize("mode", ["sse", "lite_ws"])
def test_final_failure_reports_terminal_status_without_sdk_resend(monkeypatch, mode):
    async def run():
        h = Harness(monkeypatch, mode, [THINKING], [{"type": "response.created"}])
        try:
            await h.collect()
            assert h.attempts == 2
            assert [s["state"] for s in h.statuses] == [
                "stalled", "reconnecting" if mode == "lite_ws" else "retrying",
                "stalled", "failed",
            ]
            tracker, published, _clock = _tracker()
            tracker.on_turn_start()
            for status in h.statuses:
                tracker.on_proxy_status(status)
            assert _phases(published)[-3:] == [
                "reconnecting" if mode == "lite_ws" else "retrying",
                "stalled", "waiting_model",
            ]
        finally:
            await h.adapter.close()
    virtual_clock.run_timing(run())


@pytest.mark.parametrize("mode", ["sse", "lite_ws"])
def test_final_failure_still_closes_tool_block_with_flushed_arguments(monkeypatch, mode):
    async def run():
        h = Harness(monkeypatch, mode, [
            THINKING, {"type": "response.output_item.added", "item": TOOL},
            {"type": "response.function_call_arguments.delta", "item_id": "fc_new",
             "delta": '{"value":"x"}'},
            {"type": "response.function_call_arguments.done", "item_id": "fc_new"},
        ], success_frames())
        try:
            await h.collect()
            frames, starts, stops = _tool_frames(h)
            assert len(starts) == 1 and starts[0] in stops
            assert any(f.get("type") == "error" for f in frames)
        finally:
            await h.adapter.close()
    virtual_clock.run_timing(run())


# ── turn phase ──────────────────────────────────────────────────────────


def test_proxy_stall_backdates_and_waiting_clears_stale_retry():
    tracker, published, clock = _tracker()
    tracker.on_turn_start()
    tracker.on_proxy_status({"state": "stalled", "message": "Model stopped responding",
                             "since": clock.now - 30})
    assert published[-1][1]["since"] == clock.now - 30
    tracker.on_proxy_status({"state": "retrying", "message": "Model stalled; retrying"})
    # The CLI's non-streaming fallback is a fresh request: no stranded status.
    tracker.on_proxy_status({"state": "waiting"})
    tracker.on_proxy_status({"state": "waiting"})  # outside trouble: no-op

    assert _phases(published) == ["waiting_model", "stalled", "retrying", "waiting_model"]


def test_proxy_trouble_survives_sdk_pings_but_watchdog_stall_does_not():
    tracker, published, _clock = _tracker(stall_after=90)
    tracker.on_turn_start()
    tracker.on_proxy_status({"state": "stalled", "message": "Model stopped responding"})
    tracker.on_sdk_message()  # heartbeat pings keep reaching the SDK
    assert tracker.phase.value == "stalled"
    tracker.on_proxy_status({"state": "recovered"})
    assert tracker.phase.value == "streaming"

    tracker.on_health({"phase": "sdk_idle", "wait_elapsed_seconds": 120.0})
    tracker.on_sdk_message()
    assert _phases(published)[-2:] == ["stalled", "streaming"]


def test_compaction_resumes_after_proxy_recovery():
    tracker, published, _clock = _tracker()
    tracker.on_turn_start()
    tracker.on_system_status({"status": "compacting"})
    tracker.on_proxy_status({"state": "stalled", "message": "Model stopped responding"})
    tracker.on_proxy_status({"state": "retrying", "message": "Model stalled; retrying"})
    tracker.on_proxy_status({"state": "recovered"})

    assert _phases(published) == [
        "waiting_model", "compacting", "stalled", "retrying", "compacting",
    ]


def test_legacy_proxy_path_drops_waiting():
    harness = _ProxyHarness()
    harness.publish_proxy_status("req_1", {"state": "waiting"})
    harness.publish_proxy_status("req_1", {"state": "stalled", "message": "x"})

    assert [e["extra_raw"]["upstream_status"]["state"] for e in harness.events] == ["stalled"]
