"""TASK-1915: bounded, offline observations at raw/SDK/export boundaries."""
import asyncio
import gzip
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

from openai import APIError
import httpx
import pytest

from claude_code_bridge.proxy import chatgpt_transport
from claude_code_bridge.proxy.responses_progress import ResponsesProgress, _CAP
from claude_code_bridge.proxy.responses_supervisor import ResponsesSSEStream
from tests import virtual_clock
from tests.test_proxy_progress_recovery import Harness, success_frames
from tests.test_proxy_sol_progress_boundary import Wire, decoded, offline_client, terminal


def frame(event, *, event_name=None):
    return ("event: " + (event_name or event["type"]) + "\ndata: "
            + json.dumps(event, ensure_ascii=False) + "\n\n").encode()


def reports(caplog):
    return [(r.args[1], r.args[3], json.loads(r.args[4])) for r in caplog.records
            if r.name.endswith("responses_progress") and r.msg.startswith("responses_progress ")]


def trace():
    return ResponsesProgress(0, NS(request_id="fixture"), 1, clock=lambda: 2.5)


class Bytes(httpx.AsyncByteStream):
    def __init__(self, chunks, error=None):
        self.chunks, self.error, self.closed = chunks, error, False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize("compressed", [False, True])
def test_http_tap_preserves_decompressed_chunks_and_counts_before_sdk(compressed):
    raw = frame({"type": "response.output_text.delta", "delta": "private café 🐾"})
    body = gzip.compress(raw) if compressed else raw
    chunks = [body[i:i+7] for i in range(0, len(body), 7)]

    async def read(observe):
        wire = Bytes(chunks)
        response = httpx.Response(200, stream=wire, headers=(
            {"content-encoding": "gzip"} if compressed else {}))
        tap = trace()
        if observe:
            tap.attach_http(NS(response=response))
        output = [chunk async for chunk in response.aiter_bytes()]
        await response.aclose()
        assert wire.closed
        return output, tap

    baseline, _ = asyncio.run(read(False))
    observed, tap = asyncio.run(read(True))
    assert observed == baseline  # identical chunk boundaries, not just joined bytes
    assert b"".join(observed) == raw
    if not compressed:
        assert all(a is b for a, b in zip(observed, chunks))
    assert tap.raw_attached and tap.body_bytes == len(raw)
    assert tap.body_chunks == len(observed)
    assert tap.first_body_ms == tap.last_body_ms == 2500
    assert tap.counts["raw"] == {"text_delta": 1}
    assert tap.counts["decoded"] == {}


def test_real_sdk_rejection_is_distinguishable_from_raw_receipt(caplog):
    async def run():
        # The SDK raises for this error before yielding any decoded event. The
        # raw tap sees the family and comment without swallowing the SDK error.
        wire = Bytes([
            b": private comment\n\n",
            frame({"type": "error", "error": {"message": "fixture rejected event"}}),
        ])
        async with offline_client() as client:
            async def opener():
                return decoded(wire, client)
            stream = ResponsesSSEStream(opener)
            try:
                with pytest.raises(APIError):
                    _ = [event async for event in stream]
            finally:
                await stream.close()
            assert wire.closed
            assert stream.progress.counts["raw"] == {
                "comment": 1, "terminal": 1,
            }
            assert stream.progress.counts["decoded"] == {}
            assert stream.progress.counts["exported"] == {}

    with caplog.at_level("INFO"):
        virtual_clock.run(run(), follow=[chatgpt_transport])
    assert len(reports(caplog)) == 1
    assert "private" not in caplog.text


def test_real_adapter_pairs_native_decode_productivity_and_export(monkeypatch, caplog):
    async def run():
        wire = Wire([
            (0, {"type": "response.reasoning_summary_text.delta", "delta": "private heading"}),
            (80, {"type": "response.output_item.done", "item": {
                "type": "reasoning", "id": "private-id", "summary": [],
                "encrypted_content": "private native state", "status": "completed",
            }}),
            (20, {"type": "response.output_text.delta", "delta": "private answer"}),
            (0, terminal()),
        ])
        async with offline_client() as client:
            h = Harness(monkeypatch, "sse", [], [])
            h.create.side_effect = [decoded(wire, client)]
            try:
                await h.collect()
                assert h.attempts == 1 and wire.closed
                assert [s["state"] for s in h.statuses] == ["stalled", "recovered"]
                assert asyncio.get_running_loop().time() == 100
            finally:
                await h.adapter.close()

    with caplog.at_level("INFO"):
        virtual_clock.run(run(), follow=[chatgpt_transport])
    rows = reports(caplog)
    assert [(a, trigger) for a, trigger, _ in rows] == [(1, "stall"), (1, "close")]
    summary = rows[-1][2]
    for boundary in ("raw", "decoded"):
        assert summary["counts"][boundary]["reasoning_native"] == 1
        assert summary["counts"][boundary]["reasoning_summary_delta"] == 1
        assert summary["last_ms"][boundary]["reasoning_native"] == 80000
    assert summary["counts"]["exported"]["signature"] == 1
    assert summary["counts"]["exported"]["thinking"] >= 1
    assert summary["counts"]["exported"]["text"] == 1
    assert summary["last_ms"]["exported"]["signature"] == 80000
    assert summary["productive"] == 4 and summary["nonproductive"] == 0
    assert summary["last_productive_ms"] == summary["last_body_ms"] == 100000
    assert summary["raw_attached"] and summary["complete"]
    assert "private" not in caplog.text


@pytest.mark.parametrize("disconnect", [False, True])
def test_real_sdk_retry_summaries_are_attempt_local(monkeypatch, caplog, disconnect):
    async def run():
        heading = {"type": "response.reasoning_summary_text.delta", "delta": "private heading"}
        first = (Bytes([frame(heading)], httpx.ReadError("fixture disconnect")) if disconnect
                 else Wire([(0, heading)]))
        second = Wire([(0, {"type": "response.output_text.delta", "delta": "private answer"}),
                       (0, terminal())])
        async with offline_client() as client:
            h = Harness(monkeypatch, "sse", [], [])
            h.create.side_effect = [decoded(first, client), decoded(second, client)]
            try:
                await h.collect()
                assert h.attempts == 2 and first.closed and second.closed
                assert b"event: message_stop" in b"".join(h.frames)
                assert asyncio.get_running_loop().time() == (0 if disconnect else 90)
            finally:
                await h.adapter.close()

    with caplog.at_level("INFO"):
        virtual_clock.run(run(), follow=[chatgpt_transport])
    closed = [(a, s) for a, trigger, s in reports(caplog) if trigger == "close"]
    assert [a for a, _ in closed] == [1, 2]
    abandoned, completed = [s for _, s in closed]
    assert abandoned["counts"]["decoded"] == {"reasoning_summary_delta": 1}
    assert "text" not in abandoned["counts"]["exported"]
    assert not abandoned["complete"]
    assert completed["counts"]["decoded"] == {"text_delta": 1, "terminal": 1}
    assert "thinking" not in completed["counts"]["exported"]
    assert completed["complete"]
    assert completed["elapsed_ms"] == 0  # retry has its own timing origin
    assert len(reports(caplog)) == (2 if disconnect else 3)


def test_downstream_cancel_closes_tap_without_retry(monkeypatch, caplog):
    async def run():
        wire = Wire([(0, {"type": "response.reasoning_text.delta", "delta": "private heading"})])
        async with offline_client() as client:
            h = Harness(monkeypatch, "sse", [], [])
            h.create.side_effect = [decoded(wire, client)]
            pending = asyncio.create_task(h.collect())
            try:
                await h.thinking_forwarded.wait()
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
                assert h.attempts == 1 and wire.closed
            finally:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
                await h.adapter.close()

    with caplog.at_level("INFO"):
        virtual_clock.run(run(), follow=[chatgpt_transport])
    rows = reports(caplog)
    assert [(a, trigger) for a, trigger, _ in rows] == [(1, "close")]
    assert not rows[0][2]["complete"]


@pytest.mark.parametrize("separator", [b"\n", b"\r\n"])
@pytest.mark.parametrize("chunk_size", [1, 7, 200000])
def test_fixed_labels_oversize_resync_and_privacy(separator, chunk_size, caplog):
    secret = "DO_NOT_LOG_fixture_state_name_args_header"
    tap = trace()
    raw = (b": " + secret.encode() + b"\n\ndata: {bad\n\ndata: \xff\n\n"
           + frame({"type": "response.output_item.done", "item": {
               "type": "reasoning", "encrypted_content": secret,
           }})
           + frame({"type": "response.output_text.delta", "delta": secret * 4000})
           + frame({"type": "response.function_call_arguments.delta", "delta": secret})
           + frame({"type": "response.output_text.delta", "delta": "café 🐾"})
           + b'data: {"type":"response.in_progress",\ndata: "extra":true}\n\n'
           + b"".join(frame({"type": secret + str(i)}) for i in range(200)))
    raw = raw.replace(b"\n", separator)
    for start in range(0, len(raw), chunk_size):
        tap.feed("raw", raw[start:start + chunk_size])
        assert len(tap._raw._line) <= 65536
        assert sum(map(len, tap._raw._data)) <= 65536
    native = NS(type="response.output_item.done", item=NS(type="reasoning", encrypted_content=secret))
    tap.decoded(native, True)
    tap.feed("exported", frame({"type": "content_block_delta", "delta": {
        "type": "signature_delta", "signature": secret,
    }}))
    tap.feed("exported", frame({"type": "content_block_start", "content_block": {
        "type": "tool_use", "id": secret, "name": secret, "input": {secret: secret},
    }}))
    # Unfinished content must not survive close in either parser.
    tap.feed("raw", b"data: " + secret.encode())
    tap.feed("exported", b"data: " + secret.encode())
    with caplog.at_level("INFO"):
        for _ in range(5):
            tap.report("stall", "sse")
            tap.report("close", "sse")
    assert tap.counts["raw"] == {
        "comment": 1, "malformed": 2, "reasoning_native": 1, "oversize": 1,
        "tool_delta": 1, "text_delta": 1, "metadata": 1, "unknown": 200,
    }
    assert tap.counts["exported"] == {"signature": 1, "tool_header": 1}
    assert len(reports(caplog)) == 2 and len(caplog.text) < 6000
    assert reports(caplog)[-1][2]["partial"] == {"raw": True, "exported": True}
    assert secret not in caplog.text and secret not in repr(tap.__dict__)
    assert not tap._raw._line and not tap._raw._data
    assert not tap._exported._line and not tap._exported._data


def test_observer_and_logger_failure_never_changes_bytes_or_exception(monkeypatch):
    tap = trace()
    secret = "private diagnostic exception"

    def fail(*args, **kwargs):
        raise ValueError(secret)

    monkeypatch.setattr(tap._raw, "_observe", fail)
    monkeypatch.setattr("claude_code_bridge.proxy.responses_progress.logger.info", fail)
    chunk = frame({"type": "response.output_text.delta", "delta": "answer"})
    expected_error = httpx.ReadError("fixture disconnect")

    async def run():
        wire = Bytes([chunk], expected_error)
        response = httpx.Response(200, stream=wire)
        tap.attach_http(NS(response=response))
        iterator = response.aiter_bytes()
        assert await anext(iterator) is chunk
        with pytest.raises(httpx.ReadError) as error:
            await anext(iterator)
        assert error.value is expected_error
        await response.aclose()
        tap.report("close", "sse")
        assert wire.closed

    asyncio.run(run())
    assert tap.observer_errors == 2
    assert not tap._raw._line and not tap._raw._data


def test_missing_raw_hook_is_not_reported_as_observed_zero(caplog):
    tap = trace()
    tap.attach_http(NS(response=NS(headers={})))
    tap.decoded(NS(type="response.in_progress"), False)
    with caplog.at_level("INFO"):
        tap.report("close", "sse")
    summary = reports(caplog)[0][2]
    assert not summary["raw_attached"]
    assert summary["body_bytes"] == 0 and summary["last_body_ms"] is None
    assert summary["counts"]["decoded"] == {"metadata": 1}


def test_unknown_event_types_do_not_leak_into_stall_diagnostics(caplog):
    async def run():
        wire = Wire([(10, {"type": "PRIVATE_UNTRUSTED_EVENT_TYPE"}) for _ in range(10)])
        async with offline_client() as client:
            async def opener():
                return decoded(wire, client)
            stream = ResponsesSSEStream(opener)
            try:
                with pytest.raises(chatgpt_transport.ChatGPTEventTimeout):
                    _ = [event async for event in stream]
                assert asyncio.get_running_loop().time() == 60
            finally:
                await stream.close()
            assert stream.last_event_kind == "unknown"
            assert set(stream.idle_event_counts) == {"unknown"}
            assert stream.progress.counts["raw"]["unknown"] > 0
            assert stream.progress.counts["decoded"]["unknown"] > 0

    with caplog.at_level("INFO"):
        virtual_clock.run(run(), follow=[chatgpt_transport])
    assert "PRIVATE_UNTRUSTED_EVENT_TYPE" not in caplog.text


def test_ws_transport_observes_existing_json_and_tool_export(monkeypatch, caplog):
    async def run():
        h = Harness(monkeypatch, "lite_ws", success_frames(), [])
        try:
            await h.collect()
            assert h.attempts == 1
        finally:
            await h.adapter.close()
        assert h.feeds[0].close.await_count == 1

    with caplog.at_level("INFO"):
        virtual_clock.run(run(), follow=[chatgpt_transport])
    rows = reports(caplog)
    assert [(a, t) for a, t, _ in rows] == [(1, "close")]
    summary = rows[0][2]
    assert summary["counts"]["raw"] == summary["counts"]["decoded"]
    assert summary["counts"]["raw"]["reasoning_native"] == 1
    assert summary["counts"]["exported"]["tool_header"] == 1
    assert summary["counts"]["exported"]["tool_arguments"] == 1
    assert summary["ws_chars"] > 0 and summary["body_bytes"] == 0
    assert summary["raw_attached"] and summary["complete"]


@pytest.mark.parametrize("raw", ["{broken", '{"type":[]}'])
def test_ws_malformed_body_stays_a_failure_but_is_counted(raw):
    async def run():
        stream = chatgpt_transport.ChatGPTStream(
            NS(), None, NS(socket=NS(recv=AsyncMock(return_value=raw))),
        )
        with pytest.raises(ValueError):
            await stream._receive()
        assert stream.progress.counts["raw"] == {"malformed": 1}
        assert stream.progress.counts["decoded"] == {}
        assert stream.progress.raw_attached

    asyncio.run(run())


@pytest.mark.parametrize("raw", [b"fixture body", "fixture text"])
def test_ws_reuses_existing_json_and_counters_saturate(raw, caplog):
    tap = trace()
    tap.body_bytes = tap.body_chunks = tap.ws_chars = _CAP
    tap.productive = tap.nonproductive = _CAP
    tap.counts["raw"]["reasoning_native"] = _CAP
    payload = {"type": "response.output_item.done", "item": {
        "type": "reasoning", "encrypted_content": "private opaque state",
    }}
    tap.received_ws(raw, payload)
    tap.decoded(payload, True)
    tap.decoded({"type": "response.in_progress"}, False)
    with caplog.at_level("INFO"):
        tap.report("close", "websocket")
    summary = reports(caplog)[0][2]
    assert summary["raw_attached"] and summary["counts"]["raw"]["reasoning_native"] == _CAP
    assert summary["body_bytes"] == summary["body_chunks"] == summary["ws_chars"] == _CAP
    assert summary["productive"] == summary["nonproductive"] == _CAP
    assert "private" not in caplog.text
