"""Hermetic TASK-1940 evidence tests: synthetic strings, tmp_path only, no relay."""
from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace as NS

import pytest

from claude_code_bridge.proxy import relay_trace as trace_mod
from claude_code_bridge.proxy.repetition_breaker import ENV_MIN_SPAN
from claude_code_bridge.proxy.request_context import ProxyRequestContext
from claude_code_bridge.proxy.stream import responses_to_anthropic_sse


@pytest.fixture
def arm(tmp_path, monkeypatch):
    directory = tmp_path / "trace"
    directory.mkdir(mode=0o700)
    monkeypatch.setattr(trace_mod, "_directory", lambda: directory)
    monkeypatch.setattr(trace_mod.time, "time", lambda: 1000.0)

    def create(**overrides):
        value = {"bot_id": "sender", "armed_at": 1000.0, "expires_at": 1060.0, **overrides}
        path = directory / "arm.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        path.chmod(0o600)
        return directory

    create.directory = directory
    return create


def _context(bot="sender"):
    return ProxyRequestContext(request_id="request-test", provider="synthetic", bot_id=bot, attempt=2)


def _item(index=0, name=trace_mod.TOOL_NAME):
    return NS(id=f"item-{index}", call_id=f"call-{index}", type="function_call", name=name)


def _events(*, deltas=(), done='{"message":"safe relay"}', index=0, name=trace_mod.TOOL_NAME):
    item = _item(index, name)
    return [
        NS(type="response.output_item.added", item=item),
        *[NS(type="response.function_call_arguments.delta", item_id=item.id, delta=d) for d in deltas],
        NS(type="response.function_call_arguments.done", item_id=item.id, arguments=done),
        NS(type="response.output_item.done", item=item),
    ]


async def _collect(events, context=None, schemas=None):
    async def upstream():
        for event in [*events, NS(type="response.completed", response=None)]:
            yield event

    return [frame async for frame in responses_to_anthropic_sse(
        upstream(), "synthetic/model", request_context=context, tool_schemas=schemas,
    )]


def _rows(directory):
    return [json.loads(line) for line in (directory / "capture.jsonl").read_text().splitlines()]


def _partials(frames):
    data = [json.loads(frame.decode().split("data: ", 1)[1]) for frame in frames]
    return [p["delta"]["partial_json"] for p in data
            if p.get("delta", {}).get("type") == "input_json_delta"]


def _claim(context=None):
    item = _item()
    return trace_mod.claim_relay_trace(
        context or _context(),
        {"name": item.name, "item_id": item.id, "call_id": item.call_id, "index": 0},
        "msg-test",
    )


def test_default_off_creates_nothing(arm):
    asyncio.run(_collect(_events(), _context()))
    assert list(arm.directory.iterdir()) == []


@pytest.mark.parametrize("context,name", [
    (None, trace_mod.TOOL_NAME), (_context(None), trace_mod.TOOL_NAME),
    (_context("other"), trace_mod.TOOL_NAME), (_context(), "Read"),
    (_context(), "prefix_bots_send_message"),
])
def test_exact_identity_and_tool_filter(arm, context, name):
    directory = arm()
    asyncio.run(_collect(_events(name=name, done='{"message":"unrelated"}'), context))
    assert not (directory / "capture.jsonl").exists()
    assert (directory / "arm.json").exists()


def test_exact_fragments_done_buffer_and_emission(arm):
    directory = arm()
    fragments = ['{ "message": " first', '  second\\tthird\\n', 'snowman ☃ " }', '']
    done = '{ "message" : "Exactfused prose  stays\\tand\\n雪" , "optional": "" }'
    schemas = [{"name": trace_mod.TOOL_NAME, "input_schema": {"required": ["message"]}}]
    frames = asyncio.run(_collect(_events(deltas=fragments, done=done), _context(), schemas))
    rows = _rows(directory)
    assert [r["delta"] for r in rows if r["stage"] == "delta"] == fragments
    before = next(r for r in rows if r["stage"] == "before_selection")
    assert before == {"stage": "before_selection", "arguments": done,
                      "buffered": "".join(fragments), "selected": "done"}
    emitted = next(r["partial_json"] for r in rows if r["stage"] == "translator_partial_json")
    assert _partials(frames) == [emitted]
    assert json.loads(emitted) == {"message": "Exactfused prose  stays\tand\n雪"}
    assert rows[0] == {"stage": "start", "version": 1, "expires_at": 1060.0,
                       "bot_id": "sender", "request_id": "request-test", "attempt": 2,
                       "message_id": rows[0]["message_id"], "item_id": "item-0",
                       "call_id": "call-0", "block_index": 0}
    assert rows[-1] == {"stage": "end", "status": "captured"}
    assert not (directory / "arm.json").exists()
    assert (directory / "capture.jsonl").stat().st_mode & 0o777 == 0o600
    assert sum(1 for f in frames if rows[0]["message_id"].encode() in f) == 1


@pytest.mark.parametrize("done,expected", [(None, "buffer"), ("", "buffer"), (None, "empty_object")])
def test_authoritative_absent_and_empty_fallback(arm, done, expected):
    directory = arm()
    deltas = [] if expected == "empty_object" else ['{ "message": " spaced  text " }']
    asyncio.run(_collect(_events(deltas=deltas, done=done), _context()))
    before = next(r for r in _rows(directory) if r["stage"] == "before_selection")
    assert before["arguments"] == done
    assert before["buffered"] == "".join(deltas)
    assert before["selected"] == expected


def test_only_one_call_across_parallel_items_and_subsequent_requests(arm):
    directory = arm()
    events = _events(name="Read", done='{"message":"excluded tool"}')
    first, second = _item(1), _item(2)
    events += [NS(type="response.output_item.added", item=i) for i in (first, second)]
    events += [NS(type="response.function_call_arguments.delta", item_id=i.id, delta=text)
               for i, text in ((second, '{"message":"excluded second"}'),
                               (first, '{"message":"selected first"}'))]
    events += [NS(type="response.function_call_arguments.done", item_id=i.id,
                  arguments=json.dumps({"message": text}))
               for i, text in ((second, "excluded second"), (first, "selected first"))]
    asyncio.run(_collect(events, _context()))
    original = (directory / "capture.jsonl").read_bytes()
    # Even deliberate arm recreation does not bypass the persistent O_EXCL latch.
    arm()
    asyncio.run(_collect(_events(done='{"message":"excluded later request"}'), _context()))
    assert (directory / "capture.jsonl").read_bytes() == original
    assert b"excluded" not in original
    assert _rows(directory)[0]["call_id"] == "call-1"


@pytest.mark.parametrize("overrides", [
    {"expires_at": 1000}, {"expires_at": 999}, {"expires_at": 1301},
    {"armed_at": 1001}, {"armed_at": 600}, {"expires_at": float("nan")},
    {"expires_at": float("inf")}, {"expires_at": True}, {"armed_at": None},
])
def test_invalid_or_expired_arm_is_inert(arm, overrides):
    directory = arm(**overrides)
    asyncio.run(_collect(_events(), _context()))
    assert not (directory / "capture.jsonl").exists()


@pytest.mark.parametrize("case", ["directory_mode", "arm_mode", "arm_symlink", "directory_symlink",
                                  "artifact_symlink", "arm_hardlink", "invalid_json", "oversized"])
def test_unsafe_files_fail_closed_without_output_change(arm, tmp_path, monkeypatch, case):
    directory = arm()
    path = directory / "arm.json"
    target = tmp_path / "untouched"
    target.write_text("preserve me")
    if case == "directory_mode":
        directory.chmod(0o755)
    elif case == "arm_mode":
        path.chmod(0o644)
    elif case == "arm_symlink":
        path.unlink()
        path.symlink_to(target)
    elif case == "directory_symlink":
        link = tmp_path / "link"
        link.symlink_to(directory, target_is_directory=True)
        monkeypatch.setattr(trace_mod, "_directory", lambda: link)
    elif case == "artifact_symlink":
        (directory / "capture.jsonl").symlink_to(target)
    elif case == "arm_hardlink":
        os.link(path, tmp_path / "extra-link")
    elif case == "invalid_json":
        path.write_text("not json")
    else:
        path.write_text(" " * 1025)
    frames = asyncio.run(_collect(_events(), _context()))
    assert _partials(frames) == ['{"message":"safe relay"}']
    assert target.read_text() == "preserve me"
    if case != "artifact_symlink":
        assert not (directory / "capture.jsonl").exists()


@pytest.mark.parametrize("kind", ["byte_limit", "event_limit"])
def test_hard_limits_disarm_and_do_not_change_output(arm, kind, monkeypatch):
    # The whitespace payload is a deliberate repetition-breaker trip (TASK-1915);
    # disable it so this test isolates trace hard limits.
    monkeypatch.setenv(ENV_MIN_SPAN, "0")
    directory = arm()
    deltas = [" " * trace_mod.MAX_BYTES] if kind == "byte_limit" else [" "] * trace_mod.MAX_EVENTS
    frames = asyncio.run(_collect(_events(deltas=deltas), _context()))
    rows = _rows(directory)
    assert rows[-1] == {"stage": "end", "status": kind}
    assert (directory / "capture.jsonl").stat().st_size <= trace_mod.MAX_BYTES
    assert len(rows) <= trace_mod.MAX_EVENTS
    assert _partials(frames) == ['{"message":"safe relay"}']
    assert not (directory / "arm.json").exists()


def test_expiry_timer_closes_idle_capture_without_sleep(arm, monkeypatch):
    directory = arm()

    async def run():
        trace = _claim()
        assert trace is not None
        callback = trace.timer._callback
        args = trace.timer._args
        # Invoke the scheduled expiry deterministically; no timing threshold/sleep.
        callback(*args)
        assert trace.fd is None
        trace.record("delta", delta="must not be retained")

    asyncio.run(run())
    assert _rows(directory)[-1] == {"stage": "end", "status": "expired"}
    assert b"must not" not in (directory / "capture.jsonl").read_bytes()


def test_wall_clock_expiry_is_checked_before_every_record(arm, monkeypatch):
    directory = arm()

    async def run():
        trace = _claim()
        monkeypatch.setattr(trace_mod.time, "time", lambda: 1061)
        trace.record("delta", delta="excluded after expiry")
        assert trace.fd is None

    asyncio.run(run())
    assert _rows(directory)[-1]["status"] == "expired"


def test_capture_write_failure_does_not_break_translation(arm, monkeypatch):
    directory = arm()

    def fail(*_args):
        raise OSError("synthetic write failure")

    monkeypatch.setattr(trace_mod.os, "write", fail)
    frames = asyncio.run(_collect(_events(), _context()))
    assert _partials(frames) == ['{"message":"safe relay"}']
    assert (directory / "capture.jsonl").read_bytes() == b""
    assert not (directory / "arm.json").exists()


def test_early_generator_close_disarms_and_closes_file(arm):
    directory = arm()

    async def run():
        async def upstream():
            yield NS(type="response.output_item.added", item=_item())
        stream = responses_to_anthropic_sse(upstream(), "synthetic/model", request_context=_context())
        await anext(stream)  # message_start
        await anext(stream)  # selected tool start / claimed artifact
        await stream.aclose()

    asyncio.run(run())
    assert _rows(directory)[-1] == {"stage": "end", "status": "stream_closed_before_capture"}
    assert not (directory / "arm.json").exists()


def test_missing_done_is_not_a_successful_capture(arm):
    directory = arm()
    events = _events(deltas=['{"message":"unterminated"}'])[:2]
    frames = asyncio.run(_collect(events, _context()))
    assert _partials(frames) == ['{"message":"unterminated"}']
    assert _rows(directory)[-1] == {"stage": "end", "status": "missing_done"}


def test_concurrent_claims_have_exactly_one_winner(arm):
    directory = arm()

    async def run():
        first = _claim()
        assert first is not None
        # Simulates a second request which read the same arm before the winner.
        arm()
        assert _claim() is None
        first.finish("captured")

    asyncio.run(run())
    assert [r["stage"] for r in _rows(directory)] == ["start", "end"]
