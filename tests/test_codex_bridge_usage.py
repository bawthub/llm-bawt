"""TASK-1319: Codex usage is per turn, read from the rollout, not the SDK's
thread-cumulative ``turn.completed`` usage."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from codex_bridge.bridge import CodexBridge
from codex_bridge.rollout_usage import (
    RESIDENT_SOURCE,
    RolloutUsageReader,
    TokenCounts,
    parse_latest_turn,
)

THREAD_ID = "01a169f0-3976-71d1-a77d-7868e43b182e"


def _counts(input_tokens: int, cached: int, output: int) -> dict:
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "cache_write_input_tokens": 0,
        "output_tokens": output,
        "reasoning_output_tokens": 0,
        "total_tokens": input_tokens + output,
    }


def _event(payload: dict) -> str:
    return json.dumps({"timestamp": "2026-10-23T18:05:17.105Z",
                       "type": "event_msg", "payload": payload})


def build_rollout(turns: list[list[tuple[int, int, int]]]) -> tuple[list[str], dict]:
    """Rollout lines for turns of (input, cached, output) model calls, shaped
    like the codex CLI's: task_started, a count-less token_count, one
    token_count per call (the last one repeated), task_complete."""
    total = {"input": 0, "cached": 0, "output": 0}
    lines = [json.dumps({"type": "session_meta", "payload": {"id": THREAD_ID}})]
    for n, calls in enumerate(turns):
        lines.append(_event({"type": "task_started", "turn_id": f"t{n}"}))
        lines.append(_event({"type": "token_count", "info": None, "rate_limits": {}}))
        for inp, cached, out in calls:
            total = {"input": total["input"] + inp, "cached": total["cached"] + cached,
                     "output": total["output"] + out}
            lines.append(json.dumps({"type": "response_item",
                                     "payload": {"type": "function_call"}}))
            info = {
                "total_token_usage": _counts(total["input"], total["cached"], total["output"]),
                "last_token_usage": _counts(inp, cached, out),
                "model_context_window": 258400,
            }
            lines.append(_event({"type": "token_count", "info": info, "rate_limits": {}}))
        lines.append(lines[-1])  # the CLI can repeat an unchanged count
        lines.append(_event({"type": "task_complete", "turn_id": f"t{n}"}))
    return lines, total


# Turns 1 and 2 are the live thread 01a169f0's first two turns; turn 3 is a
# tool turn whose last call matches that thread's turn 3.
TURN_1 = [(19957, 11648, 32)]
TURN_2 = [(22512, 19840, 32)]
TURN_3 = [(60000, 50000, 400), (95000, 90000, 900), (129116, 128128, 1200)]


def write_rollout(home: Path, lines: list[str], thread_id: str = THREAD_ID) -> Path:
    day = home / "sessions" / "2026" / "10" / "23"
    day.mkdir(parents=True, exist_ok=True)
    path = day / f"rollout-2026-10-23T18-05-13-{thread_id}.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return path


def sdk_usage(total: dict) -> SimpleNamespace:
    return SimpleNamespace(input_tokens=total["input"],
                           cached_input_tokens=total["cached"],
                           output_tokens=total["output"])


def test_parse_latest_turn_tool_turn_reports_last_call_and_turn_output():
    lines, total = build_rollout([TURN_1, TURN_2, TURN_3])

    turn = parse_latest_turn(lines)

    assert turn is not None
    assert turn.thread_total == TokenCounts(input=total["input"], cached=total["cached"],
                                            output=total["output"])
    assert turn.turn_total == TokenCounts(input=284116, cached=268128, output=2500)
    assert turn.last_call == TokenCounts(input=129116, cached=128128, output=1200)
    assert turn.model_calls == 3  # the repeated count is not a call
    assert turn.context_window == 258400

    usage = turn.to_token_usage()
    assert usage["input_tokens"] == 129116 - 128128
    assert usage["cache_read_tokens"] == 128128
    assert usage["cache_creation_tokens"] == 0
    assert usage["output_tokens"] == 2500
    assert usage["resident_tokens"] == 129116
    assert usage["resident_source"] == RESIDENT_SOURCE
    assert usage["turn_input_tokens"] == 284116
    assert usage["turn_cached_input_tokens"] == 268128
    assert usage["turn_model_calls"] == 3


def test_parse_latest_turn_second_turn_is_a_delta_not_the_thread_total():
    lines, _ = build_rollout([TURN_1, TURN_2])

    turn = parse_latest_turn(lines)

    assert turn is not None
    assert turn.thread_total.input == 42469  # what the SDK reported as "this turn"
    assert turn.turn_total == TokenCounts(input=22512, cached=19840, output=32)
    assert turn.to_token_usage()["resident_tokens"] == 22512


def test_parse_latest_turn_skips_a_partly_flushed_tail():
    lines, _ = build_rollout([TURN_1, TURN_2])
    lines.append('{"type":"event_msg","payload":{"type":"token_count","info":{"tot')

    turn = parse_latest_turn(lines)

    assert turn is not None and turn.last_call.input == 22512


def test_parse_latest_turn_without_a_model_call_is_none():
    lines, _ = build_rollout([TURN_1])
    lines.append(_event({"type": "task_started", "turn_id": "t9"}))

    assert parse_latest_turn(lines) is None


def _append_turn(lines: list[str], total: dict, last: dict) -> None:
    lines.append(_event({"type": "task_started", "turn_id": "tx"}))
    lines.append(_event({"type": "token_count", "info": {
        "total_token_usage": total, "last_token_usage": last,
        "model_context_window": 258400,
    }}))


@pytest.mark.asyncio
async def test_resolve_rejects_a_counter_reset_even_when_sdk_matches(tmp_path):
    # Codex review cmvld8ffg005a0mpze318bo9v: a shrinking running total gave
    # negative turn usage that still "matched" the SDK.
    lines, _ = build_rollout([TURN_1])
    reset_counts = _counts(500, 100, 10)
    _append_turn(lines, reset_counts, reset_counts)
    write_rollout(tmp_path, lines)
    reader = RolloutUsageReader(tmp_path, attempts=1, retry_delay=0)
    sdk = TokenCounts.from_sdk_usage(sdk_usage({"input": 500, "cached": 100, "output": 10}))

    assert reader.read_latest_turn(THREAD_ID) is None
    assert await reader.resolve(THREAD_ID, sdk) is None


@pytest.mark.parametrize("total, last", [
    # the running total grew by more than the call it reports
    (_counts(19957 + 30000, 11648 + 20000, 32 + 50), _counts(25000, 20000, 50)),
    # cached exceeds input: not a possible call
    (_counts(19957 + 100, 11648 + 200, 32 + 5), _counts(100, 200, 5)),
])
def test_parse_latest_turn_rejects_counts_that_do_not_add_up(total, last):
    lines, _ = build_rollout([TURN_1])
    _append_turn(lines, total, last)

    assert parse_latest_turn(lines) is None


def test_parse_latest_turn_rejects_an_unparseable_count():
    lines, _ = build_rollout([TURN_1, TURN_2])
    lines.insert(-2, _event({"type": "token_count", "info": {
        "total_token_usage": {"input_tokens": "lots"}, "last_token_usage": {},
    }}))

    assert parse_latest_turn(lines) is None


def test_parse_latest_turn_only_judges_the_latest_turn():
    lines, _ = build_rollout([TURN_1])
    reset = _counts(500, 100, 10)  # an earlier turn broke the running total
    _append_turn(lines, reset, reset)
    assert parse_latest_turn(lines) is None
    # the next turn counts cleanly on top of the reset total
    _append_turn(lines, _counts(500 + 22512, 100 + 19840, 10 + 32), _counts(22512, 19840, 32))

    turn = parse_latest_turn(lines)

    assert turn is not None
    assert turn.turn_total == TokenCounts(input=22512, cached=19840, output=32)
    assert turn.last_call.input == 22512


@pytest.mark.asyncio
async def test_resolve_accepts_the_sdk_running_total(tmp_path):
    lines, total = build_rollout([TURN_1, TURN_2, TURN_3])
    write_rollout(tmp_path, lines)
    reader = RolloutUsageReader(tmp_path, retry_delay=0)

    turn = await reader.resolve(THREAD_ID, TokenCounts.from_sdk_usage(sdk_usage(total)))

    assert turn is not None and turn.last_call.input == 129116


@pytest.mark.asyncio
async def test_resolve_accepts_a_per_turn_sdk_usage(tmp_path):
    lines, _ = build_rollout([TURN_1, TURN_2])
    write_rollout(tmp_path, lines)
    reader = RolloutUsageReader(tmp_path, retry_delay=0)
    per_turn = {"input": 22512, "cached": 19840, "output": 32}

    turn = await reader.resolve(THREAD_ID, TokenCounts.from_sdk_usage(sdk_usage(per_turn)))

    assert turn is not None and turn.turn_total.input == 22512


@pytest.mark.asyncio
async def test_resolve_waits_for_the_rollout_to_catch_up(tmp_path):
    lines, _ = build_rollout([TURN_1])
    path = write_rollout(tmp_path, lines)
    full, total = build_rollout([TURN_1, TURN_2])
    reader = RolloutUsageReader(tmp_path, attempts=3, retry_delay=0)
    original = reader.read_latest_turn
    reads = []

    def read_then_flush(thread_id):
        reads.append(thread_id)
        result = original(thread_id)
        path.write_text("\n".join(full) + "\n")
        return result

    reader.read_latest_turn = read_then_flush

    turn = await reader.resolve(THREAD_ID, TokenCounts.from_sdk_usage(sdk_usage(total)))

    assert len(reads) == 2
    assert turn is not None and turn.turn_total.input == 22512


@pytest.mark.asyncio
async def test_resolve_drops_usage_the_rollout_cannot_confirm(tmp_path, caplog):
    lines, _ = build_rollout([TURN_1, TURN_2])
    write_rollout(tmp_path, lines)
    reader = RolloutUsageReader(tmp_path, attempts=2, retry_delay=0)
    wrong = TokenCounts(input=1, cached=0, output=1)

    assert await reader.resolve(THREAD_ID, wrong) is None
    assert "does not match" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("thread_id", [None, "", "../../etc/passwd", "*", "missing-thread-0001"])
async def test_resolve_without_a_usable_rollout_is_none(tmp_path, thread_id):
    lines, total = build_rollout([TURN_1])
    write_rollout(tmp_path, lines)
    reader = RolloutUsageReader(tmp_path, attempts=1, retry_delay=0)

    assert await reader.resolve(thread_id, TokenCounts.from_sdk_usage(sdk_usage(total))) is None


def test_locate_reads_codex_home_lazily(tmp_path, monkeypatch):
    lines, _ = build_rollout([TURN_1])
    path = write_rollout(tmp_path, lines)
    reader = RolloutUsageReader()
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))

    assert reader.locate(THREAD_ID) == path


@pytest.mark.parametrize("carrier", [
    SimpleNamespace(input_tokens=5, cached_input_tokens=3, output_tokens=2),
    SimpleNamespace(usage=SimpleNamespace(input_tokens=5, cached_input_tokens=3, output_tokens=2)),
    {"input_tokens": 5, "cached_input_tokens": 3, "output_tokens": 2},
    {"usage": {"input_tokens": 5, "cached_input_tokens": 3, "output_tokens": 2}},
])
def test_sdk_usage_shapes(carrier):
    assert TokenCounts.from_sdk_usage(carrier) == TokenCounts(input=5, cached=3, output=2)


def _bridge(tmp_path: Path, model_info: dict) -> CodexBridge:
    bridge = object.__new__(CodexBridge)
    bridge._rollout_usage = RolloutUsageReader(tmp_path, attempts=1, retry_delay=0)
    bridge._model_info = SimpleNamespace(get=AsyncMock(return_value=model_info))
    bridge._publish_event = Mock()
    return bridge


@pytest.mark.asyncio
async def test_turn_completed_publishes_per_turn_usage(tmp_path):
    lines, total = build_rollout([TURN_1, TURN_2, TURN_3])
    write_rollout(tmp_path, lines)
    bridge = _bridge(tmp_path, {"context_window": 272000, "max_output_tokens": 128000})
    event = SimpleNamespace(type="turn.completed", usage=sdk_usage(total))

    await bridge._handle_event(
        event, bot_slug="codex", session_key="codex", request_id="r1", seq=0,
        text_parts=["done"], item_text={}, model="gpt-6", actual_model_ref=["gpt-6"],
        resume_id=THREAD_ID, session_persisted=True, item_buffers={},
        thread=SimpleNamespace(id=THREAD_ID),
    )

    usage = bridge._publish_event.call_args.kwargs["token_usage"]
    assert usage["input_tokens"] == 988
    assert usage["cache_read_tokens"] == 128128
    assert usage["output_tokens"] == 2500
    assert usage["resident_tokens"] == 129116
    assert usage["context_window"] == 272000  # the catalog wins
    assert usage["max_output_tokens"] == 128000


@pytest.mark.asyncio
async def test_turn_usage_falls_back_to_the_rollout_context_window(tmp_path):
    lines, total = build_rollout([TURN_1])
    write_rollout(tmp_path, lines)
    bridge = _bridge(tmp_path, {})

    usage = await bridge._turn_token_usage(
        sdk_usage(total), thread=SimpleNamespace(id=THREAD_ID), model="gpt-6",
    )

    assert usage is not None and usage["context_window"] == 258400


@pytest.mark.asyncio
async def test_turn_usage_is_none_without_sdk_usage_or_a_confirmed_rollout(tmp_path):
    bridge = _bridge(tmp_path, {})
    thread = SimpleNamespace(id=THREAD_ID)

    assert await bridge._turn_token_usage(None, thread=thread, model="gpt-6") is None
    assert await bridge._turn_token_usage(
        sdk_usage({"input": 7_740_000, "cached": 7_400_000, "output": 27_000}),
        thread=thread, model="gpt-6",
    ) is None
