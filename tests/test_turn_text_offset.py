"""TASK-1509: a mid-turn message anchors even after the capped stream trims turn_start."""
import asyncio
import json
from types import SimpleNamespace

from agent_bridge.subscriber import RedisSubscriber


def subscriber_over(events: list[dict]) -> RedisSubscriber:
    """A subscriber whose stream holds ``events`` oldest→newest."""
    entries = [(f"{index + 1}-0", {"payload": json.dumps(event)}) for index, event in enumerate(events)]

    async def xrevrange(_key, max_id, _min, count):
        newest_first = entries[::-1]
        if max_id != "+":
            bound = int(max_id.lstrip("(").split("-")[0])
            newest_first = [entry for entry in newest_first if int(entry[0].split("-")[0]) < bound]
        return newest_first[:count]

    subscriber = RedisSubscriber("redis://localhost:6379/0")
    subscriber._pub_redis = SimpleNamespace(xrevrange=xrevrange)
    return subscriber


def offset(events: list[dict], **kwargs) -> int | None:
    return asyncio.run(subscriber_over(events).turn_text_offset("caid", "nick", "turn-live", **kwargs))


def text(at: int, delta: str, turn: str = "turn-live") -> dict:
    return {"_type": "text_delta", "turn_id": turn, "text_offset": at, "delta": delta}


def tool(at: int, turn: str = "turn-live") -> dict:
    return {"_type": "tool_event", "event": "tool_start", "turn_id": turn, "text_offset": at}


def reasoning(turn: str = "turn-live") -> dict:
    return {"_type": "reasoning_delta", "turn_id": turn, "delta": "thinking"}


def test_trimmed_turn_start_still_yields_the_live_text_end():
    # The prod case: turn_start fell out of the stream under reasoning deltas.
    events = [text(0, "Reading the board."), tool(18), *[reasoning() for _ in range(40)]]
    assert offset(events, batch_size=7) == 18


def test_newest_text_delta_measures_utf16_units():
    assert offset([tool(0), text(0, "Hi 👀")]) == 5


def test_tool_start_marks_the_position_when_it_is_newest():
    assert offset([text(0, "Checking."), tool(9), reasoning()]) == 9


def test_other_turns_never_supply_the_position():
    events = [text(0, "mine"), text(0, "someone else's longer reply", turn="turn-other")]
    assert offset(events) == 4


def test_turn_without_text_yet_anchors_at_its_start():
    events = [text(0, "older turn", turn="turn-old"), {"_type": "turn_start", "turn_id": "turn-live"}, reasoning()]
    assert offset(events) == 0


def test_no_position_left_in_the_stream_is_unknown():
    assert offset([reasoning() for _ in range(5)], batch_size=2) is None
    assert offset([]) is None


def test_hard_cap_bounds_the_scan():
    events = [text(0, "too old"), *[reasoning() for _ in range(10)]]
    assert offset(events, batch_size=5, hard_cap=10) is None
