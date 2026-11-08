"""Live per-iteration usage event contract (TASK-733)."""

from datetime import datetime, timezone

from agent_bridge.events import AgentEvent, AgentEventKind
from claude_code_bridge.send_usage import ClaudeUsageMixin, LiveUsagePublisher
from llm_bawt.service.turn_usage import TurnUsageCoordinator


class _Harness(ClaudeUsageMixin):
    def __init__(self) -> None:
        self.events: list[dict] = []

    @staticmethod
    def _model_provider_prefix(_model: str) -> str:
        return "openai_chatgpt"

    def _publish_event(self, _request_id, _session_key, _seq, **kwargs) -> None:
        self.events.append(kwargs)


def test_live_usage_update_normalizes_and_deduplicates_iteration_snapshot() -> None:
    harness = _Harness()
    publisher = LiveUsagePublisher(
        harness,
        "req-1",
        "snark:nick",
        "openai_chatgpt/gpt-5.6-sol",
        "openai_chatgpt/gpt-5.6-sol",
        372_000,
    )
    seq = publisher.publish(
        4,
        assistant_usage=None,
        stream_usage={
            "input_tokens": 1_885,
            "cache_read_input_tokens": 137_728,
            "cache_creation_input_tokens": 0,
            "output_tokens": 44,
        },
    )

    assert seq == 5
    assert publisher.last_signature is not None
    assert harness.events == [{
        "kind": AgentEventKind.USAGE_UPDATE,
        "model": "openai_chatgpt/gpt-5.6-sol",
        "token_usage": {
            "input_tokens": 1_885,
            "cache_read_tokens": 137_728,
            "cache_creation_tokens": 0,
            "output_tokens": 44,
            "resident_tokens": 139_613,
            "resident_source": "live_iteration_total_input",
            "context_window": 372_000,
            "total_cost_usd": harness.events[0]["token_usage"]["total_cost_usd"],
            "usage_status": "partial",
        },
    }]

    next_seq = publisher.publish(
        seq,
        assistant_usage=None,
        stream_usage={"output_tokens": 44},
    )

    assert next_seq == seq
    assert len(harness.events) == 1
    assert publisher.stream_usage == {
        "input_tokens": 1_885,
        "cache_read_input_tokens": 137_728,
        "cache_creation_input_tokens": 0,
        "output_tokens": 44,
    }


def test_live_usage_update_ignores_output_only_frame_without_resident_context() -> None:
    harness = _Harness()

    publisher = LiveUsagePublisher(
        harness,
        "req-1",
        "snark:nick",
        "claude-opus-4-6",
        "claude-opus-4-6",
        200_000,
    )
    seq = publisher.publish(
        2,
        assistant_usage=None,
        stream_usage={"output_tokens": 12},
    )

    assert seq == 2
    assert publisher.last_signature is None
    assert harness.events == []


def test_turn_usage_coordinator_checkpoints_and_builds_both_channels() -> None:
    class Store:
        def __init__(self) -> None:
            self.updates = []

        def update_turn(self, **kwargs) -> None:
            self.updates.append(kwargs)

    store = Store()
    coordinator = TurnUsageCoordinator(store)
    event = coordinator.capture(
        turn_id="turn-1",
        trigger_message_id="user-1",
        bot_id="snark",
        user_id="nick",
        model="codex-gpt-5.6-sol",
        token_usage={"resident_tokens": 139_613, "usage_status": "partial"},
    )

    assert event is not None
    assert store.updates == [{
        "turn_id": "turn-1",
        "token_usage": {"resident_tokens": 139_613, "usage_status": "partial"},
    }]
    assert event["_type"] == "turn_usage"
    assert event["trigger_message_id"] == "user-1"
    assert coordinator.http_chunk(event) == {
        "object": "chat.completion.usage",
        "turn_id": "turn-1",
        "trigger_message_id": "user-1",
        "model": "codex-gpt-5.6-sol",
        "token_usage": {"resident_tokens": 139_613, "usage_status": "partial"},
    }


def test_usage_checkpoint_write_failure_does_not_end_the_turn() -> None:
    """TASK-1385: a lock timeout on a mid-run checkpoint must not raise."""
    class LockedStore:
        def update_turn(self, **_kwargs) -> None:
            raise RuntimeError("canceling statement due to lock timeout")

    event = TurnUsageCoordinator(LockedStore()).capture(
        turn_id="turn-1",
        trigger_message_id="user-1",
        bot_id="caid",
        user_id="nick",
        model="m",
        token_usage={"resident_tokens": 10, "usage_status": "partial"},
    )

    assert event is not None
    assert event["token_usage"] == {"resident_tokens": 10, "usage_status": "partial"}


def test_usage_update_round_trips_through_agent_event_transport() -> None:
    event = AgentEvent(
        event_id="evt-usage",
        session_key="snark:nick",
        run_id="req-1",
        kind=AgentEventKind.USAGE_UPDATE,
        origin="system",
        model="openai_chatgpt/gpt-5.6-sol",
        timestamp=datetime.now(timezone.utc),
        token_usage={"resident_tokens": 139_613, "usage_status": "partial"},
    )

    restored = AgentEvent.from_dict(event.to_dict())

    assert restored.kind is AgentEventKind.USAGE_UPDATE
    assert restored.token_usage == event.token_usage


def test_post_done_context_update_persists_final_status_and_breakdown() -> None:
    class Store:
        def __init__(self) -> None:
            self.updates = []

        def update_turn(self, **kwargs) -> None:
            self.updates.append(kwargs)

    store = Store()
    coordinator = TurnUsageCoordinator(store)
    final = {"input_tokens": 13, "cache_read_tokens": 30,
             "cache_creation_tokens": 1, "output_tokens": 4,
             "usage_status": "final", "context_breakdown": {"read_ms": 18.2}}
    event = coordinator.capture(
        turn_id="turn-1", trigger_message_id="user-1", bot_id="bot",
        user_id="user", model="model", token_usage=final,
    )
    assert event["token_usage"] == final
    assert store.updates[-1] == {"turn_id": "turn-1", "token_usage": final}
    assert coordinator.http_chunk(event)["token_usage"]["context_breakdown"] == {"read_ms": 18.2}


def _start(usage: dict | None = None) -> dict:
    return {"type": "message_start", "message": {"usage": usage or {
        "input_tokens": 0, "output_tokens": 0,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
    }}}


def _delta(fresh: int, cached: int, created: int = 0) -> dict:
    return {"type": "message_delta", "usage": {
        "input_tokens": fresh, "output_tokens": 5,
        "cache_creation_input_tokens": created, "cache_read_input_tokens": cached,
    }}


def test_request_totals_sum_every_upstream_request_not_only_the_last() -> None:
    """TASK-1961: proxy routes report usage only in message_delta."""
    harness = _Harness()
    publisher = LiveUsagePublisher(harness, "req-1", "vex:nick", "m", "m", 372_000)
    assert publisher.request_totals() is None
    for fresh, cached in ((40_641, 0), (41_607, 0), (18_235, 128_640)):
        publisher.observe_stream_event(_start())
        publisher.observe_stream_event(_delta(fresh, cached))

    assert publisher.request_totals() == {
        "requests": 3,
        "input_tokens": 40_641 + 41_607 + 18_235 + 128_640,
        "cache_read_tokens": 128_640,
        "cache_creation_tokens": 0,
        "requests_with_cache_read": 1,
    }
    usage = publisher.with_totals({"input_tokens": 18_235, "cache_read_tokens": 128_640})
    assert usage["input_tokens"] == 18_235  # final-request context is untouched
    assert usage["turn_request_totals"]["requests"] == 3
    assert publisher.with_totals(None) == {"turn_request_totals": publisher.request_totals()}


def test_request_totals_keep_native_start_usage_and_route_interleaved_subagents() -> None:
    harness = _Harness()
    publisher = LiveUsagePublisher(harness, "req-1", "byte:nick", "m", "m", 1_000_000)
    # Native Anthropic: usage arrives in message_start; delta may repeat zeros.
    publisher.observe_stream_event(_start({
        "input_tokens": 10, "cache_read_input_tokens": 9_000, "cache_creation_input_tokens": 500,
    }))
    publisher.observe_stream_event(_start(), "toolu_sub")
    publisher.observe_stream_event(_delta(0, 0))
    publisher.observe_stream_event(_delta(3_000, 0), "toolu_sub")

    assert publisher.request_totals() == {
        "requests": 2,
        "input_tokens": 10 + 9_000 + 500 + 3_000,
        "cache_read_tokens": 9_000,
        "cache_creation_tokens": 500,
        "requests_with_cache_read": 1,
    }


def test_live_usage_update_carries_running_request_totals() -> None:
    harness = _Harness()
    publisher = LiveUsagePublisher(harness, "req-1", "vex:nick", "m", "m", 372_000)
    publisher.observe_stream_event(_start())
    publisher.observe_stream_event(_delta(1_000, 0))
    publisher.publish(0, assistant_usage=None, stream_usage=_delta(1_000, 0)["usage"])

    assert harness.events[-1]["token_usage"]["turn_request_totals"]["requests"] == 1


def test_live_usage_republishes_when_only_request_totals_change() -> None:
    """Two same-sized consecutive requests leave the context signature equal;
    the changed request count must still publish."""
    harness = _Harness()
    publisher = LiveUsagePublisher(harness, "req-1", "vex:nick", "m", "m", 372_000)
    seq = 0
    for _ in range(2):
        publisher.observe_stream_event(_start())
        publisher.observe_stream_event(_delta(1_000, 0))
        seq = publisher.publish(seq, assistant_usage=None, stream_usage=_delta(1_000, 0)["usage"])
    assert seq == 2
    assert [e["token_usage"]["turn_request_totals"]["requests"] for e in harness.events] == [1, 2]


def test_with_totals_keeps_totals_without_inventing_final_request_usage() -> None:
    harness = _Harness()
    publisher = LiveUsagePublisher(harness, "req-1", "byte:nick", "m", "m", 1_000_000)
    assert publisher.with_totals(None) is None  # nothing measured yet
    publisher.observe_stream_event(_start({"input_tokens": 50, "cache_read_input_tokens": 0}))
    assert publisher.with_totals(None) == {"turn_request_totals": {
        "requests": 1, "input_tokens": 50, "cache_read_tokens": 0,
        "cache_creation_tokens": 0, "requests_with_cache_read": 0,
    }}
