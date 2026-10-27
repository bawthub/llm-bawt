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
