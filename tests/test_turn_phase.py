"""TASK-1074: transient turn phase tracker."""

from __future__ import annotations

from agent_bridge.events import AgentEventKind
from claude_code_bridge.event_ops import ClaudeEventMixin
from claude_code_bridge.turn_phase import TurnPhaseTracker


class _Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def _tracker(**kwargs) -> tuple[TurnPhaseTracker, list[tuple[str, dict]], _Clock]:
    published: list[tuple[str, dict]] = []
    clock = _Clock()
    tracker = TurnPhaseTracker(
        lambda text, status: published.append((text, status)),
        clock=clock,
        **kwargs,
    )
    return tracker, published, clock


def _phases(published: list[tuple[str, dict]]) -> list[str]:
    return [status["phase"] for _text, status in published]


def _block(block_type: str) -> dict:
    return {"type": "content_block_start", "content_block": {"type": block_type}}


def test_snapshot_carries_frozen_contract_fields():
    tracker, published, _clock = _tracker(model="glm-5", context_tokens=lambda: 4200)
    tracker.on_turn_start()

    text, status = published[0]
    assert text == "Waiting on model"
    assert status == {
        "phase": "waiting_model",
        "state": "waiting_model",
        "since": 1000.0,
        "model": "glm-5",
        "context_tokens": 4200,
    }


def test_publishes_only_on_transition_and_since_tracks_phase_entry():
    tracker, published, clock = _tracker()
    tracker.on_turn_start()
    clock.now = 1005.0
    tracker.on_stream_event(_block("thinking"))
    clock.now = 1010.0
    tracker.on_stream_event(_block("thinking"))  # same phase: no event
    tracker.on_stream_event({"type": "content_block_delta"})  # ignored
    tracker.on_stream_event(_block("text"))

    assert _phases(published) == ["waiting_model", "reasoning", "streaming"]
    assert [s["since"] for _t, s in published] == [1000.0, 1005.0, 1010.0]


def test_tool_results_return_to_waiting_only_when_all_tools_closed():
    tracker, published, _clock = _tracker()
    tracker.on_turn_start()
    tracker.on_stream_event(_block("tool_use"))
    tracker.on_tool_results(1)
    tracker.on_tool_results(0)

    assert _phases(published) == ["waiting_model", "streaming", "waiting_model"]


def test_compaction_lifecycle_is_a_phase_not_text():
    tracker, published, _clock = _tracker()
    tracker.on_turn_start()
    tracker.on_system_status({"status": "compacting"})
    tracker.on_system_status({"status": "compacting"})
    tracker.on_system_status({"compact_result": "success"})

    assert _phases(published) == ["waiting_model", "compacting", "waiting_model"]
    assert published[1][0] == "Compacting conversation"


def test_compact_result_outside_compaction_is_ignored():
    tracker, published, _clock = _tracker()
    tracker.on_turn_start()
    tracker.on_system_status({"compact_result": "failed"})
    assert _phases(published) == ["waiting_model"]


def test_retry_detail_refresh_republishes_without_resetting_since():
    tracker, published, clock = _tracker()
    tracker.on_turn_start()
    clock.now = 1001.0
    tracker.on_api_retry(attempt=1, max_attempts=10, error_status=529)
    clock.now = 1009.0
    tracker.on_api_retry(attempt=2, max_attempts=10, error_status=529)

    assert _phases(published) == ["waiting_model", "retrying", "retrying"]
    assert published[2][1]["attempt"] == 2
    assert published[2][1]["since"] == 1001.0


def test_stall_after_silence_then_resume_on_next_sdk_message():
    tracker, published, _clock = _tracker(stall_after=90)
    tracker.on_turn_start()
    tracker.on_stream_event(_block("thinking"))
    tracker.on_health({"phase": "sdk_idle", "wait_elapsed_seconds": 30.0})
    tracker.on_health({"phase": "sdk_idle", "wait_elapsed_seconds": 90.0})
    tracker.on_health({"phase": "sdk_idle", "wait_elapsed_seconds": 120.0})
    tracker.on_sdk_message()
    tracker.on_sdk_message()

    assert _phases(published) == ["waiting_model", "reasoning", "stalled", "reasoning"]


def test_no_stall_while_tool_runs_or_silence_is_explained():
    tracker, published, _clock = _tracker(stall_after=90)
    tracker.on_turn_start()
    tracker.on_health({"phase": "tool_running", "wait_elapsed_seconds": 600.0})
    tracker.on_system_status({"status": "compacting"})
    tracker.on_health({"phase": "sdk_idle", "wait_elapsed_seconds": 600.0})

    assert _phases(published) == ["waiting_model", "compacting"]


def test_proxy_states_map_to_phases_and_unknown_states_drop():
    tracker, published, _clock = _tracker()
    tracker.on_turn_start()
    tracker.on_proxy_status({
        "state": "reconnecting",
        "message": "Upstream stalled; reconnecting over HTTPS…",
        "attempt": 1,
        "fallback_transport": "sse",
    })
    tracker.on_proxy_status({"state": "mystery"})
    tracker.on_proxy_status({"state": "recovered", "message": "Recovered"})

    assert _phases(published) == ["waiting_model", "reconnecting", "streaming"]
    text, status = published[1]
    assert text == "Upstream stalled; reconnecting over HTTPS…"
    assert status["fallback_transport"] == "sse"
    assert status["attempt"] == 1


class _ProxyHarness(ClaudeEventMixin):
    _backend_name = "claude-code"

    def __init__(self):
        self._proxy_request_sessions = {"req_1": "codex:nick"}
        self._trigger_message_ids = {}
        self.events = []

    def _publish_event(self, *_args, **kwargs):
        self.events.append(kwargs)


def test_proxy_status_routes_through_registered_tracker():
    harness = _ProxyHarness()
    tracker, published, _clock = _tracker()
    harness._turn_phase_trackers = {"req_1": tracker}

    harness.publish_proxy_status("req_1", {"state": "reconnecting", "attempt": 1})

    assert harness.events == []  # the tracker is the single publisher
    assert _phases(published) == ["reconnecting"]


def test_proxy_status_without_tracker_keeps_legacy_event():
    harness = _ProxyHarness()
    harness.publish_proxy_status("req_1", {"state": "reconnecting", "message": "x"})

    assert len(harness.events) == 1
    assert harness.events[0]["kind"] is AgentEventKind.UPSTREAM_STATUS
