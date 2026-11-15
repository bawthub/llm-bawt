from agent_bridge.events import AgentEventKind
from claude_code_bridge.event_ops import ClaudeEventMixin
from claude_code_bridge.proxy.request_context import ProxyRequestContext


class StatusHarness(ClaudeEventMixin):
    _backend_name = "claude-code"

    def __init__(self):
        self._proxy_request_sessions = {"req_1": "codex:operator"}
        self._trigger_message_ids = {"req_1": "user_1"}
        self.events = []

    def _publish_run_event_with_changed_file(self, request_id, event):
        self.events.append((request_id, event))


def test_proxy_status_is_structured_and_not_assistant_text():
    harness = StatusHarness()
    harness.publish_proxy_status("req_1", {
        "state": "reconnecting",
        "message": "Upstream stalled; reconnecting over HTTPS…",
        "attempt": 1,
        "fallback_transport": "sse",
    })

    assert len(harness.events) == 1
    request_id, event = harness.events[0]
    assert request_id == "req_1"
    assert event.kind is AgentEventKind.UPSTREAM_STATUS
    assert event.text == "Upstream stalled; reconnecting over HTTPS…"
    assert event.raw["upstream_status"]["fallback_transport"] == "sse"
    assert event.trigger_message_id == "user_1"


def test_proxy_status_for_inactive_run_is_dropped():
    harness = StatusHarness()
    harness.publish_proxy_status("missing", {"state": "reconnecting"})
    assert harness.events == []


def test_status_delivery_failure_never_breaks_upstream_recovery():
    def fail(_request_id, _status):
        raise RuntimeError("redis unavailable")

    context = ProxyRequestContext(
        request_id="req_1",
        provider="openai_chatgpt",
        status_callback=fail,
    )
    context.report_status({"state": "reconnecting"})
