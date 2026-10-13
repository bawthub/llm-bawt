"""Offline TASK-861 audit transport/storage and policy admin contracts."""
# ruff: noqa: F811 — imported fixtures intentionally share test argument names
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from test_task861 import store, client  # noqa: F401 — isolated SQLite/API fixtures
from tests.test_approval_bridge_gate import _bridge
from agent_bridge.events import AgentEvent, AgentEventKind
from agent_bridge.subscriber import RedisSubscriber
from llm_bawt.agent_backends import agent_bridge as backend_module
from llm_bawt.service.routes import approval_policies as routes


def event_from_gate(*, hook=True, command="dd of=/dev/sda token=DO_NOT_LOG"):
    bridge = _bridge()
    events = []
    def publish(request_id, session_key, seq, **kw):
        events.append(AgentEvent(
            event_id=f"event-{seq}", session_key=session_key, run_id=request_id,
            kind=kw["kind"], origin="system", tool_name=kw.get("tool_name"),
            tool_use_id=kw.get("tool_use_id"), tool_arguments=kw.get("tool_arguments"),
            provider="claude-code", raw=kw.get("extra_raw", {}),
        ))
    bridge._publish_event = publish
    args = ("Bash", {"command": command, "secret": "DO_NOT_LOG"})
    if hook:
        asyncio.run(bridge._evaluate_tool_gate_hook(*args, "tool-1", "req-1", "session-1", [0]))
    else:
        asyncio.run(bridge._evaluate_tool_gate(*args, SimpleNamespace(tool_use_id="tool-1"), "req-1", "session-1", [0]))
    return events[0]


@pytest.mark.parametrize("hook", [True, False])
@pytest.mark.parametrize("command,action,policy", [
    ("ls", "allow", None), ("rm -rf /tmp/safe/x", "allow", "allow1"),
    ("dd of=/dev/sda token=DO_NOT_LOG", "deny", "deny1"),
    ("rm -rf /home/x", "require_approval", "req1"),
])
def test_bridge_event_redacted_and_committed(store, hook, command, action, policy):
    event = AgentEvent.from_dict(event_from_gate(hook=hook, command=command).to_dict())
    assert event.kind == AgentEventKind.APPROVAL_DECISION
    assert event.tool_arguments is None
    assert "DO_NOT_LOG" not in json.dumps(event.to_dict())
    for _ in range(2):
        store.record_bridge_decision(event, bot_id="bot", session_id="thread")
    rows, total = store.list_decisions()
    assert total == 1
    row = rows[0]
    assert (row.action, row.policy_id) == (action, policy)
    assert row.policy_version == (1 if policy else None)
    assert row.bundle_etag == "x" and len(row.invocation_hash) == 64
    assert (row.bot_id, row.session_id, row.session_key, row.request_id, row.tool_use_id) == (
        "bot", "thread", "session-1", "req-1", "tool-1")
    assert row.subject == "[subject redacted]" and row.source == "bridge"


def test_app_run_consumer_persists_without_ui_callback(store, monkeypatch):
    event = event_from_gate()
    class Subscriber:
        async def connect(self): pass
        async def close(self): pass
        async def send_command(self, **kwargs): pass
        async def subscribe_run(self, *args, **kwargs):
            yield event
    monkeypatch.setattr("agent_bridge.subscriber.RedisSubscriber", lambda *a: Subscriber())
    monkeypatch.setattr(backend_module, "get_agent_subscriber", lambda: SimpleNamespace(
        _redis=SimpleNamespace(connection_pool=SimpleNamespace(connection_kwargs={"url": "redis://invalid"}))))
    backend = backend_module.AgentBridgeBackend.__new__(backend_module.AgentBridgeBackend)
    import threading
    backend._thread_local = threading.local()
    backend._approval_audit_store = store
    assert list(backend.stream_raw("test", {"bot_id": "b", "thread_session_id": "thread"})) == []
    rows, total = store.list_decisions()
    assert total == 1 and rows[0].bot_id == "b" and rows[0].session_id == "thread"


def test_app_commit_retry_and_explicit_loss(store, caplog):
    backend = backend_module.AgentBridgeBackend.__new__(backend_module.AgentBridgeBackend)
    calls = []
    def fail(*args, **kw):
        calls.append(1)
        raise RuntimeError("DO_NOT_LOG_SQL_PARAMETERS")
    backend._approval_audit_store = SimpleNamespace(engine=True, record_bridge_decision=fail)
    assert asyncio.run(backend._persist_approval_decision(event_from_gate(), {})) is False
    assert len(calls) == 3 and "audit LOST" in caplog.text
    assert "DO_NOT_LOG_SQL_PARAMETERS" not in caplog.text
    def recover(event, **kwargs):
        if calls.pop():
            raise RuntimeError("transient")
        store.record_bridge_decision(event, **kwargs)
    calls[:] = [0, 1]
    backend._approval_audit_store.record_bridge_decision = recover
    assert asyncio.run(backend._persist_approval_decision(event_from_gate(), {})) is True
    assert store.list_decisions()[1] == 1


def test_revision_pagination_real_total_and_deleted_policy(client, store):
    row = store.create({"tool_name": "Bash"})
    for index in range(4):
        store.update(row.id, {"order": index})
    store.delete(row.id)
    page = client.get(f"/v1/tool-approval-policies/{row.id}/revisions?offset=2&limit=2").json()
    assert page["total"] == 6 and [r["version"] for r in page["revisions"]] == [4, 3]
    assert (page["limit"], page["offset"]) == (2, 2)
    assert client.get(f"/v1/tool-approval-policies/{row.id}/revisions?offset=99").json()["total"] == 6


@pytest.mark.parametrize("recipients", [0, 2, None, "failure"])
def test_reload_publication_not_installation(client, monkeypatch, recipients):
    publish = AsyncMock(return_value=recipients)
    if recipients == "failure":
        publish.side_effect = RuntimeError("offline")
    monkeypatch.setattr(routes, "_subscriber", lambda: SimpleNamespace(publish_approval_reload=publish))
    body = client.post("/v1/admin/reload-tool-approval-policies").json()
    assert body["published"] is isinstance(recipients, int)
    assert body["bridge_installed"] == "unknown" and body["status"] != "reloaded"


def test_missing_subscriber_and_status_coverage(client, store, monkeypatch):
    assert client.post("/v1/admin/reload-tool-approval-policies").json()["published"] is False
    monkeypatch.setattr(routes, "get_service", lambda: SimpleNamespace(config=None))
    monkeypatch.setattr(routes, "get_tool_approval_policy_store", lambda cfg: store)
    body = client.get("/v1/tool-approval-policies/status").json()
    assert all(r["enforcement"] == "unsupported" for r in body["coverage"] if r["backend"] in ("codex", "openclaw"))
    assert "best-effort" in " ".join(body["notes"])


def test_grant_consumption_audit_and_transport_outage():
    from tests.test_approval_bridge_gate import _grant, _continuation
    bridge = _bridge()
    arguments = {"command": "rm -rf /home/x"}
    decision = bridge._decide_approval("Bash", arguments)
    _grant(bridge, decision.grant_key)
    events = []
    bridge._publish_event = lambda *a, **kw: events.append(kw)
    assert asyncio.run(bridge._evaluate_tool_gate_hook(
        "Bash", arguments, "retry-tool", _continuation(), "s", [0])) == {}
    audits = [e["extra_raw"] for e in events if e["kind"] == AgentEventKind.APPROVAL_DECISION]
    assert [a["outcome"] for a in audits] == ["require_approval", "grant_allowed"]
    assert all(a["invocation_hash"] == decision.grant_key for a in audits)
    def offline(*a, **kw):
        raise RuntimeError("offline")
    bridge._publish_event = offline
    assert asyncio.run(bridge._evaluate_tool_gate_hook(
        "Bash", {"command": "ls"}, "tool", "r", "s", [0])) == {}
    result = asyncio.run(bridge._evaluate_tool_gate_hook(
        "Bash", {"command": "dd of=/dev/sda"}, "tool", "r", "s", [0]))
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
    bridge._approval_fail_closed, bridge._policy_fetch_ok = True, False
    bridge._publish_event = lambda *a, **kw: events.append(kw)
    asyncio.run(bridge._evaluate_tool_gate_hook("Bash", {}, "tool", "r", "s", [0]))
    assert events[-1]["extra_raw"]["outcome"] == "policy_unavailable"


def test_subscriber_propagates_failure_and_returns_zero():
    sub = RedisSubscriber.__new__(RedisSubscriber)
    sub._pub_redis = SimpleNamespace(publish=AsyncMock(return_value=0))
    assert asyncio.run(sub.publish_approval_reload()) == 0
    sub._pub_redis.publish.side_effect = RuntimeError("offline")
    with pytest.raises(RuntimeError):
        asyncio.run(sub.publish_approval_reload())
