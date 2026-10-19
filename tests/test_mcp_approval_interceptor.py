from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet

from agent_bridge.approval import (
    ApprovalPolicy,
    MatcherType,
    PolicyAction,
    PolicyBundle,
)
from agent_bridge.mcp_call_context import (
    MCP_CALL_CONTEXT_KEY,
    mint_mcp_call_context,
    mint_mcp_request_context,
)
from llm_bawt import task_turn_context as turn_codec
from llm_bawt.mcp_server import task_association
from llm_bawt.mcp_server.approval_interceptor import ApprovalAwareFastMCP


def _turn_values():
    return {
        "session_id": str(uuid4()),
        "turn_id": "turn-" + "a" * 32,
        "trigger_message_id": str(uuid4()),
        "bot_id": "snark",
        "user_id": "nick",
    }


class FakeStore:
    def __init__(self, policy, order):
        self.policy = policy
        self.order = order
        self.rows = []

    def compile_bundle(self):
        policies = [self.policy] if self.policy else []
        return PolicyBundle(version=1, etag="test", policies=policies)

    def record_decision(self, **kwargs):
        self.decision = kwargs

    def record_mcp_request(self, **kwargs):
        with_created = kwargs.pop("with_created", False)
        self.order.append("persist")
        for existing, _persisted in self.rows:
            if existing.id == kwargs["request_id"]:
                return (existing, False) if with_created else existing
        row = SimpleNamespace(
            id=kwargs["request_id"],
            tool_use_id=kwargs.get("tool_use_id"),
            turn_id=kwargs["turn_id"],
            trigger_message_id=kwargs.get("trigger_message_id"),
            bot_id=kwargs["bot_id"],
            user_id=kwargs["user_id"],
            tool_name=kwargs["tool_name"],
            subject=kwargs["subject"],
            prompt=kwargs["prompt"],
            severity=kwargs["severity"],
            policy_id=kwargs["policy_id"],
            session_key=kwargs.get("session_key"),
            backend=kwargs["backend"],
            continuation_capable=kwargs["continuation_capable"],
        )
        self.rows.append((row, kwargs))
        return (row, True) if with_created else row


def _policy(action: PolicyAction):
    return ApprovalPolicy(
        id="p1",
        tool_name="ops_run",
        matcher_type=MatcherType.ALWAYS,
        action=action,
    )


def _server(store, order, executed):
    async def publish(_payload):
        order.append("publish")

    mcp = ApprovalAwareFastMCP(
        "test",
        json_response=True,
        approval_store_provider=lambda: store,
        approval_publisher=publish,
        operations_preparer=lambda operation, args: {
            "operation_slug": operation,
            "args": args,
        },
    )

    @mcp.tool(name="ops_run")
    async def ops_run(operation: str, args: dict | None = None):
        executed.append((operation, args or {}))
        return {"ran": operation}

    return mcp


def _stamped_args(capability):
    clean = {"operation": "llm-bawt.restart-app", "args": {}}
    stamp = mint_mcp_call_context(
        capability=capability,
        tool_name="ops_run",
        tool_input=clean,
        tool_use_id="toolu_123",
        agent_request_id="req_123",
        session_key="snark:nick",
        backend="claude-code",
    )
    return {**clean, MCP_CALL_CONTEXT_KEY: stamp}


@pytest.fixture
def capability(monkeypatch):
    fernet = Fernet(Fernet.generate_key())
    monkeypatch.setattr(turn_codec, "_get_fernet", lambda: fernet)
    return turn_codec.mint_task_turn_context(**_turn_values())


def _call(mcp, capability, args):
    binding = task_association.set_current_task_turn_capability(capability)
    try:
        return asyncio.run(mcp.call_tool("ops_run", args))
    finally:
        task_association.reset_current_task_turn_capability(binding)


def _call_codex(mcp, capability, args, *, request_id="rpc-42", envelope=None):
    envelope = envelope or mint_mcp_request_context(
        capability=capability,
        agent_request_id="req_codex",
        session_key="codex:nick",
        backend="codex",
    )
    capability_binding = task_association.set_current_task_turn_capability(capability)
    request_binding = task_association.set_current_mcp_request_context(envelope)
    mcp.get_context = lambda: SimpleNamespace(request_id=request_id)
    try:
        return asyncio.run(mcp.call_tool("ops_run", args))
    finally:
        task_association.reset_current_mcp_request_context(request_binding)
        task_association.reset_current_task_turn_capability(capability_binding)


def test_allow_strips_reserved_context_before_binding_and_executes(capability):
    order, executed = [], []
    store = FakeStore(_policy(PolicyAction.ALLOW), order)
    result = _call(
        _server(store, order, executed), capability, _stamped_args(capability)
    )
    assert executed == [("llm-bawt.restart-app", {})]
    assert store.rows == []
    assert len(result) == 1
    assert '"ran": "llm-bawt.restart-app"' in result[0].text


def test_deny_never_executes_or_persists(capability):
    order, executed = [], []
    store = FakeStore(_policy(PolicyAction.DENY), order)
    result = _call(
        _server(store, order, executed), capability, _stamped_args(capability)
    )
    assert result["status"] == "denied"
    assert executed == []
    assert store.rows == []


def test_require_commits_before_publish_and_returns_pending(capability):
    order, executed = [], []
    store = FakeStore(_policy(PolicyAction.REQUIRE_APPROVAL), order)
    result = _call(
        _server(store, order, executed), capability, _stamped_args(capability)
    )
    assert result["status"] == "approval_required"
    assert "Do not retry" in result["message"]
    assert executed == []
    assert order == ["persist", "publish"]
    row, persisted = store.rows[0]
    assert persisted["tool_arguments"] == {
        "operation": "llm-bawt.restart-app",
        "args": {},
    }
    assert persisted["tool_use_id"] == "toolu_123"
    assert persisted["continuation_capable"] is True
    assert row.bot_id == "snark"


def test_routing_approval_displays_exact_target_image_and_delta(capability):
    from tests.test_ops_routing_reconcile import RoutingDocker, snapshot

    order, executed = [], []
    store = FakeStore(_policy(PolicyAction.REQUIRE_APPROVAL), order)
    mcp = _server(store, order, executed)
    approved = snapshot(RoutingDocker())
    mcp._operations_preparer = lambda operation, args: approved
    clean = {"operation": "bawthub.reconcile-prod-routing", "args": {}}
    stamp = mint_mcp_call_context(
        capability=capability, tool_name="ops_run", tool_input=clean,
        tool_use_id="toolu_routing", agent_request_id="req_routing",
        session_key="snark:nick", backend="claude-code",
    )
    result = _call(mcp, capability, {**clean, MCP_CALL_CONTEXT_KEY: stamp})
    assert result["status"] == "approval_required" and not executed
    prompt = store.rows[0][0].prompt
    for key in ("container_id", "image_ref", "config_hash", "before_rule", "after_rule"):
        assert approved["routing"][key] in prompt
    assert "No build, pull, environment, mount or permission changes" in prompt


def test_tampered_args_fail_before_policy_or_execution(capability):
    order, executed = [], []
    store = FakeStore(_policy(PolicyAction.ALLOW), order)
    args = _stamped_args(capability)
    args["operation"] = "llm-bawt.restart-redis"
    result = _call(_server(store, order, executed), capability, args)
    assert result["status"] == "approval_context_invalid"
    assert executed == []
    assert store.rows == []


def test_codex_require_persists_routable_context_once_per_protocol_call(capability):
    order, executed = [], []
    store = FakeStore(_policy(PolicyAction.REQUIRE_APPROVAL), order)
    mcp = _server(store, order, executed)
    clean = {"operation": "llm-bawt.restart-app", "args": {}}

    first = _call_codex(mcp, capability, clean)
    retry = _call_codex(mcp, capability, clean)

    assert first["approval_request_id"] == retry["approval_request_id"]
    assert order == ["persist", "publish", "persist"]
    assert len(store.rows) == 1
    _, persisted = store.rows[0]
    assert persisted["backend"] == "codex"
    assert persisted["bot_id"] == "snark"
    assert persisted["user_id"] == "nick"
    assert persisted["trigger_message_id"]
    assert persisted["session_key"] == "codex:nick"
    assert persisted["tool_use_id"].startswith("mcp-")
    assert persisted["continuation_capable"] is True
    assert executed == []


def test_contextual_call_missing_or_forged_header_executes_nothing(capability):
    order, executed = [], []
    store = FakeStore(_policy(PolicyAction.ALLOW), order)
    mcp = _server(store, order, executed)

    result = _call(mcp, capability, {"operation": "llm-bawt.restart-app"})
    assert result["status"] == "approval_context_missing"

    forged = mint_mcp_request_context(
        capability="wrong",
        agent_request_id="req_codex",
        session_key="codex:nick",
        backend="codex",
    )
    result = _call_codex(
        mcp,
        capability,
        {"operation": "llm-bawt.restart-app"},
        envelope=forged,
    )
    assert result["status"] == "approval_context_invalid"
    assert executed == []
    assert store.rows == []


def test_codex_context_missing_capability_or_invalid_turn_executes_nothing(capability):
    clean = {"operation": "llm-bawt.restart-app"}
    envelope = mint_mcp_request_context(
        capability=capability,
        agent_request_id="req_codex",
        session_key="codex:nick",
        backend="codex",
    )
    order, executed = [], []
    store = FakeStore(_policy(PolicyAction.ALLOW), order)
    mcp = _server(store, order, executed)

    request_binding = task_association.set_current_mcp_request_context(envelope)
    try:
        result = asyncio.run(mcp.call_tool("ops_run", clean))
    finally:
        task_association.reset_current_mcp_request_context(request_binding)
    assert result["status"] == "approval_context_missing"

    result = _call_codex(mcp, capability + "forged", clean, envelope=envelope)
    assert result["status"] == "approval_context_invalid"
    assert executed == []
    assert store.rows == []


def test_raw_mcp_non_gated_call_remains_allowed_but_gated_call_fails_closed():
    clean = {"operation": "llm-bawt.restart-app"}
    order, executed = [], []
    allowed = _server(FakeStore(_policy(PolicyAction.ALLOW), order), order, executed)
    result = asyncio.run(allowed.call_tool("ops_run", clean))
    assert executed == [("llm-bawt.restart-app", {})]
    assert len(result) == 1

    order, executed = [], []
    gated_store = FakeStore(_policy(PolicyAction.REQUIRE_APPROVAL), order)
    result = asyncio.run(
        _server(gated_store, order, executed).call_tool("ops_run", clean)
    )
    assert result["status"] == "approval_context_missing"
    assert executed == []
    assert gated_store.rows == []
