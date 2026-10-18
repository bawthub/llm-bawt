"""Tests for the claude-code bridge approval gate decision logic (TASK-292).

Exercises the bridge's _decide_approval / grant store / _evaluate_tool_gate
against injected policy bundles and grants — no Redis, SDK import, app/DB
import, or HTTP. AST extraction executes the production mixin with only
stdlib dependencies and inert permission-result stand-ins.

Runnable under pytest, or standalone: ``python tests/test_approval_bridge_gate.py``.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import math
import shutil
import time
import types
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from agent_bridge.mcp_call_context import (
    MCP_CALL_CONTEXT_KEY, mint_mcp_call_context, verify_mcp_call_context,
)
from agent_bridge.approval import (
    ApprovalDecision, ApprovalPolicy, MatcherType, PolicyAction, PolicyBundle,
    evaluate as evaluate_policies, grant_key,
)
from agent_bridge.events import AgentEventKind
from claude_code_bridge.playwright_preflight import playwright_syntax_error  # noqa: F401 (AST test namespace)


# Extract the actual mixin without importing bridge.py's SDK/app/DB graph.
# This runs every gate test even in a stdlib-only bridge test environment.
_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = _ROOT / "src/claude_code_bridge/approval_ops.py"
_tree = ast.parse(_SOURCE.read_text())
_classes = [node for node in _tree.body if isinstance(node, ast.ClassDef)]


@dataclass
class PermissionResultAllow:
    pass


@dataclass
class PermissionResultDeny:
    message: str
    interrupt: bool = False


_namespace = dict(globals(), logger=logging.getLogger(__name__), COMMANDS_STREAM="commands")
exec(compile(ast.fix_missing_locations(ast.Module(
    body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *_classes],
    type_ignores=[],
)), str(_SOURCE), "exec"), _namespace)
_SDK_OK = True


class ClaudeCodeBridge(_namespace["ClaudeApprovalMixin"]):
    def __init__(self, publisher, *, backend_name, app_api_url):
        self._backend_name = backend_name
        self._app_api_url = app_api_url
        self._cwd = "/app"
        self._policy_bundle_ttl = 60
        self._approval_fail_closed = False
        self._policy_fetch_ok = True
        self._APPROVAL_PENDING_ACK = "Approval pending"
        self._approval_grants = {}


class _FakePub:
    class _R:
        connection_pool = None

    _redis = _R()

    def close(self):
        pass


def _bridge():
    b = ClaudeCodeBridge(_FakePub(), backend_name="claude-code", app_api_url="")
    pols = [
        ApprovalPolicy(id="allow1", order=10, tool_name="Bash",
                       matcher_type=MatcherType.PREFIX, pattern="rm -rf /tmp/safe",
                       action=PolicyAction.ALLOW),
        ApprovalPolicy(id="req1", order=100, tool_name="Bash",
                       matcher_type=MatcherType.PREFIX, pattern="rm -rf",
                       action=PolicyAction.REQUIRE_APPROVAL),
        ApprovalPolicy(id="deny1", order=5, tool_name="Bash",
                       matcher_type=MatcherType.CONTAINS, pattern="/dev/sda",
                       action=PolicyAction.DENY),
    ]
    b._policy_bundle = PolicyBundle(version=1, etag="x", policies=pols)
    b._policy_bundle_fetched_at = time.monotonic()  # fresh → no HTTP
    return b


def _skip_if_no_sdk():
    if not _SDK_OK:
        try:
            import pytest

            pytest.skip("claude_agent_sdk not importable (run inside bridge container)")
        except Exception:
            raise SystemExit(0)


def test_decide_allow_require_deny_default():
    _skip_if_no_sdk()
    b = _bridge()
    assert b._decide_approval("Bash", {"command": "rm -rf /tmp/safe/x"}).is_allowed
    assert b._decide_approval("Bash", {"command": "rm -rf /home/x"}).requires_approval
    assert b._decide_approval("Bash", {"command": "dd of=/dev/sda"}).is_denied
    assert b._decide_approval("Bash", {"command": "ls"}).is_allowed


def _continuation(approval_id="tuid-1"):
    return "req_delivery_approval_approval-cont-" + sha256(approval_id.encode()).hexdigest()[:32]


def _grant(b, gk, approval_id="tuid-1", session="s", origin="origin"):
    b._remember_approval(gk, session, approval_id, origin)
    assert b._grant_approval(gk, 600, session_key=session, request_id=approval_id)


def test_grant_one_shot_and_expiry():
    b = _bridge()
    gk = b._decide_approval("Bash", {"command": "rm -rf /home/x"}).grant_key
    assert not b._consume_grant(gk)
    _grant(b, gk)
    assert b._consume_grant(gk, session_key="s", request_id=_continuation())
    assert not b._consume_grant(gk, session_key="s", request_id=_continuation())
    _grant(b, "k2", approval_id="tuid-2")
    b._pending_approvals[("s", "tuid-2")].grant_expires_at = time.monotonic() - 1
    assert not b._consume_grant("k2", session_key="s", request_id=_continuation("tuid-2"))
    # An expired activation cannot be refreshed by delayed delivery.
    assert not b._grant_approval("k2", 600, session_key="s", request_id="tuid-2")


def test_gate_allow_deny_require_emits_event():
    _skip_if_no_sdk()
    b = _bridge()

    class Ctx:
        tool_use_id = "tuid-1"

    captured: list[dict] = []
    b._publish_event = lambda *a, **k: captured.append(k)  # type: ignore[assignment]
    seq = [0]

    async def gate(cmd):
        return await b._evaluate_tool_gate("Bash", {"command": cmd}, Ctx(), "req", "sess", seq)

    assert isinstance(asyncio.run(gate("ls")), PermissionResultAllow)
    assert isinstance(asyncio.run(gate("dd of=/dev/sda")), PermissionResultDeny)
    assert isinstance(asyncio.run(gate("rm -rf /home/x")), PermissionResultDeny)
    assert captured[-1]["kind"].value == "approval_required"
    assert captured[-1]["tool_use_id"] == "tuid-1"
    assert captured[-1]["extra_raw"]["grant_key"]


def test_gate_granted_reissue_allows():
    _skip_if_no_sdk()
    b = _bridge()

    class Ctx:
        tool_use_id = "tuid-1"

    b._publish_event = lambda *a, **k: None  # type: ignore[assignment]
    gk = b._decide_approval("Bash", {"command": "rm -rf /home/x"}).grant_key
    _grant(b, gk)

    async def gate():
        return await b._evaluate_tool_gate("Bash", {"command": "rm -rf /home/x"}, Ctx(), _continuation(), "s", [0])

    assert isinstance(asyncio.run(gate()), PermissionResultAllow)


def test_gate_fail_closed_denies_when_unreachable():
    _skip_if_no_sdk()
    b = _bridge()
    b._approval_fail_closed = True
    b._policy_fetch_ok = False

    class Ctx:
        tool_use_id = "t"

    async def gate():
        return await b._evaluate_tool_gate("Bash", {"command": "ls"}, Ctx(), "r", "s", [0])

    assert isinstance(asyncio.run(gate()), PermissionResultDeny)


def _hook_decision(out: dict) -> str | None:
    """Extract permissionDecision from a PreToolUse hook output dict."""
    return (out or {}).get("hookSpecificOutput", {}).get("permissionDecision")


def test_hook_gate_allow_deny_require_emits_event():
    _skip_if_no_sdk()
    b = _bridge()
    captured: list[dict] = []
    b._publish_event = lambda *a, **k: captured.append(k)  # type: ignore[assignment]
    seq = [0]

    async def gate(cmd):
        return await b._evaluate_tool_gate_hook(
            "Bash", {"command": cmd}, "tuid-1", "req", "sess", seq
        )

    # ALLOW → empty dict (no decision; tool proceeds)
    assert asyncio.run(gate("ls")) == {}
    # DENY → permissionDecision="deny"
    assert _hook_decision(asyncio.run(gate("dd of=/dev/sda"))) == "deny"
    # REQUIRE_APPROVAL with no grant → "deny" + APPROVAL_REQUIRED event
    assert _hook_decision(asyncio.run(gate("rm -rf /home/x"))) == "deny"
    assert captured[-1]["kind"].value == "approval_required"
    assert captured[-1]["tool_use_id"] == "tuid-1"
    assert captured[-1]["extra_raw"]["grant_key"]


def test_hook_gate_granted_reissue_allows():
    _skip_if_no_sdk()
    b = _bridge()
    b._publish_event = lambda *a, **k: None  # type: ignore[assignment]
    gk = b._decide_approval("Bash", {"command": "rm -rf /home/x"}).grant_key
    _grant(b, gk)

    async def gate():
        return await b._evaluate_tool_gate_hook(
            "Bash", {"command": "rm -rf /home/x"}, "tuid-1", _continuation(), "s", [0]
        )

    # consumed grant → allow (empty dict, no decision)
    assert asyncio.run(gate()) == {}


def test_hook_gate_fail_closed_denies_when_unreachable():
    _skip_if_no_sdk()
    b = _bridge()
    b._approval_fail_closed = True
    b._policy_fetch_ok = False

    async def gate():
        return await b._evaluate_tool_gate_hook("Bash", {"command": "ls"}, "t", "r", "s", [0])

    assert _hook_decision(asyncio.run(gate())) == "deny"


def test_hook_gate_no_tool_use_id_still_denies_on_require():
    _skip_if_no_sdk()
    b = _bridge()
    b._publish_event = lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not publish"))  # type: ignore[assignment]

    async def gate():
        # empty tool_use_id → can't persist a request row, but must still block
        return await b._evaluate_tool_gate_hook("Bash", {"command": "rm -rf /home/x"}, "", "r", "s", [0])

    assert _hook_decision(asyncio.run(gate())) == "deny"


def test_pre_tool_use_hook_passes_through_ask_user_question():
    _skip_if_no_sdk()
    b = _bridge()
    hook = b._make_pre_tool_use_hook(request_id="r", session_key="s", seq_holder=[0])

    async def run(name):
        return await hook({"tool_name": name, "tool_input": {}, "tool_use_id": "x"}, "x", {})

    # AskUserQuestion (bare + MCP-namespaced) must NOT be gated — handed back to SDK
    assert asyncio.run(run("AskUserQuestion")) == {}
    assert asyncio.run(run("mcp__foo__AskUserQuestion")) == {}


def test_pre_tool_use_hook_gates_dangerous_bash():
    _skip_if_no_sdk()
    b = _bridge()
    b._publish_event = lambda *a, **k: None  # type: ignore[assignment]
    hook = b._make_pre_tool_use_hook(request_id="r", session_key="s", seq_holder=[0])

    async def run(cmd):
        return await hook(
            {"tool_name": "Bash", "tool_input": {"command": cmd}, "tool_use_id": "x"},
            "x", {},
        )

    assert asyncio.run(run("ls")) == {}                                  # allow
    assert _hook_decision(asyncio.run(run("dd of=/dev/sda"))) == "deny"  # deny
    assert _hook_decision(asyncio.run(run("rm -rf /home/x"))) == "deny"  # require→deny


def test_pre_tool_use_playwright_syntax_error_is_model_visible_and_does_not_execute():
    if not shutil.which("node"):
        return  # Other CI environments may delegate validation to Playwright.
    b = _bridge()
    published = []
    b._publish_event = lambda *a, **k: published.append(k)  # type: ignore[assignment]
    hook = b._make_pre_tool_use_hook(request_id="r", session_key="s", seq_holder=[0])

    async def run(name, args):
        return await hook({"tool_name": name, "tool_input": args, "tool_use_id": "x"}, "x", {})

    tool = "mcp__playwright__browser_evaluate"
    denied = asyncio.run(run(tool, {"function": "() => { return 1; } [sic]"}))
    assert _hook_decision(denied) == "deny"
    reason = denied["hookSpecificOutput"]["permissionDecisionReason"]
    assert "`function`" in reason and "not executed" in reason and "NEW tool call" in reason
    assert not any(e["kind"] == AgentEventKind.TOOL_EXECUTING for e in published)
    syntax = asyncio.run(run(tool, {"function": "() => { return 1; } stray_text"}))
    assert _hook_decision(syntax) == "deny"
    assert "Unexpected identifier 'stray_text'" in syntax["hookSpecificOutput"]["permissionDecisionReason"]
    assert asyncio.run(run(tool, {"function": "() => { return 1; }"})) == {}
    assert any(e["kind"] == AgentEventKind.TOOL_EXECUTING for e in published)
    # Approval policy still gates a syntactically valid call; preflight must
    # never turn a denial into an allow.
    b._policy_bundle.policies.insert(0, ApprovalPolicy(
        id="deny-browser", order=0, tool_name=tool,
        matcher_type=MatcherType.ALWAYS, action=PolicyAction.DENY,
    ))
    assert _hook_decision(asyncio.run(run(tool, {"function": "() => 1"}))) == "deny"
    surrogate = asyncio.run(run(tool, {"function": "() => 1\ud800"}))
    assert _hook_decision(surrogate) == "deny"
    assert "Playwright JavaScript" not in surrogate["hookSpecificOutput"]["permissionDecisionReason"]


def test_pre_tool_use_hook_stamps_bawthub_mcp_without_bridge_gating():
    _skip_if_no_sdk()
    b = _bridge()
    capability = "opaque-turn-capability"
    hook = b._make_pre_tool_use_hook(
        request_id="req-1",
        session_key="snark:nick",
        seq_holder=[0],
        task_turn_capability=capability,
    )
    clean = {"operation": "llm-bawt.restart-app", "args": {}}

    async def run():
        return await hook(
            {
                "tool_name": "mcp__bawthub__ops_run",
                "tool_input": clean,
                "tool_use_id": "toolu-1",
            },
            "toolu-1",
            {},
        )

    output = asyncio.run(run())
    specific = output["hookSpecificOutput"]
    assert "permissionDecision" not in specific
    updated = specific["updatedInput"]
    assert updated["operation"] == "llm-bawt.restart-app"
    stamp = updated[MCP_CALL_CONTEXT_KEY]
    verified = verify_mcp_call_context(
        capability=capability,
        tool_name="ops_run",
        tool_input=clean,
        raw_context=stamp,
    )
    assert verified.tool_use_id == "toolu-1"
    assert verified.agent_request_id == "req-1"
    assert verified.session_key == "snark:nick"


def test_pre_tool_use_hook_fails_open_on_unexpected_error():
    _skip_if_no_sdk()
    b = _bridge()

    def boom(*a, **k):
        raise RuntimeError("bundle blew up")

    b._get_policy_bundle = boom  # type: ignore[assignment]
    hook = b._make_pre_tool_use_hook(request_id="r", session_key="s", seq_holder=[0])

    async def run():
        return await hook(
            {"tool_name": "Bash", "tool_input": {"command": "ls"}, "tool_use_id": "x"},
            "x", {},
        )

    # default posture: gate error must NOT wedge tools
    assert asyncio.run(run()) == {}
    # fail-closed posture: gate error blocks instead
    b._approval_fail_closed = True
    assert _hook_decision(asyncio.run(run())) == "deny"


def test_grants_require_pending_session_and_approval_id():
    b = _bridge()
    gk = b._decide_approval("Bash", {"command": "rm -rf /home/x"}).grant_key
    assert not b._grant_approval(gk, 600)
    assert not b._grant_approval(gk, 600, session_key="s", request_id="unknown")
    b._remember_approval(gk, "s", "a", "origin")
    for key, session, approval in (("wrong", "s", "a"), (gk, "other", "a"), (gk, "s", "b")):
        assert not b._grant_approval(key, 600, session_key=session, request_id=approval)
    assert b._grant_approval(gk, 600, session_key="s", request_id="a")
    for session, request in (("other", _continuation("a")), ("s", "origin"),
                             ("s", "unrelated-later-turn"), ("s", _continuation("b"))):
        assert not b._consume_grant(gk, session_key=session, request_id=request)
    assert b._consume_grant(gk, session_key="s", request_id=_continuation("a"))


def test_duplicate_deliveries_never_extend_or_resurrect_grants():
    b = _bridge()
    _grant(b, "g")
    record = b._pending_approvals[("s", "tuid-1")]
    expiry = record.grant_expires_at
    assert not b._grant_approval("g", 9999, session_key="s", request_id="tuid-1")
    assert record.grant_expires_at == expiry
    assert b._consume_grant("g", session_key="s", request_id=_continuation())
    b._remember_approval("g", "s", "tuid-1", "different-origin")
    assert record.origin_request_id == "origin"
    assert not b._grant_approval("g", 600, session_key="s", request_id="tuid-1")
    assert not b._consume_grant("g", session_key="s", request_id=_continuation())
    # A second explicitly approved request for the identical call is independent.
    _grant(b, "g", approval_id="second")
    assert not b._consume_grant("g", session_key="s", request_id=_continuation())
    assert b._consume_grant("g", session_key="s", request_id=_continuation("second"))


def test_invalid_ttl_and_expired_pending_authority_fail_closed():
    b = _bridge()
    b._remember_approval("g", "s", "a", "origin")
    for ttl in (0, -1, float("nan"), float("inf"), -float("inf")):
        assert not b._grant_approval("g", ttl, session_key="s", request_id="a")
    b._pending_approvals[("s", "a")].expires_at = time.monotonic() - 1
    assert not b._grant_approval("g", 600, session_key="s", request_id="a")
    assert ("s", "a") not in b._pending_approvals


def test_live_hook_exact_reissue_and_hard_deny_wins():
    for hook_gate in (True, False):
        b = _bridge()
        b._policy_bundle = PolicyBundle(version=1, etag="all", policies=[
            ApprovalPolicy(id="all", tool_name="Bash", matcher_type=MatcherType.ALWAYS),
        ])
        events = []
        b._publish_event = lambda *a, **kw: events.append(kw)
        original = {"command": "rm -rf /home/x", "timeout": 100}

        def gate(arguments, request="origin", session="s", cwd="/repo"):
            if hook_gate:
                hook = b._make_pre_tool_use_hook(request_id=request, session_key=session, seq_holder=[0])
                result = asyncio.run(hook({"tool_name": "Bash", "tool_input": arguments, "cwd": cwd}, "tuid-1", {}))
                return _hook_decision(result) != "deny"
            result = asyncio.run(b._evaluate_tool_gate("Bash", arguments, types.SimpleNamespace(tool_use_id="tuid-1"),
                                                      request, session, [0], cwd=cwd))
            return isinstance(result, PermissionResultAllow)

        assert not gate(original)
        gk = events[-1]["extra_raw"]["grant_key"]
        assert b._grant_approval(gk, 600, session_key="s", request_id="tuid-1")
        assert not gate(original, request="unrelated")
        assert not gate(original, request=_continuation(), session="other")
        assert not gate(original, request=_continuation(), cwd="/elsewhere")
        assert not gate({**original, "timeout": 101}, request=_continuation())
        assert not gate({**original, "command": "rm  -rf /home/x"}, request=_continuation())
        b._policy_bundle.policies.insert(0, ApprovalPolicy(id="hard", order=0, tool_name="Bash",
            matcher_type=MatcherType.ALWAYS, action=PolicyAction.DENY))
        assert not gate(original, request=_continuation())
        assert b._pending_approvals[("s", "tuid-1")].state == "granted"
        b._policy_bundle.policies.pop(0)
        assert gate(original, request=_continuation())
        # TASK-1095: the hook path then stamps the real execution start.
        tail = [AgentEventKind.TOOL_PREAPPROVED] + ([AgentEventKind.TOOL_EXECUTING] if hook_gate else [])
        assert [e["kind"] for e in events[-len(tail):]] == tail
        assert not gate(original, request=_continuation())


def test_legacy_stored_key_cannot_authorize_exact_pending_invocation():
    b = _bridge()
    original = {"command": "rm -rf /home/x"}
    decision = b._decide_approval("Bash", original)
    legacy = grant_key("claude-code", "Bash", decision.subject)
    b._approval_grants[legacy] = time.monotonic() + 600
    b._remember_approval(decision.grant_key, "s", "a", "origin")
    assert not b._grant_approval(legacy, 600, session_key="s", request_id="a")
    assert not b._consume_grant(decision.grant_key, session_key="s", request_id=_continuation("a"))


def test_invalid_invocation_identity_denies_even_in_fail_open_mode():
    b = _bridge()
    hook = b._make_pre_tool_use_hook(request_id="r", session_key="s", seq_holder=[0])
    circular = {}
    circular["self"] = circular
    for arguments in (None, [], circular, {"command": "ls", "bad": object()},
                      {"command": "ls", "bad": float("nan")},
                      {"command": "ls", "bad": {1: "value"}},
                      {"command": "ls", "bad": (1, 2)}):
        result = asyncio.run(hook({"tool_name": "Bash", "tool_input": arguments}, "t", {}))
        assert _hook_decision(result) == "deny"
    for cwd in (None, "", 123, []):
        result = asyncio.run(hook({"tool_name": "Bash", "tool_input": {"command": "ls"}, "cwd": cwd}, "t", {}))
        assert _hook_decision(result) == "deny"


def test_backend_record_request_preserves_key_and_grant_roundtrip():
    # The real store method, isolated at its SQL session boundary. No app import
    # or inherited DB settings can create a live connection during collection.
    path = _ROOT / "src/llm_bawt/approval_request_store.py"
    store_class = next(n for n in ast.parse(path.read_text()).body
                       if isinstance(n, ast.ClassDef) and n.name == "ApprovalRequestStoreMixin")
    method = next(n for n in store_class.body if isinstance(n, ast.FunctionDef) and n.name == "record_request")
    rows = {}

    class Session:
        def __init__(self, engine):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, model, key):
            return rows.get(key)

        def add(self, row):
            rows[row.id] = row

        def commit(self):
            pass

        def refresh(self, row):
            pass

    namespace = dict(json=json, Session=Session, ToolApprovalRequest=types.SimpleNamespace,
                     REQ_PENDING="pending", ApprovalPersistError=RuntimeError,
                     logger=logging.getLogger(__name__))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    b = _bridge()
    original = {"command": "rm -rf /home/x", "timeout": 100}
    decision = b._decide_approval("Bash", original, cwd="/repo")
    legacy = evaluate_policies(b._policy_bundle.policies, "claude-code", "Bash", original)
    assert legacy.grant_key == grant_key("claude-code", "Bash", legacy.subject)
    for request_id, key in (("exact", decision.grant_key), ("legacy", legacy.grant_key)):
        row = namespace["record_request"](types.SimpleNamespace(engine=object()),
            request_id=request_id, bot_id="b", user_id="u", turn_id="origin", backend="claude-code",
            tool_name="Bash", tool_arguments=original, subject=decision.subject, grant_key=key,
            policy_id="req1", severity="high", prompt="Approve?", session_key="s")
        assert row.grant_key == key
        assert json.loads(row.tool_arguments_json) == original
        b._remember_approval(decision.grant_key, "s", request_id, "origin")
        accepted = b._grant_approval(row.grant_key, 600, session_key=row.session_key, request_id=row.id)
        assert accepted == (request_id == "exact")
    assert b._consume_grant(decision.grant_key, session_key="s", request_id=_continuation("exact"))


def test_grant_transport_ack_and_backend_continuation_identity():
    b = _bridge()
    b._remember_approval("g", "s", "a", "origin")
    acks = []

    class Redis:
        async def xack(self, *args):
            acks.append(args)

    fields = {"grant_key": "g", "session_key": "s", "request_id": "a", "ttl_seconds": "600"}
    for message in (fields, fields, {**fields, "ttl_seconds": "garbage"}):
        asyncio.run(b._handle_approval_grant(message, "msg", Redis()))
    assert len(acks) == 3
    # Execute the real backend ID helpers, not a duplicated test formula.
    from uuid import NAMESPACE_URL, uuid5

    namespace = {"sha256": sha256, "uuid5": uuid5, "NAMESPACE_URL": NAMESPACE_URL}
    for relative, symbol in (("llm_bawt/approval_request_store.py", "_continuation_id"),
                             ("llm_bawt/service/approval_continuations.py", "_continuation_identity")):
        path = _ROOT / "src" / relative
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == symbol)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    row = types.SimpleNamespace(continuation_id=namespace["_continuation_id"]("a"))
    identity = namespace["_continuation_identity"](row)["inter_bot_bridge_request_id"]
    assert b._consume_grant("g", session_key="s", request_id=identity)
    asyncio.run(b._handle_approval_grant(fields, "duplicate", Redis()))
    assert not b._consume_grant("g", session_key="s", request_id=identity)


if __name__ == "__main__":
    import sys
    import traceback

    if not _SDK_OK:
        print("SKIP: claude_agent_sdk not importable here")
        sys.exit(0)
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
        except SystemExit:
            print(f"SKIP {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


def test_pre_tool_use_hook_stamps_execution_only_when_tool_will_run():
    """TASK-1095 #3: PreToolUse = execution start, never for a denied call."""
    _skip_if_no_sdk()
    b = _bridge()
    published = []
    b._publish_event = lambda *a, **k: published.append(k)  # type: ignore[assignment]
    seq = [0]
    hook = b._make_pre_tool_use_hook(request_id="r", session_key="s", seq_holder=seq)

    def executing():
        return [k for k in published if k.get("kind") == AgentEventKind.TOOL_EXECUTING]

    async def run(cmd, tuid):
        return await hook(
            {"tool_name": "Bash", "tool_input": {"command": cmd}, "tool_use_id": tuid}, tuid, {},
        )

    assert asyncio.run(run("ls", "t-allow")) == {}
    assert [(k["tool_use_id"], k["tool_name"]) for k in executing()] == [("t-allow", "Bash")]
    assert seq[0] >= 1
    asyncio.run(run("dd of=/dev/sda", "t-deny"))        # deny
    asyncio.run(run("rm -rf /home/x", "t-require"))     # require → deny pending approval
    assert [k["tool_use_id"] for k in executing()] == ["t-allow"]
