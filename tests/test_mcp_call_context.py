from __future__ import annotations

import copy

import pytest

from agent_bridge.mcp_call_context import (
    MCP_CALL_CONTEXT_KEY,
    McpCallContextError,
    canonical_invocation_hash,
    derive_mcp_call_context,
    mint_mcp_call_context,
    mint_mcp_request_context,
    verify_mcp_call_context,
)


def _stamp(**over):
    values = {
        "capability": "opaque-trusted-turn-token",
        "tool_name": "ops_run",
        "tool_input": {"operation": "llm-bawt.restart-app", "args": {}},
        "tool_use_id": "toolu_123",
        "agent_request_id": "req_abc",
        "session_key": "snark:operator",
        "backend": "claude-code",
    }
    values.update(over)
    return values, mint_mcp_call_context(**values)


def test_round_trip_binds_exact_tool_and_args():
    values, stamp = _stamp()
    opened = verify_mcp_call_context(
        capability=values["capability"],
        tool_name=values["tool_name"],
        tool_input=values["tool_input"],
        raw_context=stamp,
    )
    assert opened.tool_use_id == "toolu_123"
    assert opened.agent_request_id == "req_abc"
    assert opened.session_key == "snark:operator"
    assert opened.invocation_hash == canonical_invocation_hash(
        "ops_run", values["tool_input"]
    )


def test_reserved_context_field_is_excluded_from_invocation_hash():
    base = {"operation": "x", "args": {"b": 2, "a": 1}}
    stamped = {**base, MCP_CALL_CONTEXT_KEY: {"forged": True}}
    assert canonical_invocation_hash("ops_run", base) == canonical_invocation_hash(
        "ops_run", stamped
    )


def test_tampered_args_or_tool_fail_closed():
    values, stamp = _stamp()
    with pytest.raises(McpCallContextError, match="does not match"):
        verify_mcp_call_context(
            capability=values["capability"],
            tool_name="ops_run",
            tool_input={"operation": "llm-bawt.restart-redis", "args": {}},
            raw_context=stamp,
        )
    with pytest.raises(McpCallContextError, match="does not match"):
        verify_mcp_call_context(
            capability=values["capability"],
            tool_name="ops_job_status",
            tool_input=values["tool_input"],
            raw_context=stamp,
        )


def test_forged_signature_or_wrong_capability_fails_closed():
    values, stamp = _stamp()
    forged = copy.deepcopy(stamp)
    forged["signature"] = "0" * 64
    with pytest.raises(McpCallContextError, match="signature is invalid"):
        verify_mcp_call_context(
            capability=values["capability"],
            tool_name=values["tool_name"],
            tool_input=values["tool_input"],
            raw_context=forged,
        )
    with pytest.raises(McpCallContextError, match="signature is invalid"):
        verify_mcp_call_context(
            capability="different-turn",
            tool_name=values["tool_name"],
            tool_input=values["tool_input"],
            raw_context=stamp,
        )


def test_codex_request_context_derives_stable_exact_call_identity():
    capability = "opaque-trusted-turn-token"
    raw = mint_mcp_request_context(
        capability=capability,
        agent_request_id="req_codex",
        session_key="codex:operator",
        backend="codex",
    )
    values = dict(
        capability=capability,
        raw_request_context=raw,
        protocol_request_id=42,
        tool_name="ops_run",
        tool_input={"operation": "llm-bawt.restart-app", "args": {}},
    )
    first = derive_mcp_call_context(**values)
    retry = derive_mcp_call_context(**values)
    other_call = derive_mcp_call_context(**{**values, "protocol_request_id": 43})

    assert first == retry
    assert first.backend == "codex"
    assert first.agent_request_id == "req_codex"
    assert first.tool_use_id != other_call.tool_use_id


def test_codex_request_context_rejects_forgery_and_sentinel_identity():
    with pytest.raises(McpCallContextError, match="missing or invalid"):
        mint_mcp_request_context(
            capability="cap",
            agent_request_id="unknown",
            session_key="codex:operator",
            backend="codex",
        )
    raw = mint_mcp_request_context(
        capability="cap",
        agent_request_id="req",
        session_key="codex:operator",
        backend="codex",
    )
    with pytest.raises(McpCallContextError, match="signature is invalid"):
        derive_mcp_call_context(
            capability="other",
            raw_request_context=raw,
            protocol_request_id=1,
            tool_name="ops_run",
            tool_input={},
        )
