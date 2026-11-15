from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from agent_bridge.mcp_call_context import (
    MCP_REQUEST_CONTEXT_ENV,
    MCP_TASK_TURN_CONTEXT_ENV,
    verify_mcp_request_context,
)
from codex_bridge.mcp_context import codex_mcp_environment


def test_codex_mcp_environment_is_request_local_and_does_not_mutate_base():
    base = {"PATH": "/bin", "UNCHANGED": "yes"}
    first = codex_mcp_environment(
        base,
        capability="cap-one",
        agent_request_id="req-one",
        session_key="codex:operator:one",
        backend="codex",
    )
    second = codex_mcp_environment(
        base,
        capability="cap-two",
        agent_request_id="req-two",
        session_key="codex:operator:two",
        backend="codex",
    )

    assert base == {"PATH": "/bin", "UNCHANGED": "yes"}
    assert first[MCP_TASK_TURN_CONTEXT_ENV] == "cap-one"
    assert second[MCP_TASK_TURN_CONTEXT_ENV] == "cap-two"
    assert first[MCP_REQUEST_CONTEXT_ENV] != second[MCP_REQUEST_CONTEXT_ENV]
    assert (
        verify_mcp_request_context(
            capability="cap-one", raw_context=first[MCP_REQUEST_CONTEXT_ENV]
        ).session_key
        == "codex:operator:one"
    )


def test_concurrent_codex_environments_do_not_cross_sessions():
    def build(index: int):
        capability = f"cap-{index}"
        environment = codex_mcp_environment(
            {"BASE": "value"},
            capability=capability,
            agent_request_id=f"req-{index}",
            session_key=f"codex:operator:{index}",
            backend="codex",
        )
        opened = verify_mcp_request_context(
            capability=capability,
            raw_context=environment[MCP_REQUEST_CONTEXT_ENV],
        )
        return environment, opened

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(build, range(32)))

    assert len({env[MCP_REQUEST_CONTEXT_ENV] for env, _ in results}) == 32
    assert [opened.session_key for _, opened in results] == [
        f"codex:operator:{index}" for index in range(32)
    ]
