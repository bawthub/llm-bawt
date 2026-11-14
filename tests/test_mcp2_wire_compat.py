"""mcp 1.x behaviour the bawthub server keeps on mcp 2.x (TASK-2298).

Exercised through a real protocol session using the initialize handshake that
Claude and Codex use, not by calling Python methods directly.
"""

import json
import threading
from types import SimpleNamespace

import pytest
from agent_bridge.approval import ApprovalPolicy, PolicyAction
from mcp import Client

from llm_bawt.mcp_server.catalog import CatalogFastMCP


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _server(policies=()):
    store = SimpleNamespace(
        compile_bundle=lambda: SimpleNamespace(policies=list(policies)),
        record_decision=lambda **kwargs: None,
    )
    return CatalogFastMCP("wire-compat", approval_store_provider=lambda: store)


@pytest.mark.anyio
async def test_tool_exception_text_reaches_the_model():
    # 2.x would send only "Error executing tool refuse"; bots act on the reason.
    server = _server()

    @server.tool(name="refuse")
    def refuse() -> str:
        raise ValueError("[args_invalid] use args, not top-level fields")

    async with Client(server, mode="legacy") as client:
        result = await client.call_tool("refuse", {})
    assert result.is_error is True
    assert result.content[0].text == (
        "Error executing tool refuse: [args_invalid] use args, not top-level fields")


@pytest.mark.anyio
async def test_sync_tools_still_run_on_the_event_loop_thread():
    # 2.x moves sync tools to a worker thread; that change is not part of this migration.
    server = _server()
    seen = []

    @server.tool(name="where")
    def where() -> str:
        seen.append(threading.get_ident())
        return "ok"

    async with Client(server, mode="legacy") as client:
        await client.call_tool("where", {})
    assert seen == [threading.get_ident()]


@pytest.mark.anyio
async def test_policy_outcome_keeps_its_wire_form():
    server = _server([ApprovalPolicy(id="p1", tool_name="guarded", action=PolicyAction.DENY)])
    ran = []

    @server.tool(name="guarded", structured_output=False)
    async def guarded():
        ran.append(True)
        return "ran"

    async with Client(server, mode="legacy") as client:
        result = await client.call_tool("guarded", {})
    assert ran == []
    assert result.is_error is False
    assert result.structured_content["status"] == "denied"
    # Indented as 1.x wrote it; the catalog's compaction leaves policy outcomes alone.
    assert result.content[0].text == json.dumps(result.structured_content, indent=2)


@pytest.mark.anyio
async def test_policy_outcome_on_an_output_schema_tool_reaches_the_model():
    # 1.x failed output validation here and sent only "Output validation error".
    server = _server([ApprovalPolicy(id="p1", tool_name="guarded", action=PolicyAction.DENY)])

    @server.tool(name="guarded")
    async def guarded() -> str:  # str return => the tool has an output schema
        return "ran"

    async with Client(server, mode="legacy") as client:
        result = await client.call_tool("guarded", {})
    assert result.is_error is True
    assert result.structured_content is None
    assert json.loads(result.content[0].text)["status"] == "denied"
