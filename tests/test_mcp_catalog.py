"""Catalog size, semantic compatibility, skill discovery and dispatch contracts."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import tiktoken
from mcp.server.fastmcp import FastMCP

from llm_bawt.mcp_server.approval_interceptor import ApprovalAwareFastMCP
from llm_bawt.mcp_server.catalog import CatalogFastMCP, compact_schema
from llm_bawt.mcp_server.catalog_contracts import TOOL_SUMMARIES, tool_reference
from llm_bawt.mcp_server.server import mcp


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_catalog_covers_every_registered_tool_without_changing_validation():
    original = await FastMCP.list_tools(mcp)
    compact = await mcp.list_tools()
    assert [tool.name for tool in original] == [tool.name for tool in compact]
    assert set(TOOL_SUMMARIES) == {tool.name for tool in original}
    for before, after in zip(original, compact, strict=True):
        assert after.inputSchema == compact_schema(before.inputSchema)
        # Everything outside the two presentation fields stays identical.
        excluded = {"description", "inputSchema"}
        assert before.model_dump(exclude=excluded) == after.model_dump(exclude=excluded)
        assert after.description.endswith(f"Details: {tool_reference(after.name)}.")
    # Reading discovery must not mutate FastMCP's actual validator schemas.
    assert original == await FastMCP.list_tools(mcp)
    assert CatalogFastMCP.call_tool is ApprovalAwareFastMCP.call_tool
    assert CatalogFastMCP.call_approved_tool is ApprovalAwareFastMCP.call_approved_tool


def test_schema_compaction_preserves_title_fields_and_arbitrary_data():
    schema = {
        "title": "Root generated title", "type": "object",
        "properties": {
            "title": {"title": "Title annotation", "type": "string", "minLength": 1},
            "payload": {"type": "object", "default": {"title": "KEEP"},
                        "const": {"title": "KEEP"}, "examples": [{"title": "KEEP"}]},
            "optional": {"anyOf": [{"title": "String", "type": "string"}, {"type": "null"}], "default": None},
            "ref": {"$ref": "#/$defs/Row"},
        },
        "required": ["title"], "additionalProperties": False,
        "$defs": {"Row": {"title": "Row", "type": "object", "properties": {"title": {"type": "string"}}}},
        "description": "KEEP semantic description",
    }
    before = deepcopy(schema)
    result = compact_schema(schema)
    assert schema == before
    assert "title" not in result
    assert "title" in result["properties"]
    assert result["properties"]["title"] == {"type": "string", "minLength": 1}
    assert result["properties"]["payload"] == before["properties"]["payload"]
    assert result["properties"]["optional"]["anyOf"] == [{"type": "string"}, {"type": "null"}]
    assert result["properties"]["optional"]["default"] is None
    assert result["properties"]["ref"] == {"$ref": "#/$defs/Row"}
    assert result["required"] == ["title"]
    assert result["additionalProperties"] is False
    assert "title" in result["$defs"]["Row"]["properties"]
    assert "title" not in result["$defs"]["Row"]
    assert result["description"] == schema["description"]


def test_nested_schema_keywords_and_data_are_distinct():
    schema = {
        "dependentSchemas": {"title": {"title": "annotation", "properties": {"title": {"type": "string"}}}},
        "patternProperties": {"title": {"title": "annotation", "type": "string"}},
        "if": {"title": "annotation", "const": {"title": "data"}},
        "then": {"not": {"title": "annotation", "enum": [{"title": "data"}]}},
        "items": [{"title": "annotation", "type": "string"}],
        "prefixItems": [{"title": "annotation", "type": "integer"}],
    }
    result = compact_schema(schema)
    assert "title" in result["dependentSchemas"]
    assert "title" not in result["dependentSchemas"]["title"]
    assert result["patternProperties"]["title"] == {"type": "string"}
    assert result["if"] == {"const": {"title": "data"}}
    assert result["then"]["not"] == {"enum": [{"title": "data"}]}
    assert result["items"] == [{"type": "string"}]
    assert result["prefixItems"] == [{"type": "integer"}]


@pytest.mark.anyio
async def test_catalog_token_budgets():
    encoder = tiktoken.get_encoding("o200k_base")
    tools = await mcp.list_tools()
    descriptions = [len(encoder.encode(tool.description or "")) for tool in tools]
    assert max(descriptions) <= 100
    # x_counts + x_search relevancy/operator hints: measured 3,251 (was <=3,200).
    # TASK-1047 self_model (+48 description tokens): measured 3,306.
    assert sum(descriptions) <= 3310
    definitions = [{"name": f"mcp__bawthub__{t.name}", "description": t.description or "", "input_schema": t.inputSchema} for t in tools]
    total = len(encoder.encode(json.dumps(definitions, ensure_ascii=False, separators=(",", ":"))))
    # TASK-900: explicit X search adds ~160 tokens; measured catalog 11,239.
    # Keep its paid-call warning and typed time/pagination fields intact.
    # x_counts tool + x_search sort_order/include_authors: measured 11,425.
    # TASK-1047 self_model (+146 tokens incl. typed action enum): measured 11,618.
    assert total <= 11625, f"Catalog grew to {total} tokens; see docs/MCP_TOOL_DESIGN.md"
    assert all("Args:" not in (tool.description or "") for tool in tools)


@pytest.mark.parametrize("name,words", [
    ("tasks_update", ("REVIEW", "COMPLETED", "BUG")),
    ("tasks_associate_current", ("actually", "mention", "fails closed")),
    ("steps_set", ("ENTIRE", "deleting", "[]")),
    ("tasks_delete", ("Permanently", "CANCELLED")),
    ("projects_delete", ("Permanently", "unassigned")),
    ("memory_clear", ("ALL", "Destructive")),
    ("messages_clear", ("ALL", "Destructive")),
    ("messages_restore_ignored", ("ALL",)),
    ("messages_remove_last_partial", ("does NOT verify partial",)),
    ("sessions_rotate", ("active thread",)),
    ("self_system_prompt", ("fully replace", "own")),
    ("self_model", ("persistently", "next turn", "when_idle")),
    ("self_tail", ("do not reprint",)),
    ("bots_send_message", ("async", "when_idle", "force never")),
    ("bots_delivery_cancel", ("QUEUED", "cannot abort")),
    ("ops_run", ("Approval/queued is not success", "Never bypass")),
    ("media_add", ("downloading", "first match")),
])
def test_call_critical_warnings_stay_inline(name, words):
    for word in words:
        assert word in TOOL_SUMMARIES[name]


@pytest.mark.anyio
async def test_unknown_tools_remain_discoverable_and_callables_unchanged():
    server = CatalogFastMCP("test-catalog")

    @server.tool(name="new_unmigrated_tool")
    async def sample(title: str = "hello") -> str:
        """An unconverted tool's description must not disappear."""
        return title

    tool = (await server.list_tools())[0]
    assert tool.description == "An unconverted tool's description must not disappear."
    assert tool.inputSchema["properties"]["title"]["default"] == "hello"
    # Isolated pure callable only; direct base dispatch tests unchanged binding.
    result = await FastMCP.call_tool(server, "new_unmigrated_tool", {"title": "preserved"})
    assert "preserved" in str(result)
    assert await sample() == "hello"


@pytest.mark.anyio
async def test_real_mutation_still_passes_through_policy_gate(monkeypatch):
    import llm_bawt.mcp_server.approval_interceptor as interceptor
    from agent_bridge.approval import PolicyAction

    store = SimpleNamespace(compile_bundle=lambda: SimpleNamespace(policies=[]), record_decision=lambda **kw: None)
    server = CatalogFastMCP("test-denial", approval_store_provider=lambda: store)
    called = []

    @server.tool(name="tasks_delete")
    async def delete(task_id: str):
        called.append(task_id)
        return {"ok": True}

    monkeypatch.setattr(interceptor, "evaluate", lambda *args: SimpleNamespace(action=PolicyAction.DENY, subject="test"))
    await server.list_tools()
    result = await server.call_tool("tasks_delete", {"task_id": "test-task"})
    assert result["status"] == "denied"
    assert called == []


@pytest.mark.anyio
async def test_mcp_protocol_advertises_compact_catalog():
    from mcp.shared.memory import create_connected_server_and_client_session

    # Exercises the MCP handler bound during FastMCP initialization, not just
    # direct Python calls. No DB, model execution or external service involved.
    async with create_connected_server_and_client_session(mcp) as session:
        result = await session.list_tools()
    assert len(result.tools) == len(TOOL_SUMMARIES)
    task = next(t for t in result.tools if t.name == "tasks_create")
    assert "Details: bawthub-mcp/" in task.description
    assert "title" in task.inputSchema["properties"]
    assert "title" not in task.inputSchema


def test_skill_references_resolve_when_skills_checkout_is_present():
    root = Path(__file__).resolve().parents[2] / "agent-skills"
    if not root.is_dir():
        pytest.skip("Separate agent-skills checkout absent; validate references when publishing")
    skill = root / "bawthub-mcp" / "SKILL.md"
    assert skill.is_file()
    assert "name: bawthub-mcp" in skill.read_text()
    for name in TOOL_SUMMARIES:
        reference = root / tool_reference(name)
        assert reference.is_file(), reference
        assert reference.stat().st_size > 200
