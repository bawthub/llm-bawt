"""TASK-2653: memory_system_enabled=false hides and refuses memory tools,
skips extraction and refuses the context_recall memories source."""

import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from llm_bawt import memory_switch
from llm_bawt.mcp_server.server import mcp


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def memory_off(monkeypatch):
    monkeypatch.setattr(memory_switch, "_override", False)


def test_switch_ships_on_and_is_bot_overridable():
    from llm_bawt.setting_definitions import SETTING_DEFINITIONS

    definition = SETTING_DEFINITIONS["memory_system_enabled"]
    assert definition.default is True
    assert set(definition.applies_to) == {"chat", "agent"}


class FakeStore:
    """RuntimeSettingsStore.get_raw_value_json stand-in: rows by (scope, id)."""

    def __init__(self, rows=None, engine=True):
        self.rows, self.engine = rows or {}, engine

    def get_raw_value_json(self, scope_type, scope_id, key):
        if not self.engine:
            raise RuntimeError("Runtime settings DB unavailable")
        assert key == "memory_system_enabled"
        return self.rows.get((scope_type, scope_id))


@pytest.fixture
def store(monkeypatch):
    from llm_bawt.service import dependencies

    holder = {"store": FakeStore()}
    monkeypatch.setattr(memory_switch, "_override", None)
    monkeypatch.setattr(dependencies, "get_runtime_settings_store", lambda config: holder["store"])
    return holder


def test_absent_rows_default_on(store):
    assert memory_switch.memory_system_enabled(object(), "sample") is True


def test_bot_row_wins_over_global(store):
    store["store"] = FakeStore({("global", "*"): "false", ("bot", "kept"): "true"})
    assert memory_switch.memory_system_enabled(object()) is False
    assert memory_switch.memory_system_enabled(object(), "Kept") is True
    assert memory_switch.memory_system_enabled(object(), "other") is False


@pytest.mark.parametrize("rows,engine", [
    ({}, False),                                 # store has no engine
    ({("global", "*"): "not json"}, True),       # undecodable row
    ({("global", "*"): '"yes"'}, True),          # truthy non-boolean
    ({("bot", "sample"): "1"}, True),            # non-boolean bot row
])
def test_unreadable_or_invalid_counts_as_off(store, rows, engine):
    store["store"] = FakeStore(rows, engine)
    assert memory_switch.memory_system_enabled(object(), "sample") is False


@pytest.mark.anyio
async def test_approved_call_uses_recorded_caller(store, monkeypatch):
    from llm_bawt.mcp_server.approval_interceptor import ApprovalAwareFastMCP, ApprovedCallerContext

    store["store"] = FakeStore({("bot", "off-bot"): "false", ("global", "*"): "false",
                                ("bot", "kept"): "true"})
    reached = []

    async def fake_approved(self, name, arguments, **kwargs):
        reached.append(kwargs["caller_context"].bot_id)
        from mcp.types import CallToolResult
        return CallToolResult(content=[])

    monkeypatch.setattr(ApprovalAwareFastMCP, "call_approved_tool", fake_approved)
    with pytest.raises(ToolError, match="memory system is disabled"):
        await mcp.call_approved_tool("memory_store", {}, expected_invocation_hash="h",
                                     caller_context=ApprovedCallerContext(bot_id="off-bot"))
    await mcp.call_approved_tool("memory_store", {}, expected_invocation_hash="h",
                                 caller_context=ApprovedCallerContext(bot_id="kept"))
    assert reached == ["kept"]


def test_off_skips_memory_maintenance(memory_off):
    from types import SimpleNamespace

    from llm_bawt.service.background_tasks import BackgroundTasksMixin

    processor = BackgroundTasksMixin()
    processor.config = SimpleNamespace()
    processor.get_memory_client = lambda bot_id: pytest.fail("must not reach memory client")
    task = SimpleNamespace(bot_id="sample", payload={})
    assert processor._process_maintenance(task)["skipped"] is True


def test_maintenance_tool_is_a_memory_tool():
    assert memory_switch.is_memory_tool("system_run_maintenance")


@pytest.mark.anyio
async def test_off_hides_memory_tools(memory_off):
    names = {tool.name for tool in await mcp.list_tools()}
    assert not {name for name in names if memory_switch.is_memory_tool(name)}
    assert {"context_recall", "messages_find", "working_memory_list"} <= names


@pytest.mark.anyio
async def test_on_lists_memory_tools():
    names = {tool.name for tool in await mcp.list_tools()}
    assert {"memory_search_all", "facts_extract"} <= names


@pytest.mark.anyio
async def test_off_refuses_memory_tool_calls(memory_off):
    with pytest.raises(ToolError, match="memory system is disabled"):
        await mcp.call_tool("memory_search_all", {"query": "x"})


@pytest.mark.anyio
async def test_off_refuses_recall_memories_source(memory_off, monkeypatch):
    from llm_bawt.mcp_server import recall_tools
    from llm_bawt.recall.contracts import RecallRequest

    monkeypatch.setattr(recall_tools, "_runtime", lambda: pytest.fail("must not reach providers"))
    from types import SimpleNamespace
    monkeypatch.setattr(recall_tools, "current_task_turn_context", lambda: SimpleNamespace(bot_id="sample"))
    result = json.loads(await recall_tools.context_recall(
        RecallRequest(query="audit", sources=("messages", "memories"))))
    assert result["status"] == "error"
    assert result["error_code"] == "source_unavailable"


def test_off_skips_extraction(memory_off):
    from types import SimpleNamespace

    from llm_bawt.service.background_tasks import BackgroundTasksMixin

    processor = BackgroundTasksMixin()
    processor.config = SimpleNamespace(MEMORY_EXTRACTION_ENABLED=True)
    processor._default_bot = "sample"
    task = SimpleNamespace(bot_id="sample", payload={"extraction_enabled": True})
    result = processor._process_memory_extraction(task)
    assert result["skipped"] is True
    assert "memory_system_enabled" in result["reason"]
