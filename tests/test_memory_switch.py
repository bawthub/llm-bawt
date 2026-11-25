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


def test_bot_row_wins_over_global(monkeypatch):
    import llm_bawt.runtime_setting_resolution as resolution

    monkeypatch.setattr(memory_switch, "_override", None)
    monkeypatch.setattr(resolution, "resolve_global_runtime_setting", lambda *a, **k: False)
    monkeypatch.setattr(resolution, "resolve_bot_runtime_setting",
                        lambda config, bot_id, key, fallback=None: bot_id == "kept")
    assert memory_switch.memory_system_enabled(object()) is False
    assert memory_switch.memory_system_enabled(object(), "kept") is True
    assert memory_switch.memory_system_enabled(object(), "other") is False


def test_unreadable_setting_counts_as_off(monkeypatch):
    import llm_bawt.runtime_setting_resolution as resolution

    def boom(*args, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(memory_switch, "_override", None)
    monkeypatch.setattr(resolution, "resolve_global_runtime_setting", boom)
    assert memory_switch.memory_system_enabled(object()) is False


def test_memory_tool_names():
    assert memory_switch.is_memory_tool("memory_search_all")
    assert memory_switch.is_memory_tool("facts_extract")
    assert not memory_switch.is_memory_tool("working_memory_list")
    assert not memory_switch.is_memory_tool("context_recall")


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
