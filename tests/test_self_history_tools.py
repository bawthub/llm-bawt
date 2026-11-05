"""Raw self-history tools remain available without inference or persistence."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from llm_bawt.mcp_server import self_tools
from llm_bawt.mcp_server import server


def test_self_tool_catalog_matches_supported_tools():
    tools = asyncio.run(server.mcp.list_tools())
    assert {tool.name for tool in tools if tool.name.startswith("self_")} == {
        "self_tail", "self_fwd", "self_system_prompt", "self_model",
    }


def test_tail_reads_raw_real_bubbles_without_writes(monkeypatch):
    storage = SimpleNamespace(get_messages=AsyncMock(return_value=[
        {"role": "summary", "content": "not a bubble"},
        {"role": "user", "content": "first", "timestamp": 1.0},
        {"role": "tool", "content": "not a bubble"},
        {"role": "assistant", "content": "second", "timestamp": 2.0},
        {"role": "user", "content": "third", "timestamp": 3.0},
    ]))
    monkeypatch.setattr(server, "_get_storage", lambda: storage)
    result = asyncio.run(self_tools.self_tail(bot_id="fixture", count=2))
    storage.get_messages.assert_awaited_once_with(bot_id="fixture", raw=True)
    assert result["total_available"] == 3
    assert result["count_returned"] == 2
    assert [m["content"] for m in result["messages"]] == ["second", "third"]
    assert "first" not in result["transcript"]
    assert "[ASSISTANT @ 1970-01-01 00:00 UTC]\nsecond" in result["transcript"]
    assert result["truncated"] is False


def test_tail_minimum_count_and_empty_history(monkeypatch):
    storage = SimpleNamespace(get_messages=AsyncMock(return_value=[]))
    monkeypatch.setattr(server, "_get_storage", lambda: storage)
    result = asyncio.run(self_tools.self_tail(bot_id="fixture", count=0))
    assert result["messages"] == []
    assert result["transcript"] == ""
    assert result["count_returned"] == result["total_available"] == 0
    assert result["truncated"] is False


def test_tail_formatter_keeps_existing_bound_and_reports_oldest_drop(monkeypatch):
    monkeypatch.setattr(self_tools, "_MAX_TAIL_CHARS", 30)
    transcript, count, truncated = self_tools._format_bubbles([
        {"role": "user", "content": "old" * 20},
        {"role": "assistant", "content": "new"},
    ])
    assert transcript == "[ASSISTANT]\nnew"
    assert count == 1
    assert truncated is True
