"""Text/ReAct tool streaming emits the same tool-event items as native tools."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from llm_bawt.models.message import Message
from llm_bawt.tools import streaming


def _run(monkeypatch, result, **kwargs):
    executor = SimpleNamespace(execute=Mock(return_value=result))
    monkeypatch.setattr(streaming, "ToolExecutor", Mock(return_value=executor))
    replies = iter([['Action: time\nAction Input: {"zone": "UTC"}'], ["It is noon."]])
    details: list[dict] = []
    items = list(streaming.stream_with_tools(
        messages=[Message(role="user", content="Time?")],
        stream_fn=lambda _msgs, stop_sequences=None: iter(next(replies)),
        tool_format="react", tool_call_details=details, user_id="fixture-owner",
        trigger_message_id="fixture-message", **kwargs,
    ))
    return items, details, executor


def test_react_yields_tool_events_for_the_interceptor(monkeypatch):
    items, details, executor = _run(
        monkeypatch, "<tool_result>12:00</tool_result>", emit_tool_events=True)
    executor.execute.assert_called_once()
    call, result = [item for item in items if isinstance(item, dict)]
    assert call == {"event": "tool_call", "name": "time", "arguments": {"zone": "UTC"},
                    "tool_use_id": "react_1", "trigger_message_id": "fixture-message"}
    assert result["event"] == "tool_result" and result["tool_use_id"] == "react_1"
    assert result["result"] == "12:00" and result["is_error"] is False
    assert details == []  # The interceptor records details from the events.
    assert "It is noon." in "".join(i for i in items if isinstance(i, str))


def test_react_without_events_keeps_plain_text_and_details(monkeypatch):
    items, details, _ = _run(monkeypatch, "<tool_result>12:00</tool_result>")
    assert all(isinstance(item, str) for item in items)
    assert [d["tool"] for d in details] == ["time"]
