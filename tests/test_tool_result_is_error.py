"""Chat-bot tool failures carry is_error into recorded tool calls (TASK-1782)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_bawt.tools.executor import ToolExecutor
from llm_bawt.tools.loop import ToolLoop
from llm_bawt.tools.parser import ToolCall, format_tool_result, tool_result_is_error
from llm_bawt.service.tool_event_coordinator import ToolEventCoordinator

ERROR = format_tool_result("time", None, error="clock unavailable")
OK = format_tool_result("time", "12:00")


@pytest.mark.parametrize("text,expected", [
    (ERROR, True),
    (OK, False),
    # A success envelope that merely mentions an error is still a success.
    (format_tool_result("search", "Error: none found"), False),
    ("clock unavailable", False),
    ("", False),
    (None, False),
    ({"status": "error"}, False),
])
def test_tool_result_is_error(text, expected):
    assert tool_result_is_error(text) is expected


class _Client:
    def __init__(self, responses):
        self.responses, self.calls = responses, 0

    def query(self, messages, stop=None, **kwargs):
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response

    def supports_native_tools(self):
        return False


class _Msg:
    def __init__(self, role, content):
        self.role, self.content = role, content


def test_tool_loop_details_flag_failed_and_successful_calls():
    loop = ToolLoop(user_id="fixture-user", tool_format="react")
    loop.executor.execute = Mock(side_effect=[ERROR, OK])
    action = 'Action: time\nAction Input: {}'
    loop.run([_Msg("user", "time?")], _Client([action, action, "Final Answer: noon"]))
    details = loop.get_tool_call_details()
    assert [d["is_error"] for d in details] == [True, False]


def test_record_completed_passes_failure_flag_to_end():
    coordinator = ToolEventCoordinator.__new__(ToolEventCoordinator)
    coordinator.start = Mock()
    coordinator.end = Mock()
    coordinator.record_completed(
        [
            {"tool": "time", "parameters": {}, "result": ERROR, "is_error": True},
            {"tool": "time", "parameters": {}, "result": OK, "is_error": False},
            # Details from older producers carry no flag: stay unknown.
            {"tool": "time", "parameters": {}, "result": OK},
        ],
        turn_id="t", trigger_message_id="m", bot_id="b", user_id="u",
    )
    flags = [call.args[0]["is_error"] for call in coordinator.end.call_args_list]
    assert flags == [True, False, None]


def test_end_persists_failure_flag():
    coordinator = ToolEventCoordinator.__new__(ToolEventCoordinator)
    coordinator.store = Mock()
    coordinator.store.save_result.return_value = ("record-1", True)
    coordinator.end({"tool_name": "time", "turn_id": "t", "call_id": "c",
                     "result": ERROR, "is_error": True, "ts": 1.0})
    assert coordinator.store.save_result.call_args.kwargs["is_error"] is True


def test_native_stream_flags_failed_tool_end(monkeypatch):
    from threading import Event

    from llm_bawt.models.message import Message
    from llm_bawt.service.native_tool_stream import NativeToolStream

    executor = SimpleNamespace(execute=Mock(return_value=ERROR))
    monkeypatch.setattr("llm_bawt.tools.executor.ToolExecutor", Mock(return_value=executor))
    monkeypatch.setattr("llm_bawt.tools.formats.get_format_handler",
                        lambda _f: SimpleNamespace(get_tools_schema=lambda _d: []))
    monkeypatch.setattr("llm_bawt.service.native_tool_stream.put_queue_item_threadsafe",
                        lambda *_a: None)
    responses = [[{"tool_calls": [{"id": "c1", "function": {"name": "time", "arguments": "{}"}}]}],
                 ["It failed."]]
    client = SimpleNamespace(stream_with_tools=Mock(side_effect=lambda *_a, **_k: iter(responses.pop(0))))
    llm = SimpleNamespace(
        client=client, bot=SimpleNamespace(uses_tools=True), tool_format="native",
        _get_tool_definitions=Mock(return_value=[]), _ha_mode=False, memory=None,
        profile_manager=None, search_client=None, home_client=None, ha_native_client=None,
        model_lifecycle=None, config=object(), user_id="u", bot_id="b",
    )
    ctx = SimpleNamespace(
        llm_bawt=llm, cancel_event=Event(), loop=object(), chunk_queue=object(),
        turn_log_id="t", trigger_message_id="m", bot_id="b", user_id="u",
        tool_call_details_holder=[], tool_context_holder=[""],
    )
    events = []
    list(NativeToolStream(ctx, publish_event_direct=events.append)
         .stream([Message(role="user", content="time?")], {}))
    end = [e for e in events if e["event"] == "tool_end"]
    assert [e["is_error"] for e in end] == [True]
    assert end[0]["result"] == "clock unavailable"  # Tag still stripped for display.
    assert ctx.tool_call_details_holder[0]["is_error"] is True


def _real_ha_executor(monkeypatch, *, credentials=True, run=None):
    from llm_bawt.integrations.ha_mcp.client import HomeAssistantNativeClient

    client = HomeAssistantNativeClient.__new__(HomeAssistantNativeClient)
    client._initialized, client._tool_names = True, {"HassTurnOn"}
    monkeypatch.setattr(client, "_credentials", lambda: object() if credentials else None)
    monkeypatch.setattr(client, "_call_tool_async", lambda *_a: None)
    if run is not None:
        monkeypatch.setattr(client, "_run_async", run)
    executor = ToolExecutor.__new__(ToolExecutor)
    executor.ha_native_client = client
    return executor


def _ha_call(executor):
    return executor._execute_ha_native_tool(ToolCall(name="HassTurnOn", arguments={}, raw_text=""))


def _content(text, *, is_error):
    return SimpleNamespace(content=[SimpleNamespace(text=text)], is_error=is_error)


def test_real_ha_client_failures_are_error_results(monkeypatch):
    # Missing credentials: the client's own early return, not an escaping exception.
    assert tool_result_is_error(_ha_call(_real_ha_executor(monkeypatch, credentials=False)))

    def transport_failure(_coro):
        raise RuntimeError("connection refused")
    result = _ha_call(_real_ha_executor(monkeypatch, run=transport_failure))
    assert tool_result_is_error(result) and "connection refused" in result

    # HA answered, but its MCP result says the tool failed.
    result = _ha_call(_real_ha_executor(
        monkeypatch, run=lambda _c: _content("No device named kitchen", is_error=True)))
    assert tool_result_is_error(result) and "No device named kitchen" in result


def test_real_ha_client_success_stays_success(monkeypatch):
    result = _ha_call(_real_ha_executor(
        monkeypatch, run=lambda _c: _content("Turned on kitchen", is_error=False)))
    assert not tool_result_is_error(result) and "Turned on kitchen" in result


def test_ha_native_failure_is_an_error_result():
    executor = ToolExecutor.__new__(ToolExecutor)
    executor.ha_native_client = SimpleNamespace(call_tool=Mock(side_effect=RuntimeError("HA down")))
    result = executor._execute_ha_native_tool(ToolCall(name="HassTurnOn", arguments={}, raw_text=""))
    assert tool_result_is_error(result)
    assert "HA down" in result
    executor.ha_native_client = None
    assert tool_result_is_error(
        executor._execute_ha_native_tool(ToolCall(name="HassTurnOn", arguments={}, raw_text=""))
    )
