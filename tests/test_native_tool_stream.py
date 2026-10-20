"""Behavioral contract for the native-tool loop extracted from the worker."""
from __future__ import annotations

from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_bawt.models.message import Message
from llm_bawt.service.native_tool_stream import NativeToolStream


@pytest.fixture
def rig(monkeypatch):
    schema = [{"type": "function", "function": {"name": "GetLiveContext"}}]
    events, queued = [], []
    executor = SimpleNamespace(execute=Mock(return_value="<tool_result>72 degrees</tool_result>"))
    factory = Mock(return_value=executor)
    monkeypatch.setattr("llm_bawt.tools.executor.ToolExecutor", factory)
    monkeypatch.setattr("llm_bawt.tools.formats.get_format_handler", lambda _format:
                        SimpleNamespace(get_tools_schema=lambda _definitions: schema))
    monkeypatch.setattr("llm_bawt.service.native_tool_stream.put_queue_item_threadsafe",
                        lambda _loop, _queue, item: queued.append(item))
    client = SimpleNamespace(stream_with_tools=Mock())
    llm = SimpleNamespace(
        client=client, bot=SimpleNamespace(uses_tools=True), tool_format="native",
        _get_tool_definitions=Mock(return_value=[]), _ha_mode=False,
        memory=None, profile_manager=None, search_client=None, home_client=None,
        ha_native_client=None, model_lifecycle=None, config=object(),
        user_id="fixture-owner", bot_id="fixture-bot",
    )
    ctx = SimpleNamespace(
        llm_bawt=llm, cancel_event=Event(), loop=object(), chunk_queue=object(),
        turn_log_id="fixture-turn", trigger_message_id="fixture-message",
        bot_id=llm.bot_id, user_id=llm.user_id,
        tool_call_details_holder=[], tool_context_holder=[""],
    )
    return SimpleNamespace(
        ctx=ctx, llm=llm, client=client, schema=schema, events=events, queued=queued,
        executor=executor, factory=factory,
        stream=NativeToolStream(ctx, publish_event_direct=events.append),
        messages=[Message(role="user", content="Temperature?")],
    )


def tool_result(arguments='{"room":"office"}'):
    return {"tool_calls": [{"id": "call-fixture", "function": {
        "name": "GetLiveContext", "arguments": arguments,
    }}]}


@pytest.mark.parametrize("arguments,expected", [('{"room":"office"}', {"room": "office"}),
                                               ("broken-json", {})])
def test_tool_execution_and_synthesis_preserve_events_and_context(rig, arguments, expected):
    requests = []
    def stream(messages, **kwargs):
        requests.append((list(messages), kwargs))
        return iter(["Checking", tool_result(arguments)] if len(requests) == 1 else ["72 degrees"])
    rig.client.stream_with_tools.side_effect = stream
    iterator = rig.stream.stream(rig.messages, {"temperature": 0.4})
    rig.factory.assert_called_once()  # Executor/schema setup remains eager.
    rig.client.stream_with_tools.assert_not_called()  # Provider execution remains lazy.
    assert list(iterator) == ["Checking", "\n\n", "72 degrees"]
    assert [request[1]["tool_choice"] for request in requests] == ["auto", "none"]
    assert all(request[1]["tools_schema"] == rig.schema for request in requests)
    assert all(request[1]["temperature"] == 0.4 for request in requests)
    call = rig.executor.execute.call_args.args[0]
    assert call.name == "GetLiveContext" and call.arguments == expected
    assert [item["event"] for item in rig.events] == ["tool_start", "tool_end"]
    assert [item["_type"] for item in rig.queued] == ["tool_call_delta", "tool_calls_finish"]
    assert all(item["turn_id"] == rig.ctx.turn_log_id for item in rig.events)
    assert all(item["trigger_message_id"] == rig.ctx.trigger_message_id for item in rig.events)
    assert all(item["bot_id"] == rig.ctx.bot_id and item["user_id"] == rig.ctx.user_id
               for item in rig.events)
    assert rig.events[-1]["result"] == "72 degrees"
    assert rig.ctx.tool_context_holder == ["[GetLiveContext]\n72 degrees"]
    assert rig.ctx.tool_call_details_holder[0]["result"] == "72 degrees"
    continuation = requests[1][0]
    assert [message.role for message in continuation] == ["user", "assistant", "tool", "system"]
    assert continuation[1].content == ""
    assert continuation[2].tool_call_id == "call-fixture"
    assert continuation[2].content == "72 degrees"
    assert len(rig.messages) == 1  # No mutation of the caller's original messages.


@pytest.mark.parametrize("cancel_after_text", [False, True])
def test_cancellation_never_executes_pending_tool(rig, cancel_after_text):
    rig.client.stream_with_tools.return_value = iter(["Checking", tool_result()])
    iterator = rig.stream.stream(rig.messages, {})
    if cancel_after_text:
        assert next(iterator) == "Checking"
    rig.ctx.cancel_event.set()
    assert list(iterator) == []
    rig.executor.execute.assert_not_called()
    assert rig.events == [] and rig.queued == []
    if not cancel_after_text:
        rig.client.stream_with_tools.assert_not_called()


def test_empty_tool_calls_ends_after_already_streamed_text(rig):
    rig.client.stream_with_tools.return_value = iter(["No tools needed", {"tool_calls": []}])
    assert list(rig.stream.stream(rig.messages, {})) == ["No tools needed"]
    rig.client.stream_with_tools.assert_called_once()
    rig.executor.execute.assert_not_called()
    assert rig.events == []


def test_tool_failure_propagates_without_false_end_event(rig):
    rig.client.stream_with_tools.return_value = iter([tool_result()])
    rig.executor.execute.side_effect = RuntimeError("tool unavailable")
    with pytest.raises(RuntimeError, match="tool unavailable"):
        list(rig.stream.stream(rig.messages, {}))
    assert [item["event"] for item in rig.events] == ["tool_start"]
    assert rig.ctx.tool_call_details_holder == []


@pytest.mark.parametrize("ha_mode,limit", [(False, 5), (True, 3)])
def test_iteration_limit_is_preserved(rig, ha_mode, limit):
    rig.llm._ha_mode = ha_mode
    rig.client.stream_with_tools.side_effect = lambda *_a, **_kw: iter([tool_result()])
    assert list(rig.stream.stream(rig.messages, {})) == ["\n\n"] * limit
    assert rig.client.stream_with_tools.call_count == limit
    assert rig.executor.execute.call_count == limit
    assert len(rig.ctx.tool_call_details_holder) == limit
