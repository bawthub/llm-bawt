"""Behavioral contract for the native-tool loop extracted from the worker."""
from __future__ import annotations

from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_bawt.models.message import Message
from llm_bawt.service.native_tool_stream import NativeToolStream
from llm_bawt.service.pending_tool_calls import PendingToolCallCorrelator
from llm_bawt.service.tool_event_interceptor import ToolEventInterceptor


@pytest.fixture
def rig(monkeypatch):
    schema = [{"type": "function", "function": {"name": "GetLiveContext"}}]
    events, queued = [], []
    executor = SimpleNamespace(execute=Mock(return_value="<tool_result>72 degrees</tool_result>"),
                               mcp_loaded={}, mcp_tool_definitions=Mock(return_value=[]))
    factory = Mock(return_value=executor)
    monkeypatch.setattr("llm_bawt.tools.executor.ToolExecutor", factory)
    monkeypatch.setattr("llm_bawt.tools.formats.get_format_handler", lambda _format:
                        SimpleNamespace(get_tools_schema=lambda _definitions: schema))
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
        is_agent_backend=False, tts_scrubber=None, stream_text_offset=None,
        _upstream_model=[None], token_usage_holder=[None], agent_attachments_holder=[],
        reasoning_holder=[""], model_alias="fixture-model", _redis_sub=None,
    )
    out = SimpleNamespace(
        ctx=ctx, llm=llm, client=client, schema=schema, events=events, queued=queued,
        executor=executor, factory=factory,
        publish=lambda e: events.append(e) if e.get("_type") == "tool_event" else None,
        messages=[Message(role="user", content="Temperature?")],
    )
    out.stream = _Wired(out)
    return out


class _Wired:
    """The production wiring: NativeToolStream items pass through the shared
    ToolEventInterceptor, which publishes and records tool events."""

    def __init__(self, rig):
        self.rig = rig

    def stream(self, messages, gen_kwargs):
        rig = self.rig
        inner = NativeToolStream(rig.ctx).stream(messages, gen_kwargs)
        interceptor = ToolEventInterceptor(
            rig.ctx, svc=SimpleNamespace(_update_turn_log=Mock(), config=None,
                                         _turn_log_store=SimpleNamespace(
                                             reap_other_open_turns=Mock(return_value=[]))),
            publish_event_direct=lambda event: rig.publish(event),
            persist_publish_approval=Mock(), persist_publish_await=Mock(),
            changed_files=SimpleNamespace(persist=Mock(return_value=None)),
            turn_usage=SimpleNamespace(capture=Mock(), http_chunk=Mock()),
            pin_confirmer=SimpleNamespace(observe=Mock()),
            scheduled_notices=SimpleNamespace(observe=Mock()),
            tool_calls=PendingToolCallCorrelator(),
            exec_stamps=SimpleNamespace(on_executing=Mock(), on_tool_started=Mock()),
            confirmed_start_reaped=[True],
        )
        def texts():
            for item in interceptor.run(inner):
                if isinstance(item, str):
                    yield item
                else:
                    rig.queued.append(item)
        return texts()


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
    assert [request[1]["tool_choice"] for request in requests] == ["auto", "auto"]
    assert all(request[1]["tools_schema"] == rig.schema for request in requests)
    assert all(request[1]["temperature"] == 0.4 for request in requests)
    call = rig.executor.execute.call_args.args[0]
    assert call.name == "GetLiveContext" and call.arguments == expected
    assert [item["event"] for item in rig.events] == ["tool_start", "tool_end"]
    assert [item["event"] for item in rig.queued] == ["tool_call", "tool_result"]
    # The interceptor stamps the authoritative offset after the preceding text.
    assert rig.events[0]["text_offset"] == len("Checking")
    assert rig.events[0]["tool_use_id"] == rig.events[1]["tool_use_id"] == "call-fixture"
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
    assert "Call additional tools if needed" in continuation[3].content
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
    assert [d.get("result") for d in rig.ctx.tool_call_details_holder] == [None]


@pytest.mark.parametrize("ha_mode,limit", [(False, 5), (True, 3)])
def test_round_limit_runs_one_final_text_synthesis(rig, ha_mode, limit):
    rig.llm._ha_mode = ha_mode
    requests = []
    def stream(messages, **kwargs):
        requests.append((list(messages), kwargs))
        return iter(["Incomplete"] if kwargs["tool_choice"] == "none" else [tool_result()])
    rig.client.stream_with_tools.side_effect = stream
    assert list(rig.stream.stream(rig.messages, {})) == ["Incomplete"]
    assert rig.client.stream_with_tools.call_count == limit + 1
    assert rig.executor.execute.call_count == limit
    assert len(rig.ctx.tool_call_details_holder) == limit
    assert [kwargs["tool_choice"] for _, kwargs in requests] == ["auto"] * limit + ["none"]
    assert all(kwargs["tools_schema"] == rig.schema for _, kwargs in requests)
    assert "tool-round limit" in requests[-1][0][-1].content


def test_dependent_second_round_receives_results_then_answers(rig):
    rig.executor.execute.side_effect = [
        "<tool_result>72 degrees</tool_result>",
        "<tool_result>Search used 72 degrees</tool_result>",
    ]
    requests = []
    def stream(messages, **kwargs):
        requests.append((list(messages), kwargs))
        if len(requests) == 1:
            return iter([tool_result()])
        if len(requests) == 2:
            assert messages[-2].role == "tool"
            assert messages[-2].content == "72 degrees"
            return iter([tool_result('{"room":"72 degrees"}')])
        assert [message.content for message in messages if message.role == "tool"] == [
            "72 degrees", "Search used 72 degrees",
        ]
        return iter(["Completed"])
    rig.client.stream_with_tools.side_effect = stream
    assert list(rig.stream.stream(rig.messages, {})) == ["Completed"]
    assert rig.executor.execute.call_count == 2
    assert [call.args[0].arguments for call in rig.executor.execute.call_args_list] == [
        {"room": "office"}, {"room": "72 degrees"},
    ]
    assert [kwargs["tool_choice"] for _, kwargs in requests] == ["auto"] * 3
    assert [event["event"] for event in rig.events] == ["tool_start", "tool_end"] * 2
    assert rig.ctx.tool_context_holder == [
        "[GetLiveContext]\n72 degrees\n\n[GetLiveContext]\nSearch used 72 degrees",
    ]
    assert [call["result"] for call in rig.ctx.tool_call_details_holder] == [
        "72 degrees", "Search used 72 degrees",
    ]


def test_elapsed_budget_prevents_next_round_and_synthesizes(rig, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("llm_bawt.service.native_tool_stream.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("llm_bawt.service.native_tool_stream.TOOL_LOOP_BUDGET_SECONDS", 1.0)
    def execute(_call):
        clock[0] = 2.0
        return "<tool_result>72 degrees</tool_result>"
    rig.executor.execute.side_effect = execute
    requests = []
    def stream(messages, **kwargs):
        requests.append((list(messages), kwargs))
        return iter(["Partial answer"] if kwargs["tool_choice"] == "none" else [tool_result()])
    rig.client.stream_with_tools.side_effect = stream
    assert list(rig.stream.stream(rig.messages, {})) == ["Partial answer"]
    rig.executor.execute.assert_called_once()
    assert [kwargs["tool_choice"] for _, kwargs in requests] == ["auto", "none"]
    assert "tool time budget" in requests[-1][0][-1].content


def test_budget_expiring_during_model_wait_never_executes_tool(rig, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("llm_bawt.service.native_tool_stream.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("llm_bawt.service.native_tool_stream.TOOL_LOOP_BUDGET_SECONDS", 1.0)
    def stream(_messages, **kwargs):
        if kwargs["tool_choice"] == "none":
            return iter(["Could not finish"])
        clock[0] = 2.0
        return iter([tool_result()])
    rig.client.stream_with_tools.side_effect = stream
    assert list(rig.stream.stream(rig.messages, {})) == ["Could not finish"]
    rig.executor.execute.assert_not_called()
    assert rig.events[-1]["is_error"] is True
    assert "Not executed" in rig.events[-1]["result"]


def test_budget_is_checked_between_calls_in_same_round(rig, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("llm_bawt.service.native_tool_stream.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("llm_bawt.service.native_tool_stream.TOOL_LOOP_BUDGET_SECONDS", 1.0)
    def execute(_call):
        clock[0] = 2.0
        return "<tool_result>72 degrees</tool_result>"
    rig.executor.execute.side_effect = execute
    calls = tool_result()
    calls["tool_calls"].extend(tool_result()["tool_calls"])
    rig.client.stream_with_tools.side_effect = lambda _messages, **kw: iter(
        ["Partial answer"] if kw["tool_choice"] == "none" else [calls])
    assert list(rig.stream.stream(rig.messages, {})) == ["Partial answer"]
    rig.executor.execute.assert_called_once()
    assert [event["is_error"] for event in rig.events if event["event"] == "tool_end"] == [
        False, True,
    ]


def test_final_pass_cannot_execute_provider_ignored_tool_choice(rig, monkeypatch):
    monkeypatch.setattr("llm_bawt.service.native_tool_stream.TOOL_LOOP_BUDGET_SECONDS", 0.0)
    rig.client.stream_with_tools.return_value = iter([tool_result()])
    assert list(rig.stream.stream(rig.messages, {})) == [
        "I reached the tool limit before completing the request.",
    ]
    rig.executor.execute.assert_not_called()
    rig.client.stream_with_tools.assert_called_once()
    assert rig.client.stream_with_tools.call_args.kwargs["tool_choice"] == "none"


def test_xai_responses_adapter_keeps_schema_and_both_round_results(rig):
    from llm_bawt.clients.responses_client import ResponsesClient

    client = ResponsesClient.__new__(ResponsesClient)
    client.model = "grok-fixture"
    client.config = SimpleNamespace(SYSTEM_MESSAGE="")
    requests = []
    def payload(messages, **_kwargs):
        items, instructions = client._prepare_input(messages, log_input=False)
        return {"input": items, "instructions": instructions}
    client._build_base_payload = payload
    def create(**kwargs):
        requests.append(kwargs)
        ordinal = len(requests)
        if ordinal <= 2:
            return iter([
                SimpleNamespace(type="response.output_item.added", item=SimpleNamespace(
                    type="function_call", id=f"item-{ordinal}",
                    call_id=f"call-{ordinal}", name="GetLiveContext")),
                SimpleNamespace(type="response.function_call_arguments.delta",
                                item_id=f"item-{ordinal}", delta='{"room":"office"}'),
                SimpleNamespace(type="response.completed"),
            ])
        return iter([
            SimpleNamespace(type="response.output_text.delta", delta="Completed"),
            SimpleNamespace(type="response.completed"),
        ])
    client.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    rig.llm.client = client
    assert list(rig.stream.stream(rig.messages, {})) == ["Completed"]
    assert [request["tool_choice"] for request in requests] == ["auto"] * 3
    assert all(request["tools"][0]["name"] == "GetLiveContext" for request in requests)
    outputs = [item for item in requests[-1]["input"]
               if item.get("type") == "function_call_output"]
    assert [(item["call_id"], item["output"]) for item in outputs] == [
        ("call-1", "72 degrees"), ("call-2", "72 degrees"),
    ]


@pytest.mark.parametrize("model", ["gpt-5-fixture", "deepseek-chat-fixture"])
def test_chat_completions_adapters_preserve_dependent_rounds(rig, monkeypatch, model):
    from llm_bawt.clients.openai_client import OpenAIClient

    client = OpenAIClient.__new__(OpenAIClient)
    client.model = model
    client.config = SimpleNamespace(SYSTEM_MESSAGE="", TEMPERATURE=0, TOP_P=1)
    monkeypatch.setattr(OpenAIClient, "effective_max_tokens", property(lambda _self: 100))
    client._model_supports_temperature_top_p = lambda: False
    client._usage_chunks = lambda stream: stream
    requests = []
    def create(payload):
        requests.append(payload)
        ordinal = len(requests)
        if ordinal <= 2:
            calls = [SimpleNamespace(
                index=0, id=f"call-{ordinal}",
                function=SimpleNamespace(name="GetLiveContext", arguments='{"room":"office"}'),
            )]
            delta = SimpleNamespace(content=None, tool_calls=calls)
        else:
            delta = SimpleNamespace(content="Completed", tool_calls=None)
        return iter([SimpleNamespace(choices=[SimpleNamespace(delta=delta)])])
    client._chat_create_with_fallback = create
    rig.llm.client = client
    assert list(rig.stream.stream(rig.messages, {})) == ["Completed"]
    assert [request["tool_choice"] for request in requests] == ["auto"] * 3
    assert all(request["tools"] == rig.schema for request in requests)
    outputs = [item for item in requests[-1]["messages"] if item["role"] == "tool"]
    assert [(item["tool_call_id"], item["content"]) for item in outputs] == [
        ("call-1", "72 degrees"), ("call-2", "72 degrees"),
    ]


def test_stop_during_tool_execution_keeps_its_result_then_stops(rig):
    rig.client.stream_with_tools.return_value = iter(["Checking", tool_result()])
    def execute(_call):
        rig.ctx.cancel_event.set()  # Stop pressed while the tool runs.
        return "<tool_result>light on</tool_result>"
    rig.executor.execute.side_effect = execute
    assert list(rig.stream.stream(rig.messages, {})) == ["Checking"]
    assert [event["event"] for event in rig.events] == ["tool_start", "tool_end"]
    assert rig.events[-1]["result"] == "light on"
    assert rig.ctx.tool_call_details_holder[0]["result"] == "light on"
    rig.client.stream_with_tools.assert_called_once()  # No follow-up round.


def test_schemas_loaded_by_tool_search_join_later_rounds(rig):
    def execute(_call):
        rig.executor.mcp_loaded["memory_search"] = object()  # tool_search loaded one
        return "<tool_result>loaded</tool_result>"
    rig.executor.execute.side_effect = execute
    requests = []
    def stream(messages, **kwargs):
        requests.append(kwargs["tools_schema"])
        return iter([tool_result()] if len(requests) == 1 else ["Done"])
    rig.client.stream_with_tools.side_effect = stream
    assert list(rig.stream.stream(rig.messages, {})) == ["Done"]
    assert requests[0] == rig.schema
    assert requests[1] == rig.schema + rig.schema  # base + loaded definitions
    rig.executor.mcp_tool_definitions.assert_called()


@pytest.mark.parametrize("call_id, expected", [("call-fixture", "call-fixture"), (None, "")])
def test_only_the_provider_call_id_reaches_the_executor(rig, call_id, expected):
    call = {"function": {"name": "GetLiveContext", "arguments": "{}"}}
    if call_id:
        call["id"] = call_id
    rounds = iter([["", {"tool_calls": [call]}], ["done"]])
    rig.client.stream_with_tools.side_effect = lambda messages, **kwargs: iter(next(rounds))
    list(rig.stream.stream(rig.messages, {}))
    assert rig.executor.execute.call_args.args[0].tool_use_id == expected
