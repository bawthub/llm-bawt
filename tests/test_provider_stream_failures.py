"""Offline terminal-error regressions for TASK-2226's client paths."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import httpx
import pytest
from openai import APIConnectionError

from llm_bawt.clients.openai_client import OpenAIClient
from llm_bawt.clients.responses_client import ResponsesClient


def _openai(monkeypatch, *, partial=False, wrapped=False):
    client = OpenAIClient.__new__(OpenAIClient)
    client.model = "fixture"
    client.config = NS(TEMPERATURE=0, TOP_P=1)
    monkeypatch.setattr(OpenAIClient, "effective_max_tokens", property(lambda self: 100))
    client._prepare_api_messages = Mock(return_value=[])
    client._model_supports_temperature_top_p = Mock(return_value=False)
    error = httpx.RemoteProtocolError("fixture disconnect")
    if wrapped:
        outer = APIConnectionError(request=httpx.Request("POST", "https://fixture.invalid"))
        outer.__cause__ = error
        error = outer

    def stream():
        if partial:
            yield NS(choices=[NS(delta=NS(content="Partial answer", tool_calls=None))])
        raise error

    client._chat_create_with_fallback = Mock(side_effect=lambda payload: stream())
    client.query_with_tools = Mock(return_value=({"content": "Recovered", "tool_calls": []}, None))
    return client, error


@pytest.mark.parametrize("wrapped", [False, True])
def test_tool_stream_disconnect_after_text_is_not_success(monkeypatch, wrapped):
    client, error = _openai(monkeypatch, partial=True, wrapped=wrapped)
    stream = client.stream_with_tools([])
    assert next(stream) == "Partial answer"
    with pytest.raises(type(error)) as raised:
        next(stream)
    assert raised.value is error
    client.query_with_tools.assert_not_called()


def test_tool_stream_failed_fallback_propagates(monkeypatch):
    client, _ = _openai(monkeypatch)
    failure = RuntimeError("fixture fallback failed")
    client.query_with_tools.side_effect = failure
    with pytest.raises(RuntimeError) as raised:
        list(client.stream_with_tools([]))
    assert raised.value is failure
    client.query_with_tools.assert_called_once()


def test_tool_stream_successful_fallback_still_recovers(monkeypatch):
    client, _ = _openai(monkeypatch)
    assert list(client.stream_with_tools([])) == ["Recovered"]
    client.query_with_tools.assert_called_once()


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_raw_stream_disconnect_propagates(monkeypatch, partial, wrapped):
    client, error = _openai(monkeypatch, partial=partial, wrapped=wrapped)
    stream = client.stream_raw([])
    if partial:
        assert next(stream) == "Partial answer"
    with pytest.raises(type(error)) as raised:
        next(stream)
    assert raised.value is error
    client.query_with_tools.assert_not_called()


def _responses(events):
    client = ResponsesClient.__new__(ResponsesClient)
    client.model = "fixture"
    client._build_base_payload = Mock(return_value={})
    client.client = NS(responses=NS(create=Mock(side_effect=lambda **kw: iter(events))))
    return client


def _tool_events():
    return [
        NS(type="response.output_item.added", item=NS(
            type="function_call", id="item-1", call_id="call-1", name="do_thing")),
        NS(type="response.function_call_arguments.delta", item_id="item-1", delta="{}"),
    ]


@pytest.mark.parametrize("terminal", ["response.failed", "response.incomplete", "error"])
@pytest.mark.parametrize("with_tools", [False, True])
def test_responses_failed_terminal_does_not_emit_calls(terminal, with_tools):
    events = [NS(type="response.output_text.delta", delta="Partial answer")]
    if with_tools:
        events += _tool_events()
    events.append(NS(type=terminal, response=NS(error=NS(message="sensitive upstream payload"))))
    stream = _responses(events).stream_with_tools([])
    assert next(stream) == "Partial answer"
    with pytest.raises(RuntimeError, match=terminal) as raised:
        next(stream)
    assert "sensitive upstream payload" not in str(raised.value)


@pytest.mark.parametrize("terminal", ["response.failed", "response.incomplete"])
def test_responses_raw_empty_failure_propagates(terminal):
    with pytest.raises(RuntimeError, match=terminal):
        list(_responses([NS(type=terminal)]).stream_raw([]))


def test_responses_completed_tool_call_control():
    events = _tool_events() + [NS(type="response.completed")]
    assert list(_responses(events).stream_with_tools([])) == [{
        "tool_calls": [{"id": "call-1", "type": "function",
                        "function": {"name": "do_thing", "arguments": "{}"}}],
        "content": "",
    }]


def test_responses_completed_text_control():
    events = [NS(type="response.output_text.delta", delta="Complete answer"),
              NS(type="response.completed")]
    assert list(_responses(events).stream_raw([])) == ["Complete answer"]


def test_worker_records_partial_text_and_disconnect(monkeypatch):
    """Exercise the real consumer and error handler without booting a worker."""
    import ast
    import inspect
    import logging
    import time
    from threading import Event

    from llm_bawt.service import turn_stream_worker
    from llm_bawt.service.chat_stream_worker import consume_stream_chunks

    client, failure = _openai(monkeypatch, partial=True)
    source = ast.parse(inspect.getsource(turn_stream_worker))
    handler = next(node for node in ast.walk(source)
                   if isinstance(node, ast.ExceptHandler)
                   and any(isinstance(call, ast.Attribute) and call.attr == "_update_turn_log"
                           for call in ast.walk(node)))
    body = ast.parse("consume()").body
    harness = ast.parse("def run():\n    pass\n")
    harness.body[0].body = [ast.Try(body=body, handlers=[handler], orelse=[], finalbody=[])]
    ast.fix_missing_locations(harness)
    holder = [""]
    queued = []
    queue = NS(put_nowait=queued.append)
    loop = NS(call_soon_threadsafe=lambda callback, item: callback(item))
    cancel = Event()
    service = NS(_update_turn_log=Mock())

    def consume():
        consume_stream_chunks(client.stream_with_tools([]), cancel_event=cancel,
                              loop=loop, chunk_queue=queue, full_response_holder=holder)

    scope = dict(consume=consume, ctx=NS(execution=None), self=service,
                 _turn_was_aborted=lambda: False, cancelled_holder=[False],
                 cancel_event=cancel, loop=loop, chunk_queue=queue,
                 put_queue_item_threadsafe=turn_stream_worker.put_queue_item_threadsafe,
                 time=time, timing_holder=[None], is_agent_backend=False,
                 client_error_text=turn_stream_worker.client_error_text,
                 turn_log_id="fixture-turn", full_response_holder=holder,
                 _usage_so_far=lambda: None, log=logging.getLogger(__name__))
    exec(compile(harness, turn_stream_worker.__file__, "exec"), scope)
    scope["run"]()
    assert queued == ["Partial answer", failure]
    saved = service._update_turn_log.call_args.kwargs
    assert saved["status"] == "error"
    assert saved["response_text"] == "Partial answer"
    assert saved["error_text"]
