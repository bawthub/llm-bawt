"""Real direct-client response paths capture measured usage without live calls."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from openai import BadRequestError
import httpx

from llm_bawt.clients.base import LLMClient
from llm_bawt.clients.direct_usage import DirectTurnUsage
from llm_bawt.clients.grok_client import GrokClient
from llm_bawt.clients.openai_client import OpenAIClient
from llm_bawt.clients.responses_client import ResponsesClient
from llm_bawt.models.message import Message
from llm_bawt.service.turn_lifecycle import TurnLifecycleMixin

DEFINITION = {
    "access_path": "xai-chat", "access_vendor": "xai", "protocol": "responses",
    "pricing": {"input": 2, "output": 6, "cache_read": 0.5, "cache_write": 2},
}
MESSAGES = [Message(role="user", content="hello")]


def make_client(cls, create):
    client = cls.__new__(cls)
    config = NS(
        PLAIN_OUTPUT=True, NO_STREAM=False, TEMPERATURE=0.5, TOP_P=0.9,
        VERBOSE=False, SYSTEM_MESSAGE="",
        get_model_context_window=lambda _: 128_000,
        get_model_max_tokens=lambda _: 4096,
    )
    LLMClient.__init__(client, "example-model", config, dict(DEFINITION))
    client.client = NS(responses=NS(create=create), chat=NS(completions=NS(create=create)))
    client.begin_turn_usage()
    return client


def response(*, chat=False, usage=None):
    usage = usage or (
        {"prompt_tokens": 1000, "completion_tokens": 100, "prompt_tokens_details": {"cached_tokens": 200}}
        if chat else {"input_tokens": 1000, "output_tokens": 100, "input_tokens_details": {"cached_tokens": 200}}
    )
    return NS(
        usage=NS(model_dump=lambda: usage),
        output=[NS(type="message", content=[NS(type="output_text", text="ok")])],
        choices=[NS(message=NS(content="ok", tool_calls=[]))],
    )


@pytest.mark.parametrize("cls", [OpenAIClient, ResponsesClient, GrokClient])
@pytest.mark.parametrize("tools", [False, True])
def test_nonstream_text_and_tools_capture_usage(cls, tools):
    create = Mock(return_value=response(chat=cls is OpenAIClient))
    client = make_client(cls, create)
    if tools:
        client.query_with_tools(MESSAGES)
    else:
        assert client.query(MESSAGES, plaintext_output=True, stream=False) == "ok"
    usage = client.get_token_usage()
    assert usage["input_tokens"] == 800
    assert usage["cache_read_tokens"] == 200
    assert usage["resident_tokens"] == 1000
    assert usage["output_tokens"] == 100
    assert usage["total_cost_usd"] == pytest.approx(0.0023)
    assert usage["cost_kind"] == "billed"
    assert usage["cost_source"] == "catalog_pricing"
    assert usage["context_window"] == 128_000


@pytest.mark.parametrize("tools", [False, True])
def test_chat_stream_usage_only_frame_is_counted_once(tools):
    text = NS(choices=[NS(delta=NS(content="ok", tool_calls=[]))], usage=None)
    final = response(chat=True)
    final.choices = []
    create = Mock(return_value=iter([text, final]))
    client = make_client(OpenAIClient, create)
    chunks = list(client.stream_with_tools(MESSAGES) if tools else client.stream_raw(MESSAGES))
    assert chunks == ["ok"]
    assert create.call_args.kwargs["stream_options"] == {"include_usage": True}
    assert client.get_token_usage()["turn_request_totals"]["requests"] == 1
    assert client.get_token_usage()["total_cost_usd"] == pytest.approx(0.0023)


def test_actual_nonstream_turn_entry_resets_once_not_each_tool_call():
    from llm_bawt.service.core import ServiceLLMBawt

    client = make_client(ResponsesClient, Mock(return_value=response()))
    client.query_with_tools(MESSAGES)
    client.query_with_tools(MESSAGES)
    assert client.get_token_usage()["turn_request_totals"]["requests"] == 2
    harness = NS(client=client, tool_format="none", bot=NS(uses_tools=False))
    assert ServiceLLMBawt.execute_llm_query(harness, MESSAGES, plaintext_output=True) == ("ok", "", [])
    assert client.get_token_usage()["turn_request_totals"]["requests"] == 1
    assert client.get_token_usage()["total_cost_usd"] == pytest.approx(0.0023)


def test_agent_turn_entry_never_starts_a_direct_accumulator():
    from llm_bawt.service.core import ServiceLLMBawt

    client = NS(
        model_definition={"type": "agent_backend"},
        begin_turn_usage=Mock(side_effect=AssertionError("agent reset")),
        query=Mock(return_value="agent reply"),
    )
    harness = NS(client=client, tool_format="none", bot=NS(uses_tools=False))
    assert ServiceLLMBawt.execute_llm_query(harness, MESSAGES) == ("agent reply", "", [])
    client.begin_turn_usage.assert_not_called()


@pytest.mark.parametrize("cls", [ResponsesClient, GrokClient])
@pytest.mark.parametrize("tools", [False, True])
def test_responses_stream_uses_completed_usage(cls, tools):
    final = response(usage={"input_tokens": 1000, "output_tokens": 100, "cost_in_usd_ticks": 40_000_000})
    create = Mock(return_value=iter([
        NS(type="response.output_text.delta", delta="ok"),
        NS(type="response.completed", response=final),
    ]))
    client = make_client(cls, create)
    assert list(client.stream_with_tools(MESSAGES) if tools else client.stream_raw(MESSAGES)) == ["ok"]
    usage = client.get_token_usage()
    assert usage["total_cost_usd"] == 0.004
    assert usage["cost_source"] == "provider_reported"
    assert usage["turn_request_totals"]["requests"] == 1


def test_chat_usage_option_rejection_retries_without_option_and_stays_unpriced():
    request = httpx.Request("POST", "https://example.invalid")
    error = BadRequestError("unsupported parameter: stream_options", response=httpx.Response(400, request=request), body=None)
    create = Mock(side_effect=[error, iter([NS(choices=[NS(delta=NS(content="ok"))], usage=None)])])
    client = make_client(OpenAIClient, create)
    assert list(client.stream_raw(MESSAGES)) == ["ok"]
    assert "stream_options" in create.call_args_list[0].kwargs
    assert "stream_options" not in create.call_args_list[1].kwargs
    usage = client.get_token_usage()
    assert usage["total_cost_usd"] is None
    assert usage["cost_kind"] == "unknown"


def test_tool_iterations_sum_cost_but_keep_last_resident_then_reset():
    client = make_client(ResponsesClient, Mock(return_value=response()))
    client.query_with_tools(MESSAGES)
    client.query_with_tools(MESSAGES)
    usage = client.get_token_usage()
    assert usage["resident_tokens"] == 1000
    assert usage["output_tokens"] == 200
    assert usage["total_cost_usd"] == pytest.approx(0.0046)
    assert usage["turn_request_totals"]["requests"] == 2
    assert usage["turn_request_totals"]["input_tokens"] == 2000
    assert TurnLifecycleMixin._resolve_turn_token_usage(NS(client=client), None) == usage
    client.begin_turn_usage()
    assert client.get_token_usage() is None


@pytest.mark.parametrize("missing_first", [False, True])
def test_one_unmeasured_call_never_leaves_a_partial_sum_as_the_turn_bill(missing_first):
    accumulator = DirectTurnUsage(DEFINITION)
    calls = [{"usage": None}, response()]
    for call in calls if missing_first else reversed(calls):
        accumulator.record(call)
    usage = accumulator.snapshot()
    assert usage["total_cost_usd"] is None
    assert usage["cost_kind"] == "unknown"
    assert usage["cost_source"] is None
    assert usage["turn_request_totals"]["requests"] == 2
    if not missing_first:
        assert "resident_tokens" not in usage
        assert usage["usage_status"] == "totals_only"


@pytest.mark.parametrize(("definition", "kind", "cost"), [
    ({**DEFINITION, "access_path": "openai-oauth"}, "notional", 0.0023),
    ({**DEFINITION, "access_vendor": "local", "engine_kind": "vllm"}, "none", 0),
    ({**DEFINITION, "access_path": "unknown-api"}, "unknown", None),
    ({**DEFINITION, "pricing": {"input": 2}}, "unknown", None),
])
def test_direct_cost_uses_billing_capability_and_complete_prices(definition, kind, cost):
    accumulator = DirectTurnUsage(definition)
    accumulator.record(response())
    usage = accumulator.snapshot()
    assert usage["cost_kind"] == kind
    assert usage["total_cost_usd"] == cost


def test_failed_responses_stream_keeps_measured_spend_but_never_releases_tools():
    create = Mock(return_value=iter([
        NS(type="response.output_item.added", item=NS(type="function_call", id="tool", call_id="call", name="read")),
        NS(type="response.failed", response=response()),
    ]))
    client = make_client(ResponsesClient, create)
    with pytest.raises(RuntimeError, match="response.failed"):
        list(client.stream_with_tools(MESSAGES))
    assert client.get_token_usage()["total_cost_usd"] == pytest.approx(0.0023)


def test_overlapping_turns_on_one_cached_client_keep_separate_billed_totals():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from llm_bawt.clients.direct_usage import direct_turn_usage_scope

    client = make_client(ResponsesClient, Mock(return_value=response()))
    first, second, third = (Barrier(2, timeout=5) for _ in range(3))

    def turn_a():
        with direct_turn_usage_scope():
            client.begin_turn_usage()
            client.query_with_tools(MESSAGES)
            first.wait()
            second.wait()  # B has begun and recorded its request.
            client.query_with_tools(MESSAGES)
            client.query_with_tools(MESSAGES)
            third.wait()
            return TurnLifecycleMixin._resolve_turn_token_usage(NS(client=client), None)

    def turn_b():
        with direct_turn_usage_scope():
            first.wait()
            client.begin_turn_usage()
            client.query_with_tools(MESSAGES)
            second.wait()
            third.wait()  # A has finished its other tool-loop requests.
            return TurnLifecycleMixin._resolve_turn_token_usage(NS(client=client), None)

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.submit(turn_a), pool.submit(turn_b)
        a, b = a.result(timeout=10), b.result(timeout=10)
    assert a["cost_kind"] == b["cost_kind"] == "billed"
    assert a["turn_request_totals"]["requests"] == 3
    assert b["turn_request_totals"]["requests"] == 1
    assert a["total_cost_usd"] == pytest.approx(0.0069)
    assert b["total_cost_usd"] == pytest.approx(0.0023)


def test_worker_scope_excludes_other_clients_and_clears_no_inference_or_error_turns():
    from llm_bawt.clients.direct_usage import direct_turn_usage_scope

    client = make_client(ResponsesClient, Mock(return_value=response()))
    other = make_client(ResponsesClient, Mock(return_value=response()))
    with pytest.raises(RuntimeError, match="worker failed"):
        with direct_turn_usage_scope():
            client.begin_turn_usage()
            other.query_with_tools(MESSAGES)
            assert other.get_token_usage() is None
            assert client.get_token_usage() is None
            client.query_with_tools(MESSAGES)
            assert client.get_token_usage()["turn_request_totals"]["requests"] == 1
            raise RuntimeError("worker failed")
    with direct_turn_usage_scope():
        assert TurnLifecycleMixin._resolve_turn_token_usage(NS(client=client), None) is None
        # An unbound auxiliary call cannot revive the prior worker's snapshot.
        client.query_with_tools(MESSAGES)
        assert client.get_token_usage() is None


def test_actual_stream_worker_boundary_clears_usage_even_when_next_turn_skips_begin():
    from llm_bawt.service.turn_stream_worker import TurnStreamWorker

    client = make_client(ResponsesClient, Mock(return_value=response()))
    worker = TurnStreamWorker(NS(
        thread_binding={}, task_turn_capability=None, bot_id="fixture", user_id="fixture",
    ))

    def infer():
        client.begin_turn_usage()
        client.query_with_tools(MESSAGES)
        return client.get_token_usage()

    worker._stream_to_queue_with_context = infer
    assert worker._stream_to_queue()["total_cost_usd"] == pytest.approx(0.0023)
    worker._stream_to_queue_with_context = client.get_token_usage
    assert worker._stream_to_queue() is None
