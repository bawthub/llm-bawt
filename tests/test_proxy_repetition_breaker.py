"""TASK-1915: degenerate repetition stops early and never forwards a tool call."""
import asyncio
import json
import random
from types import SimpleNamespace as NS

import httpx
import pytest
from fastapi import FastAPI

from claude_code_bridge.proxy import repetition_breaker as rb
from claude_code_bridge.proxy import routes
from claude_code_bridge.proxy.retry import RetryBudgetRegistry, RetryPolicy
from claude_code_bridge.proxy.stream import responses_to_anthropic_sse
from claude_code_bridge.proxy.stream_cc import chat_completions_to_anthropic_sse



def frame(kind, **fields):
    return (f"event: {kind}\ndata: " + json.dumps(dict(type=kind, **fields)) + "\n\n").encode()


def terminal(reason):
    return [frame("message_delta", delta=dict(stop_reason=reason, stop_sequence=None), usage={"output_tokens": 7}),
            frame("message_stop")]


# The 2026-11-01 Nyx incident looped one closing tag inside a tool argument.
LOOP = "</invoke_placeholder_removed>"
LOOP_DELTAS = 2000  # ~58KB, like the incident


def prose(chars, seed=7):
    words = ["alpha", "beta", "gamma", "delta", "path", "src", "value", "for", "if",
             "return", "self", "def", "import", "x", "y", "0", "1", "\n", "    "]
    rnd = random.Random(seed)
    out, size = [], 0
    while size < chars:
        out.append(rnd.choice(words) + " ")
        size += len(out[-1])
    return "".join(out)[:chars]


def native_tool(pieces, reason="tool_use"):
    return [frame("message_start", message=dict(id="m", type="message", role="assistant", model="f", content=[], usage={})),
            frame("content_block_start", index=0, content_block=dict(type="tool_use", id="call-0", name="work", input={})),
            *[frame("content_block_delta", index=0, delta=dict(type="input_json_delta", partial_json=p)) for p in pieces],
            frame("content_block_stop", index=0), *terminal(reason)]


def native_text(pieces):
    return [frame("message_start", message=dict(id="m", type="message", role="assistant", model="f", content=[], usage={})),
            frame("content_block_start", index=0, content_block=dict(type="text", text="")),
            *[frame("content_block_delta", index=0, delta=dict(type="text_delta", text=p)) for p in pieces],
            frame("content_block_stop", index=0), *terminal("end_turn")]


def json_pieces(value_pieces):
    return ['{"value": "', *(json.dumps(p)[1:-1] for p in value_pieces), '"}']


async def source(mode, pieces, consumed):
    if mode == "native":
        for f in native_tool(pieces):
            consumed.append(1)
            yield f
    elif mode == "text":
        for f in native_text(pieces):
            consumed.append(1)
            yield f
    elif mode == "cc":
        async def lines():
            first = dict(index=0, id="call-0", function=dict(name="work", arguments=""))
            yield "data: " + json.dumps(dict(choices=[dict(delta=dict(tool_calls=[first]), finish_reason=None)]))
            for p in pieces:
                consumed.append(1)
                chunk = dict(index=0, function=dict(arguments=p))
                yield "data: " + json.dumps(dict(choices=[dict(delta=dict(tool_calls=[chunk]), finish_reason=None)]))
                yield ""
            yield "data: " + json.dumps(dict(choices=[dict(delta={}, finish_reason="tool_calls")]))
            yield "data: [DONE]"
        async for chunk in chat_completions_to_anthropic_sse(lines(), anthropic_model="f"):
            yield chunk
    else:
        item = NS(type="function_call", id="item", call_id="call-0", name="work")

        async def events():
            yield NS(type="response.output_item.added", item=item)
            for p in pieces:
                consumed.append(1)
                yield NS(type="response.function_call_arguments.delta", item_id="item", delta=p)
            yield NS(type="response.function_call_arguments.done", item_id="item", arguments="".join(pieces))
            yield NS(type="response.output_item.done", item=item)
            yield NS(type="response.completed", response=NS(status="completed", usage=None))
        async for chunk in responses_to_anthropic_sse(events(), anthropic_model="f"):
            yield chunk


def post(monkeypatch, mode, pieces, *, budgeted=False):
    consumed, first_consumed = [], []

    async def run():
        class Adapter:
            calls = 0

            def account_hash(self):
                return "fixture"

            def retry_policy(self, *_):
                return RetryPolicy(max_attempts=2, jitter_s=0)

            async def call(self, *_):
                self.calls += 1
                async for chunk in source(mode, pieces, consumed):
                    yield chunk

        adapter = Adapter()
        monkeypatch.setattr(routes, "lookup", lambda _provider: adapter)
        app = FastAPI()
        app.include_router(routes.router)
        app.state.proxy_retry_budgets = RetryBudgetRegistry() if budgeted else None
        headers = {"X-LLM-Bawt-Request-ID": "fixture-turn"} if budgeted else {}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://f") as client:
            body = dict(model="f/model", stream=True, messages=[dict(role="user", content="work")])
            result = await client.post("/v1/messages", json=body, headers=headers)
            events = [json.loads(line[5:]) for line in result.text.splitlines() if line.startswith("data:")]
            first_consumed.append(len(consumed))
            again = None
            if budgeted:
                again = await client.post("/v1/messages", json={**body, "stream": False}, headers=headers)
            return events, adapter.calls, again

    events, calls, again = asyncio.run(run())
    return events, calls, first_consumed[0], again


@pytest.mark.parametrize("mode", ["native", "cc", "responses"])
def test_tool_argument_loop_stops_early_without_forwarding(monkeypatch, mode):
    pieces = json_pieces([LOOP] * LOOP_DELTAS)
    events, calls, consumed, again = post(monkeypatch, mode, pieces, budgeted=True)
    assert not [e for e in events if e.get("content_block", {}).get("type") == "tool_use"]
    assert events[-1]["type"] == "error"
    assert events[-1]["error"]["type"] == "api_error"
    assert "looped" in events[-1]["error"]["message"]
    # Stopped near the 8KB span, not after ~58KB of junk.
    assert consumed < LOOP_DELTAS // 4
    # TASK-1966: a refused replay is final (400); cc's admitted retry fails 500.
    assert again.status_code == (500 if mode == "cc" else 400)
    if mode == "cc":
        # This translator emits no tool frame before the loop trips, so
        # nothing was committed: the lease admits one retry, which fails visibly.
        assert calls == 2 and "looped" in again.json()["error"]["message"]
    else:
        assert calls == 1  # tool block started: conservative no-replay tombstone


def test_whitespace_argument_runaway_stops_on_any_route(monkeypatch):
    events, _calls, consumed, _ = post(monkeypatch, "native", json_pieces([" "] * 20000))
    assert events[-1]["type"] == "error" and "1-char fragment" in events[-1]["error"]["message"]
    assert consumed < 10000


def test_text_loop_stops_visibly(monkeypatch):
    events, _calls, consumed, _ = post(monkeypatch, "text", ["I will check that. "] * 2000)
    assert events[-1]["type"] == "error" and "text block" in events[-1]["error"]["message"]
    assert consumed < 1000


@pytest.mark.parametrize("mode", ["native", "cc", "responses"])
def test_large_nonperiodic_argument_passes_intact(monkeypatch, mode):
    text = prose(200_000)
    pieces = json_pieces([text[i:i + 37] for i in range(0, len(text), 37)])
    events, calls, _consumed, _ = post(monkeypatch, mode, pieces)
    assert not any(e["type"] == "error" for e in events)
    tools = [e for e in events if e.get("content_block", {}).get("type") == "tool_use"]
    assert len(tools) == 1 and events[-1]["type"] == "message_stop"
    args = "".join(e["delta"]["partial_json"] for e in events
                   if e.get("delta", {}).get("type") == "input_json_delta")
    assert json.loads(args) == {"value": text}
    assert calls == 1


def test_short_repeats_below_span_pass(monkeypatch):
    # Separator lines, indentation and small repeated tables are normal output.
    body = ("=" * 80 + "\n" + prose(400) + "| a | b |\n" * 40) * 20
    events, _calls, _consumed, _ = post(monkeypatch, "native", json_pieces([body[i:i + 50] for i in range(0, len(body), 50)]))
    assert events[-1]["type"] == "message_stop"


def test_documented_false_positive_periodic_content_and_override(monkeypatch):
    # Deliberate tradeoff: valid content periodic for 8KB+ trips the default.
    zeros = json_pieces(["0" * 100] * 100)
    rows = json_pieces(["| same | row |\n"] * 1000)
    for pieces in (zeros, rows):
        events, *_ = post(monkeypatch, "native", pieces)
        assert events[-1]["type"] == "error" and "looped" in events[-1]["error"]["message"]
    monkeypatch.setenv(rb.ENV_MIN_SPAN, "0")
    for pieces in (zeros, rows):
        assert post(monkeypatch, "native", pieces)[0][-1]["type"] == "message_stop"
    monkeypatch.setenv(rb.ENV_MIN_SPAN, "65536")
    assert post(monkeypatch, "native", zeros)[0][-1]["type"] == "message_stop"


def test_config_disables_and_tunes(monkeypatch):
    monkeypatch.setenv(rb.ENV_MIN_SPAN, "0")
    events, *_ = post(monkeypatch, "native", json_pieces([LOOP] * LOOP_DELTAS))
    assert events[-1]["type"] == "message_stop"
    monkeypatch.setenv(rb.ENV_MIN_SPAN, "1024")
    monkeypatch.setenv(rb.ENV_MAX_PERIOD, "8")
    events, *_ = post(monkeypatch, "native", json_pieces([LOOP] * LOOP_DELTAS))
    assert events[-1]["type"] == "message_stop"  # period 29 > max period 8
    monkeypatch.setenv(rb.ENV_MAX_PERIOD, "64")
    events, _calls, consumed, _ = post(monkeypatch, "native", json_pieces([LOOP] * LOOP_DELTAS))
    assert events[-1]["type"] == "error" and consumed < 100
    monkeypatch.setenv(rb.ENV_MIN_SPAN, "not-a-number")
    assert rb.RepetitionBreaker().min_span == rb.DEFAULT_MIN_SPAN


def test_optional_tool_argument_size_cap(monkeypatch):
    text = prose(5000)
    pieces = json_pieces([text[i:i + 100] for i in range(0, len(text), 100)])
    assert post(monkeypatch, "native", pieces)[0][-1]["type"] == "message_stop"  # off by default
    monkeypatch.setenv(rb.ENV_MAX_TOOL_ARG_CHARS, "1000")
    events, *_ = post(monkeypatch, "native", pieces)
    assert events[-1]["type"] == "error" and "exceeded 1000 chars" in events[-1]["error"]["message"]


def test_loop_ending_between_strides_is_checked_at_block_stop():
    breaker = rb.RepetitionBreaker(min_span=128, max_period=32, max_tool_arg_chars=0)

    def delta(text):
        return frame("content_block_delta", index=0, delta=dict(type="input_json_delta", partial_json=text))

    breaker.feed(frame("content_block_start", index=0, content_block=dict(type="tool_use", id="c", name="w", input={})))
    breaker.feed(delta(prose(256)))  # exactly one stride, checked and clean
    breaker.feed(delta("ab" * 100))  # periodic run arrives below the next stride
    assert breaker._blocks[0].pending == 200
    with pytest.raises(rb.RepetitionLoopError):
        breaker.feed(frame("content_block_stop", index=0))


def test_periodic_tail_unit():
    assert rb.periodic_tail(LOOP * 400, 8192, 512) == len(LOOP)
    assert rb.periodic_tail(prose(20_000), 8192, 512) is None
    assert rb.periodic_tail("x" * 100, 8192, 512) is None
    assert rb.periodic_tail(" " * 9000, 8192, 512) == 1
    assert rb.periodic_tail("ab" * 5000, 8192, 0) is None
