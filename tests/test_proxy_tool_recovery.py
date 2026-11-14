"""TASK-2252: retained-incident-shaped stalls recover without executing arguments."""
import copy
import json

import pytest

from claude_code_bridge.proxy import routes
from claude_code_bridge.proxy.retry import ReplayRefusedError, RetryBudgetRegistry, RetryPolicy
from claude_code_bridge.proxy.tool_recovery import CARRIER_NAME, carrier_allowed
from claude_code_bridge.send_errors import classify_terminal_error
from tests import virtual_clock
from tests.test_proxy_progress_recovery import BODY, MODEL, THINKING, TOOL, Harness, parse_message, success_frames


def stalled_frames(*, repetition=False, complete=False, siblings=False):
    frames = [THINKING]
    if siblings:
        prior = {**TOOL, "id": "fc_prior", "call_id": "call_prior"}
        frames.extend([
            {"type": "response.output_item.added", "item": prior},
            {"type": "response.function_call_arguments.done", "item_id": "fc_prior", "arguments": '{"value":"done"}'},
            {"type": "response.output_item.done", "item": prior},
        ])
    frames.extend([
        {"type": "response.output_item.added", "item": TOOL},
        {"type": "response.function_call_arguments.delta", "item_id": TOOL["id"],
         "delta": '{"value":"private-prefix'},
    ])
    if complete:
        frames.append({"type": "response.function_call_arguments.done", "item_id": TOOL["id"],
                       "arguments": '{"value":"done"}'})
    # Retained h2 incident: 1,240 whitespace-only argument deltas then the
    # 20-second whitespace deadline. Raw argument bytes were not retained;
    # this privacy-safe fixture reconstructs that observed event shape.
    delta = " " * 129 if repetition else " "
    frames.extend({"type": "response.function_call_arguments.delta", "item_id": TOOL["id"],
                   "delta": delta} for _ in range(80 if repetition else 1240))
    return frames


async def collect(h, body, budgets):
    return [frame async for frame in routes._proxy_iter(
        h.adapter, body, MODEL, provider="openai_chatgpt", context=h.context,
        budgets=budgets, fingerprint=budgets.fingerprint(body, "openai_chatgpt", MODEL),
    )]


def events(frames):
    return [json.loads(line[5:]) for line in b"".join(frames).splitlines() if line.startswith(b"data:")]


@pytest.mark.parametrize("mode", ["sse", "lite_ws"])
@pytest.mark.parametrize("repetition", [False, True])
def test_incomplete_stall_recovers_then_fresh_call_can_succeed(monkeypatch, mode, repetition):
    async def run():
        h = Harness(monkeypatch, mode, stalled_frames(repetition=repetition), success_frames())
        if mode == "lite_ws":
            # This is a NEW model call after the carrier result, not an HTTP
            # retry of the abandoned socket. Give its new connection new data.
            h.connector.side_effect = h.feeds
        budgets = RetryBudgetRegistry()
        body = copy.deepcopy(BODY)
        try:
            frames = await collect(h, body, budgets)
            message = await parse_message(frames)
            content = json.loads(message.model_dump_json())["content"]
            tools = [block for block in content if block["type"] == "tool_use"]
            assert [tool["name"] for tool in tools] == [CARRIER_NAME]
            assert message.stop_reason == "tool_use"
            assert b"private-prefix" not in b"".join(frames)
            assert b"tool did not run" in b"".join(frames)
            assert h.attempts == 1
            assert not any(e["type"] == "error" for e in events(frames))
            same = await collect(h, body, budgets)
            assert "cannot be replayed" in events(same)[-1]["error"]["message"]
            assert h.attempts == 1
            body["messages"].extend([
                {"role": "assistant", "content": content},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tools[0]["id"],
                                               "is_error": True, "content": "No such tool available"}]},
            ])
            recovered = await parse_message(await collect(h, body, budgets))
            assert [block.name for block in recovered.content if block.type == "tool_use"] == ["check"]
            assert h.attempts == 2
        finally:
            await h.adapter.close()
    virtual_clock.run_timing(run())


@pytest.mark.parametrize("reason", ["parallel", "catalog_collision", "prior_carrier", "turn_allowance"])
def test_ambiguous_or_repeated_stall_fails_with_real_cause_and_blocks_empty_resampling(monkeypatch, reason):
    async def run():
        h = Harness(monkeypatch, "sse", stalled_frames(siblings=reason == "parallel"), success_frames())
        budgets = RetryBudgetRegistry()
        body = copy.deepcopy(BODY)
        if reason == "catalog_collision":
            body["tools"].append({"name": CARRIER_NAME, "input_schema": {"type": "object"}})
        if reason == "prior_carrier":
            body["messages"].append({"role": "assistant", "content": [
                {"type": "tool_use", "id": "notice", "name": CARRIER_NAME, "input": {}},
            ]})
        if reason == "turn_allowance":
            assert budgets.claim_tool_recovery(h.context.request_id)
        try:
            result = events(await collect(h, body, budgets))
            assert result[-1]["type"] == "error"
            assert result[-1]["error"]["type"] == "invalid_request_error"
            cause = result[-1]["error"]["message"]
            assert "ChatGPTEventTimeout" in cause and "held_arg_chars=" in cause
            assert not any(e.get("content_block", {}).get("type") == "tool_use" for e in result)
            assert h.statuses[-1]["terminal_tool_error"] == cause
            for i in range(8):
                changed = {**body, "max_tokens": 200 + i, "messages": body["messages"] + [
                    {"role": "user", "content": "Continue"},
                ]}
                again = events(await collect(h, changed, budgets))
                assert again[-1]["error"]["message"] == cause
            assert h.attempts == 1
        finally:
            await h.adapter.close()
    virtual_clock.run_timing(run())


def test_completed_arguments_remain_terminal_not_substituted(monkeypatch):
    async def run():
        h = Harness(monkeypatch, "sse", stalled_frames(complete=True), success_frames())
        try:
            result = events(await collect(h, copy.deepcopy(BODY), RetryBudgetRegistry()))
            assert result[-1]["type"] == "error"
            assert not any(e.get("content_block", {}).get("name") == CARRIER_NAME for e in result)
            assert h.attempts == 1
        finally:
            await h.adapter.close()
    virtual_clock.run_timing(run())


def test_turn_cleanup_releases_carrier_cap_and_pinned_failure():
    budgets = RetryBudgetRegistry()
    assert budgets.claim_tool_recovery("turn")
    assert not budgets.claim_tool_recovery("turn")
    budgets.fail_tool_request("turn", "typed cause")
    with pytest.raises(ReplayRefusedError, match="typed cause"):
        budgets.acquire("turn", "new", RetryPolicy)
    budgets.clear_request("turn")
    assert budgets.claim_tool_recovery("turn")
    budgets.acquire("turn", "new", RetryPolicy).release()


def test_new_real_user_turn_resets_history_cap_but_tool_result_does_not():
    body = {"messages": [{"role": "assistant", "content": [
        {"type": "tool_use", "name": CARRIER_NAME},
    ]}, {"role": "user", "content": [{"type": "tool_result", "content": "unavailable"}]}]}
    assert not carrier_allowed(body)
    body["messages"].append({"role": "user", "content": "A new task"})
    assert carrier_allowed(body)


def test_terminal_cause_is_bound_to_active_request_even_with_phase_tracker():
    from tests.test_turn_phase import _ProxyHarness, _tracker

    harness = _ProxyHarness()
    tracker, _, _ = _tracker()
    harness._turn_phase_trackers = {"req_1": tracker}
    harness.publish_proxy_status("req_1", {"state": "failed", "terminal_tool_error": "typed cause"})
    harness.publish_proxy_status("inactive", {"state": "failed", "terminal_tool_error": "wrong turn"})
    assert harness._proxy_tool_errors == {"req_1": "typed cause"}


def test_typed_tool_failure_overrides_misleading_sdk_output_cap_only_on_proxy():
    sdk = RuntimeError("Claude's response exceeded the 16384 output token maximum")
    cause = "Upstream whitespace stall; tool=check held_arg_chars=1543"
    text, raw = classify_terminal_error(sdk, direct_anthropic=False, proxy_provider="openai_chatgpt",
                                        proxy_tool_error=cause)
    assert text == cause and raw["error_code"] == "upstream_tool_arguments"
    assert classify_terminal_error(sdk, proxy_tool_error=cause) == (str(sdk), None)
