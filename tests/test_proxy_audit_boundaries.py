"""Offline TASK-2240 regressions for patch, completion and request boundaries."""
import asyncio
import json

import pytest

from claude_code_bridge.proxy import apply_patch as ap
from claude_code_bridge.proxy import routes
from claude_code_bridge.proxy.adapters.openai_chatgpt import OpenAIChatGPTAdapter
from claude_code_bridge.proxy.request_context import ProxyRequestContext
from claude_code_bridge.proxy.stream_cc import chat_completions_to_anthropic_sse
from claude_code_bridge.proxy.tool_sanitizers import sanitize_tool_arguments
from claude_code_bridge.proxy.translate import anthropic_to_responses


@pytest.mark.parametrize("ending", ["\n", "\r\n", ""])
def test_eof_patch_targets_final_duplicate_not_first(tmp_path, ending):
    path = tmp_path / "duplicate.txt"
    text = "same\nmiddle\nsame" + ending
    path.write_bytes(text.encode())
    patch = (f"*** Begin Patch\n*** Update File: {path}\n@@\n-same\n+changed\n"
             "*** End of File\n*** End Patch")
    uses = ap.patch_to_tool_uses("fixture", patch, None)
    assert len(uses) == 1
    _, tool, edit = uses[0]
    assert tool == "Edit"
    assert text.count(edit["old_string"]) == 1
    assert text.replace(edit["old_string"], edit["new_string"]) == "same\nmiddle\nchanged" + ending


def test_eof_patch_does_not_fall_back_to_earlier_match(tmp_path):
    path = tmp_path / "not-tail.txt"
    path.write_text("same\nmiddle\nother\n")
    patch = f"*** Update File: {path}\n@@\n-same\n+changed\n*** End of File"
    [(_, tool, result)] = ap.patch_to_tool_uses("fixture", patch, None)
    assert tool == "Bash" and "re-read" in result["command"]
    assert path.read_text() == "same\nmiddle\nother\n"


@pytest.mark.parametrize("ending", ["eof", "done", "unknown", "valid"])
@pytest.mark.parametrize("with_tool", [False, True])
def test_cc_completion_required_before_tool_or_success(ending, with_tool):
    async def upstream():
        delta = {"tool_calls": [{"index": 0, "id": "call-1", "function": {
            "name": "Bash", "arguments": '{"command":"true"}'}}]} if with_tool else {"content": "partial"}
        yield "data: " + json.dumps({"choices": [{"delta": delta, "finish_reason": None}]})
        if ending in {"valid", "unknown"}:
            reason = ("tool_calls" if with_tool else "stop") if ending == "valid" else "mystery"
            yield "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": reason}]})
        if ending != "eof":
            yield "data: [DONE]"

    class Adapter:
        def account_hash(self):
            return "fixture"

        async def call(self, *_):
            async for chunk in chat_completions_to_anthropic_sse(upstream(), anthropic_model="fixture/model"):
                yield chunk

    async def collect():
        return b"".join([chunk async for chunk in routes._proxy_iter(
            Adapter(), {}, "model", provider="fixture",
            context=ProxyRequestContext(request_id="fixture", provider="fixture"))])

    wire = asyncio.run(collect()).decode()
    events = [json.loads(line[5:]) for line in wire.splitlines() if line.startswith("data:")]
    tools = [event for event in events if event.get("content_block", {}).get("type") == "tool_use"]
    if ending == "valid":
        assert len(tools) == int(with_tool)
        assert events[-1]["type"] == "message_stop"
        assert not any(event["type"] == "error" for event in events)
    else:
        assert not tools
        assert events[-1]["type"] == "error"
        assert not any(event["type"] == "message_stop" for event in events)


@pytest.mark.parametrize("choice,expected", [
    ({"type": "auto"}, "auto"), ({"type": "any"}, "required"),
    ({"type": "none"}, "none"),
    ({"type": "tool", "name": "Bash"}, {"type": "function", "name": "Bash"}),
])
@pytest.mark.parametrize("prepare", [False, True])
def test_anthropic_tool_choice_uses_responses_schema(choice, expected, prepare):
    payload = anthropic_to_responses({
        "model": "fixture", "max_tokens": 100, "messages": [], "tool_choice": choice,
        "tools": [{"name": "Bash", "description": "fixture", "input_schema": {
            "type": "object", "properties": {"command": {"type": "string"}}}}],
    }, "fixture")
    if prepare:
        payload = OpenAIChatGPTAdapter().prepare_request(payload)
    assert payload["tool_choice"] == expected


@pytest.mark.parametrize("tool,param", [
    ("browser_run_code_unsafe", "code"), ("mcp__playwright__browser_run_code_unsafe", "code"),
    ("browser_evaluate", "function"), ("mcp__playwright__browser_evaluate", "function"),
])
@pytest.mark.parametrize("code", [
    "async (page) => { return /}/.test(await page.title()); }",
    "async (page) => { return /[{}]/.test(await page.title()); }",
    "async (page) => { return `${/}/.test(await page.title())}`; }",
    "async ({page}) => { return page.title(); }",
    "async (page) => { return 8 / 2; }",
])
def test_browser_sanitizer_preserves_uncertain_valid_javascript(tool, param, code):
    assert sanitize_tool_arguments({param: code}, tool)[param] == code
