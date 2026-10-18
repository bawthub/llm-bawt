"""Comprehensive tests for the tool-specific argument sanitizer pipeline.

Covers:
- JS scanning helpers (strings, template literals, comments, brace balancing)
- JS code trailing-token stripper (real payloads + edge cases)
- JSON trailing-text recovery
- Registry (registration, composition, error resilience)
- Integration round-trips through stream.py and stream_cc.py
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from claude_code_bridge.proxy.tool_sanitizers import (
    _JS_CODE_PARAMS,
    _REGISTRY,
    _scan_js_balanced_end,
    _skip_js_string,
    _skip_js_template,
    recover_trailing_json,
    register,
    sanitize_tool_arguments,
    strip_js_trailing_tokens,
)


# ---------------------------------------------------------------------------
# JS scanning helpers
# ---------------------------------------------------------------------------


class TestSkipJsString:
    def test_double_quoted(self):
        assert _skip_js_string('"hello"', 0) == 7

    def test_single_quoted(self):
        assert _skip_js_string("'hello'", 0) == 7

    def test_escaped_quote(self):
        assert _skip_js_string(r'"he\"llo"', 0) == 9

    def test_escaped_backslash(self):
        assert _skip_js_string(r'"hello\\"', 0) == 9

    def test_unterminated(self):
        assert _skip_js_string('"hello', 0) == 6

    def test_offset(self):
        s = "x = 'test'; y"
        assert _skip_js_string(s, 4) == 10

    def test_empty_string(self):
        assert _skip_js_string('""', 0) == 2


class TestSkipJsTemplate:
    def test_simple(self):
        assert _skip_js_template("`hello`", 0) == 7

    def test_with_interpolation(self):
        s = "`hello ${name}`"
        assert _skip_js_template(s, 0) == len(s)

    def test_nested_braces_in_interpolation(self):
        s = "`${JSON.stringify({a: 1})}`"
        assert _skip_js_template(s, 0) == len(s)

    def test_nested_template(self):
        s = "`outer ${`inner ${x}`}`"
        assert _skip_js_template(s, 0) == len(s)

    def test_escaped_backtick(self):
        assert _skip_js_template(r"`hello \` world`", 0) == 16

    def test_unterminated(self):
        assert _skip_js_template("`hello", 0) == 6


class TestScanJsBalancedEnd:
    def test_simple_block(self):
        assert _scan_js_balanced_end("{ x }") == 5

    def test_function_expression(self):
        code = "async (page) => { await page.click('#btn'); }"
        assert _scan_js_balanced_end(code) == len(code)

    def test_nested_braces(self):
        code = "() => { const obj = {a: {b: 1}}; return obj; }"
        assert _scan_js_balanced_end(code) == len(code)

    def test_no_braces(self):
        assert _scan_js_balanced_end("x => x + 1") is None

    def test_unterminated(self):
        assert _scan_js_balanced_end("{ open") is None

    def test_string_with_brace(self):
        code = '() => { const s = "}"; return s; }'
        assert _scan_js_balanced_end(code) == len(code)

    def test_template_with_brace(self):
        code = "() => { const s = `${obj}`; }"
        assert _scan_js_balanced_end(code) == len(code)

    def test_template_with_nested_object(self):
        code = "() => { return `${JSON.stringify({a: 1})}`; }"
        assert _scan_js_balanced_end(code) == len(code)

    def test_single_line_comment(self):
        code = "() => { // closing } brace\nreturn 1; }"
        assert _scan_js_balanced_end(code) == len(code)

    def test_multiline_comment(self):
        code = "() => { /* } */ return 1; }"
        assert _scan_js_balanced_end(code) == len(code)

    def test_unmatched_close_returns_none(self):
        assert _scan_js_balanced_end("} extra") is None

    def test_trailing_garbage_after_balance(self):
        code = "() => { return 1; } I think this is correct"
        # Should find balance right after the first }
        pos = _scan_js_balanced_end(code)
        assert code[:pos] == "() => { return 1; }"

    def test_mixed_string_types(self):
        code = """() => { const a = "test}"; const b = 'test}'; return a + b; }"""
        assert _scan_js_balanced_end(code) == len(code)


# ---------------------------------------------------------------------------
# JS trailing-token stripper
# ---------------------------------------------------------------------------


class TestStripJsTrailingTokens:
    """Tests using realistic corrupted payloads from reasoning model leaks."""

    def test_clean_code_unchanged(self):
        args = {"code": "async (page) => { await page.click('#btn'); }"}
        result = strip_js_trailing_tokens(args, "browser_run_code_unsafe")
        assert result == args

    def test_strips_trailing_reasoning(self):
        code = (
            "async (page) => { await page.click('#btn'); }"
            " please_dont_include_this_lol?"
        )
        result = strip_js_trailing_tokens(
            {"code": code}, "browser_run_code_unsafe"
        )
        assert result["code"] == "async (page) => { await page.click('#btn'); }"

    def test_strips_natural_language(self):
        code = (
            "async (page) => {\n"
            "  const title = await page.title();\n"
            "  return title;\n"
            "} I should verify that the title matches the expected value. "
            "Let me also check the URL to make sure we're on the right page."
        )
        result = strip_js_trailing_tokens(
            {"code": code}, "browser_run_code_unsafe"
        )
        expected = (
            "async (page) => {\n"
            "  const title = await page.title();\n"
            "  return title;\n"
            "}"
        )
        assert result["code"] == expected

    def test_preserves_trailing_semicolon(self):
        code = "async (page) => { return 1; };"
        result = strip_js_trailing_tokens(
            {"code": code}, "browser_run_code_unsafe"
        )
        assert result["code"] == code

    def test_preserves_trailing_whitespace(self):
        code = "async (page) => { return 1; }  \n"
        result = strip_js_trailing_tokens(
            {"code": code}, "browser_run_code_unsafe"
        )
        assert result["code"] == code

    def test_no_braces_unchanged(self):
        """Arrow function without braces -- no balanced point, leave as-is."""
        code = "(page) => page.title()"
        result = strip_js_trailing_tokens(
            {"code": code}, "browser_run_code_unsafe"
        )
        assert result["code"] == code

    def test_mcp_prefixed_tool_name(self):
        code = "async (page) => { return 1; } extra garbage"
        result = strip_js_trailing_tokens(
            {"code": code}, "mcp__playwright__browser_run_code_unsafe"
        )
        assert result["code"] == "async (page) => { return 1; }"

    def test_browser_evaluate_preserves_wrapped_function(self):
        code = "(() => { return document.title; }) this should work"
        result = strip_js_trailing_tokens(
            {"function": code}, "browser_evaluate"
        )
        # ) after } may close a wrapper; uncertain tails go to syntax preflight.
        assert result["function"] == code

    def test_browser_evaluate_strips_after_brace(self):
        """Leaked reasoning after an actual Playwright function argument."""
        code = "() => { return document.title; } I think this returns the title"
        result = strip_js_trailing_tokens(
            {"function": code}, "mcp__playwright__browser_evaluate"
        )
        assert result["function"] == "() => { return document.title; }"

    def test_browser_evaluate_does_not_touch_unrelated_argument(self):
        args = {"expression": "() => { return 1; } extra", "function": "() => 1"}
        assert strip_js_trailing_tokens(args, "browser_evaluate") == args

    def test_browser_evaluate_preserves_regex_brace(self):
        code = "() => { return /} stray/.test(document.title); } leaked reasoning"
        assert strip_js_trailing_tokens({"function": code}, "browser_evaluate")["function"] == code

    def test_browser_evaluate_preserves_destructured_default_parameter(self):
        code = "({ id } = { id: 1 }) => { return id; } leaked reasoning"
        assert strip_js_trailing_tokens({"function": code}, "browser_evaluate")["function"] == code

    def test_browser_evaluate_preserves_trailing_comment(self):
        code = "() => { return 1; } // note: [sic]"
        assert strip_js_trailing_tokens({"function": code}, "browser_evaluate")["function"] == code

    def test_ignores_non_code_tools(self):
        args = {"command": "echo hello } extra"}
        result = strip_js_trailing_tokens(args, "Bash")
        assert result == args

    def test_preserves_method_chaining_after_brace(self):
        """If } is followed by . (method chain), don't strip."""
        code = "(() => { return 1; }).toString()"
        result = strip_js_trailing_tokens(
            {"code": code}, "browser_run_code_unsafe"
        )
        assert result["code"] == code

    def test_preserves_bracket_access_after_brace(self):
        """Closing paren after } is a safe continuation -- don't strip."""
        code = "(() => { return {a: 1}; })()['a']"
        result = strip_js_trailing_tokens(
            {"code": code}, "browser_run_code_unsafe"
        )
        # The first balanced } is the outer arrow body.  Trailing starts
        # with ")" which is in the safe continuation set, so nothing is stripped.
        assert result["code"] == code

    def test_complex_nested_code(self):
        """Real-world-ish Playwright code with nested objects and strings."""
        code = (
            "async (page) => {\n"
            "  const data = {name: 'test', config: {timeout: 5000}};\n"
            '  await page.fill("#input", JSON.stringify(data));\n'
            "  const result = await page.evaluate(() => {\n"
            "    return {status: 'ok'};\n"
            "  });\n"
            "  return result;\n"
            "} Now let me verify the result"
        )
        result = strip_js_trailing_tokens(
            {"code": code}, "browser_run_code_unsafe"
        )
        # The first balanced } is the inner evaluate callback -- no!
        # Let me trace depth:
        # async (page) => {  -- depth 1
        #   const data = {   -- depth 2
        #     name: 'test', config: { -- depth 3
        #       timeout: 5000 -- depth 3
        #     }  -- depth 2
        #   };  -- depth 1
        #   ... evaluate(() => { -- depth 2
        #     return {status: 'ok'};  -- depth 3 -> 2
        #   }); -- depth 1
        #   return result;
        # } -- depth 0 <-- first balanced
        expected_end = code.index("} Now")
        assert result["code"] == code[: expected_end + 1]

    def test_code_with_template_literal(self):
        code = (
            "async (page) => {\n"
            "  const url = `https://example.com/${id}`;\n"
            "  await page.goto(url);\n"
            "} this is leaked reasoning"
        )
        result = strip_js_trailing_tokens(
            {"code": code}, "browser_run_code_unsafe"
        )
        assert "leaked reasoning" not in result["code"]
        assert result["code"].endswith("}")

    def test_code_with_comments(self):
        code = (
            "async (page) => {\n"
            "  // Navigate to the page }\n"
            "  /* multi-line } comment */\n"
            "  return true;\n"
            "} The code navigates to the page"
        )
        result = strip_js_trailing_tokens(
            {"code": code}, "browser_run_code_unsafe"
        )
        assert result["code"].endswith("}")
        assert "navigates to the page" not in result["code"]

    def test_multiple_leaked_sentences(self):
        """Simulate 10+ tokens of leaked reasoning."""
        code = (
            "async (page) => { await page.click('#submit'); }"
            " I need to wait for the navigation to complete."
            " After that, I should check the response status."
            " Let me also verify the page title changed."
            " This approach should work correctly."
        )
        result = strip_js_trailing_tokens(
            {"code": code}, "browser_run_code_unsafe"
        )
        assert result["code"] == "async (page) => { await page.click('#submit'); }"

    def test_non_string_code_param_ignored(self):
        args = {"code": 42}
        result = strip_js_trailing_tokens(args, "browser_run_code_unsafe")
        assert result == args

    def test_missing_code_param_ignored(self):
        args = {"description": "a test"}
        result = strip_js_trailing_tokens(args, "browser_run_code_unsafe")
        assert result == args


# ---------------------------------------------------------------------------
# JSON trailing-text recovery
# ---------------------------------------------------------------------------


class TestRecoverTrailingJson:
    def test_clean_json_returned_as_is(self):
        raw = '{"command": "echo hello"}'
        assert recover_trailing_json(raw) == raw

    def test_trailing_reasoning_stripped(self):
        raw = '{"command": "echo hello"} I should verify the output'
        result = recover_trailing_json(raw)
        assert result == '{"command": "echo hello"}'

    def test_trailing_after_array(self):
        raw = '[1, 2, 3] and then we can process'
        result = recover_trailing_json(raw)
        assert result == '[1, 2, 3]'

    def test_nested_objects(self):
        raw = '{"a": {"b": {"c": 1}}} extra text'
        result = recover_trailing_json(raw)
        parsed = json.loads(result)
        assert parsed == {"a": {"b": {"c": 1}}}

    def test_string_with_braces(self):
        raw = '{"code": "function() { return 1; }"} leaked tokens'
        result = recover_trailing_json(raw)
        parsed = json.loads(result)
        assert parsed == {"code": "function() { return 1; }"}

    def test_not_json(self):
        assert recover_trailing_json("hello world") is None

    def test_empty(self):
        assert recover_trailing_json("") is None

    def test_only_whitespace(self):
        assert recover_trailing_json("   ") is None

    def test_no_trailing(self):
        raw = '{"x": 1}'
        assert recover_trailing_json(raw) == raw

    def test_escaped_quotes_in_strings(self):
        raw = r'{"msg": "he said \"hello\""} extra'
        result = recover_trailing_json(raw)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["msg"] == 'he said "hello"'

    def test_whitespace_before_json(self):
        raw = '   {"x": 1}  trailing'
        result = recover_trailing_json(raw)
        assert result == '{"x": 1}'


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_sanitize_tool_arguments_applies_matching(self):
        """The built-in JS stripper fires for browser_run_code_unsafe."""
        args = {
            "code": "async (page) => { return 1; } leaked reasoning text"
        }
        result = sanitize_tool_arguments(args, "browser_run_code_unsafe")
        assert result["code"] == "async (page) => { return 1; }"

    def test_sanitize_tool_arguments_skips_unmatched(self):
        args = {"command": "echo hello"}
        result = sanitize_tool_arguments(args, "Bash")
        assert result == args

    def test_custom_sanitizer_registration(self):
        """Register a custom sanitizer and verify it fires."""
        calls = []

        def my_sanitizer(args: dict, tool_name: str) -> dict:
            calls.append(tool_name)
            return {**args, "custom": True}

        original_len = len(_REGISTRY)
        try:
            register("my_test_tool", my_sanitizer)
            result = sanitize_tool_arguments({"x": 1}, "my_test_tool")
            assert result["custom"] is True
            assert calls == ["my_test_tool"]

            # MCP-prefixed name also matches
            result2 = sanitize_tool_arguments({"x": 1}, "mcp__server__my_test_tool")
            assert result2["custom"] is True
        finally:
            # Clean up
            del _REGISTRY[original_len:]

    def test_sanitizers_compose(self):
        """Multiple sanitizers fire in registration order."""
        original_len = len(_REGISTRY)
        try:
            register("compose_test", lambda a, _: {**a, "first": True})
            register("compose_test", lambda a, _: {**a, "second": True})
            result = sanitize_tool_arguments({}, "compose_test")
            assert result["first"] is True
            assert result["second"] is True
        finally:
            del _REGISTRY[original_len:]

    def test_sanitizer_exception_swallowed(self):
        """A failing sanitizer is logged and skipped -- doesn't crash."""
        original_len = len(_REGISTRY)
        try:
            register("bomb_test", lambda a, _: 1 / 0)  # ZeroDivisionError
            register("bomb_test", lambda a, _: {**a, "survived": True})
            result = sanitize_tool_arguments({"ok": True}, "bomb_test")
            assert result["survived"] is True
            assert result["ok"] is True
        finally:
            del _REGISTRY[original_len:]

    def test_callable_matcher(self):
        original_len = len(_REGISTRY)
        try:
            register(lambda n: "magic" in n, lambda a, _: {**a, "magic": True})
            assert sanitize_tool_arguments({}, "do_magic_stuff")["magic"] is True
            assert "magic" not in sanitize_tool_arguments({}, "Bash")
        finally:
            del _REGISTRY[original_len:]


# ---------------------------------------------------------------------------
# Integration: stream_cc._clean_tool_arguments
# ---------------------------------------------------------------------------


class TestCleanToolArgumentsIntegration:
    """Verify _clean_tool_arguments uses both recovery and sanitizer pipeline."""

    def test_clean_json_passes_through(self):
        from claude_code_bridge.proxy.stream_cc import _clean_tool_arguments

        result = _clean_tool_arguments(
            '{"command":"echo hi"}',
            tool_name="Bash",
            required_by_tool={},
        )
        assert json.loads(result) == {"command": "echo hi"}

    def test_trailing_json_recovered(self):
        from claude_code_bridge.proxy.stream_cc import _clean_tool_arguments

        raw = '{"command": "echo hi"} I think this is right'
        result = _clean_tool_arguments(
            raw,
            tool_name="Bash",
            required_by_tool={},
        )
        assert json.loads(result) == {"command": "echo hi"}

    def test_js_code_sanitized(self):
        from claude_code_bridge.proxy.stream_cc import _clean_tool_arguments

        raw = json.dumps({
            "code": "async (page) => { return 1; } leaked reasoning"
        })
        result = _clean_tool_arguments(
            raw,
            tool_name="browser_run_code_unsafe",
            required_by_tool={},
        )
        parsed = json.loads(result)
        assert parsed["code"] == "async (page) => { return 1; }"

    def test_empty_string_cleanup_still_works(self):
        from claude_code_bridge.proxy.stream_cc import _clean_tool_arguments

        raw = json.dumps({"command": "echo hi", "pages": ""})
        result = _clean_tool_arguments(
            raw,
            tool_name="Read",
            required_by_tool={"Read": frozenset(["command"])},
        )
        parsed = json.loads(result)
        assert "pages" not in parsed
        assert parsed["command"] == "echo hi"

    def test_isolation_worktree_still_stripped(self):
        from claude_code_bridge.proxy.stream_cc import _clean_tool_arguments

        raw = json.dumps({"command": "ls", "isolation": "worktree"})
        result = _clean_tool_arguments(
            raw,
            tool_name="Agent",
            required_by_tool={},
        )
        parsed = json.loads(result)
        assert "isolation" not in parsed

    def test_combined_recovery_and_sanitize(self):
        """Trailing JSON recovery + tool sanitizer in one call."""
        from claude_code_bridge.proxy.stream_cc import _clean_tool_arguments

        # JSON with trailing text, AND the code param has leaked tokens
        inner = json.dumps({
            "code": "async (page) => { return 1; } extra garbage"
        })
        raw = inner + " more leaked reasoning"
        result = _clean_tool_arguments(
            raw,
            tool_name="browser_run_code_unsafe",
            required_by_tool={},
        )
        parsed = json.loads(result)
        assert parsed["code"] == "async (page) => { return 1; }"


# ---------------------------------------------------------------------------
# Integration: stream.py Responses API round-trip
# ---------------------------------------------------------------------------


def _sse_payloads(frames: list[bytes]) -> list[dict]:
    out: list[dict] = []
    for frame in frames:
        for line in frame.decode().splitlines():
            if line.startswith("data: "):
                out.append(json.loads(line[6:]))
    return out


class TestResponsesStreamIntegration:
    """Verify the full Responses → Anthropic SSE stream sanitizes tool args."""

    def _make_events(self, tool_name: str, arguments: str) -> list[Any]:
        """Create a minimal Responses event sequence for one tool call."""
        item_id = "item_001"
        return [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(
                type="response.output_item.added",
                item=SimpleNamespace(
                    type="function_call",
                    id=item_id,
                    call_id="call_001",
                    name=tool_name,
                ),
            ),
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                item_id=item_id,
                delta=arguments,
            ),
            SimpleNamespace(
                type="response.function_call_arguments.done",
                item_id=item_id,
                arguments=arguments,
            ),
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(type="function_call", id=item_id),
            ),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    status="completed",
                    usage=SimpleNamespace(
                        input_tokens=10,
                        output_tokens=5,
                        input_tokens_details=SimpleNamespace(cached_tokens=0),
                    ),
                ),
            ),
        ]

    def _run_stream(
        self,
        events: list[Any],
        tool_schemas: list[dict] | None = None,
    ) -> list[dict]:
        from claude_code_bridge.proxy.stream import responses_to_anthropic_sse

        async def source():
            for e in events:
                yield e

        async def run():
            return [
                frame
                async for frame in responses_to_anthropic_sse(
                    source(),
                    anthropic_model="test-model",
                    tool_schemas=tool_schemas,
                )
            ]

        frames = asyncio.run(run())
        return _sse_payloads(frames)

    def test_clean_tool_call(self):
        args_json = json.dumps({"command": "echo hello"})
        events = self._make_events("Bash", args_json)
        payloads = self._run_stream(events)

        # Find the input_json_delta
        deltas = [
            p for p in payloads
            if p.get("type") == "content_block_delta"
            and p.get("delta", {}).get("type") == "input_json_delta"
        ]
        assert len(deltas) == 1
        assert json.loads(deltas[0]["delta"]["partial_json"]) == {
            "command": "echo hello"
        }

    def test_js_code_sanitized_in_stream(self):
        corrupted = json.dumps({
            "code": "async (page) => { return 1; } leaked reasoning text"
        })
        events = self._make_events("browser_run_code_unsafe", corrupted)
        payloads = self._run_stream(events)

        deltas = [
            p for p in payloads
            if p.get("type") == "content_block_delta"
            and p.get("delta", {}).get("type") == "input_json_delta"
        ]
        assert len(deltas) == 1
        parsed = json.loads(deltas[0]["delta"]["partial_json"])
        assert parsed["code"] == "async (page) => { return 1; }"

    def test_browser_evaluate_function_sanitized_in_stream(self):
        corrupted = json.dumps({
            "function": "() => { return document.title; } leaked reasoning",
            "element": "page",
            "target": "body",
        })
        events = self._make_events("mcp__playwright__browser_evaluate", corrupted)
        payloads = self._run_stream(events)
        deltas = [
            p for p in payloads
            if p.get("type") == "content_block_delta"
            and p.get("delta", {}).get("type") == "input_json_delta"
        ]
        assert len(deltas) == 1
        parsed = json.loads(deltas[0]["delta"]["partial_json"])
        assert parsed["function"] == "() => { return document.title; }"
        assert parsed["target"] == "body"

    def test_trailing_json_recovered_in_stream(self):
        """Malformed JSON (trailing text) is recovered in the stream."""
        raw = '{"command": "ls"} some reasoning text'
        events = self._make_events("Bash", raw)
        payloads = self._run_stream(events)

        deltas = [
            p for p in payloads
            if p.get("type") == "content_block_delta"
            and p.get("delta", {}).get("type") == "input_json_delta"
        ]
        assert len(deltas) == 1
        parsed = json.loads(deltas[0]["delta"]["partial_json"])
        assert parsed == {"command": "ls"}

    def test_empty_optional_still_stripped(self):
        raw = json.dumps({"file_path": "/tmp/x", "pages": ""})
        events = self._make_events("Read", raw)
        tool_schemas = [
            {
                "name": "Read",
                "input_schema": {
                    "type": "object",
                    "required": ["file_path"],
                    "properties": {
                        "file_path": {"type": "string"},
                        "pages": {"type": "string"},
                    },
                },
            }
        ]
        payloads = self._run_stream(events, tool_schemas=tool_schemas)

        deltas = [
            p for p in payloads
            if p.get("type") == "content_block_delta"
            and p.get("delta", {}).get("type") == "input_json_delta"
        ]
        assert len(deltas) == 1
        parsed = json.loads(deltas[0]["delta"]["partial_json"])
        assert "pages" not in parsed
        assert parsed["file_path"] == "/tmp/x"
