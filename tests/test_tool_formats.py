"""Unit tests for tool format handlers."""

import pytest

from llm_bawt.tools.formats import ToolFormat, get_format_handler
from llm_bawt.tools.formats.base import ToolCallRequest
from llm_bawt.tools.formats.react import ReActFormatHandler
from llm_bawt.tools.formats.native_openai import NativeOpenAIFormatHandler


class TestToolFormatRegistry:
    """Tests for format handler registry."""

    def test_get_react_handler(self):
        handler = get_format_handler(ToolFormat.REACT)
        assert isinstance(handler, ReActFormatHandler)

    def test_get_native_handler(self):
        handler = get_format_handler(ToolFormat.NATIVE_OPENAI)
        assert isinstance(handler, NativeOpenAIFormatHandler)

    def test_get_handler_by_string(self):
        handler = get_format_handler("react")
        assert isinstance(handler, ReActFormatHandler)

        handler = get_format_handler("native")
        assert isinstance(handler, NativeOpenAIFormatHandler)


class TestReActFormatHandler:
    """Tests for ReAct format handler."""

    @pytest.fixture
    def handler(self):
        return ReActFormatHandler()

    def test_stop_sequences(self, handler):
        stops = handler.get_stop_sequences()
        # Stop sequences are now post-action-JSON format: they trigger after the
        # closing brace of Action Input JSON to catch model continuations.
        assert any("Observation" in s for s in stops)
        # At least one stop that includes Observation
        assert any(s for s in stops if "}".startswith(s[:1]) or "Observation" in s)

    def test_parse_basic_action(self, handler):
        response = """Thought: I need to search for user preferences
Action: search_memories
Action Input: {"query": "user preferences"}"""

        calls, remaining = handler.parse_response(response)

        assert len(calls) == 1
        assert calls[0].name == "search_memories"
        assert calls[0].arguments == {"query": "user preferences"}

    def test_parse_action_with_surrounding_text(self, handler):
        response = """Let me help you with that.

Thought: I should search the memories
Action: get_recent_history
Action Input: {"n_messages": 10}

Some trailing text"""

        calls, remaining = handler.parse_response(response)

        assert len(calls) == 1
        assert calls[0].name == "get_recent_history"
        assert calls[0].arguments == {"n_messages": 10}

    def test_parse_final_answer(self, handler):
        response = """Thought: I now have all the information
Final Answer: Based on my search, you prefer dark mode and vim keybindings."""

        calls, remaining = handler.parse_response(response)

        assert len(calls) == 0
        assert "Based on my search" in remaining
        assert "dark mode" in remaining

    def test_parse_no_action(self, handler):
        response = "I don't need to use any tools for this question."

        calls, remaining = handler.parse_response(response)

        assert len(calls) == 0
        assert remaining == response

    def test_parse_malformed_json(self, handler):
        response = """Thought: Searching
Action: search_memories
Action Input: {"query": "test",}"""  # trailing comma

        calls, remaining = handler.parse_response(response)

        # Should still parse due to json5 fallback or fix
        assert len(calls) == 1
        assert calls[0].arguments == {"query": "test"}

    def test_parse_missing_closing_brace(self, handler):
        """Parser recovers from incomplete JSON by attempting partial extraction."""
        response = """Thought: Searching
Action: search_memories
Action Input: {"query": "test" """

        calls, remaining = handler.parse_response(response)

        # Parser now handles partial JSON gracefully - extracts what it can
        assert len(calls) == 1
        assert calls[0].name == "search_memories"
        assert calls[0].arguments.get("query") == "test"

    def test_sanitize_removes_thought_markers(self, handler):
        response = """Thought: Let me think about this
Action: search_memories
Action Input: {"query": "test"}
Observation: Found 5 results
Final Answer: Here is what I found."""

        sanitized = handler.sanitize_response(response)

        assert "Thought:" not in sanitized
        assert "Action:" not in sanitized
        assert "Observation:" not in sanitized
        assert "Here is what I found" in sanitized

    def test_sanitize_extracts_final_answer(self, handler):
        response = """Some preamble text
Final Answer: This is the actual answer the user should see."""

        sanitized = handler.sanitize_response(response)

        assert sanitized == "This is the actual answer the user should see."

    def test_sanitize_removes_xml_tags(self, handler):
        response = """Here is my answer <tool_call>{"name": "test"}</tool_call> and more text"""

        sanitized = handler.sanitize_response(response)

        assert "<tool_call>" not in sanitized
        assert "</tool_call>" not in sanitized

    def test_format_result(self, handler):
        result = handler.format_result("search_memories", {"count": 5, "items": []})

        assert result.startswith("Observation:")
        assert "count" in result

    def test_format_error_result(self, handler):
        result = handler.format_result("search_memories", None, error="Connection failed")

        assert "Observation:" in result
        assert "ERROR:" in result
        assert "Connection failed" in result


class TestNativeOpenAIFormatHandler:
    """Tests for native OpenAI format handler."""

    @pytest.fixture
    def handler(self):
        return NativeOpenAIFormatHandler()

    def test_stop_sequences_empty(self, handler):
        # Native format doesn't need stop sequences
        assert handler.get_stop_sequences() == []

    def test_parse_dict_response_with_tool_calls(self, handler):
        response = {
            "content": "I'll search for that.",
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": {
                        "name": "search_memories",
                        "arguments": '{"query": "preferences"}'
                    }
                }
            ]
        }

        calls, content = handler.parse_response(response)

        assert len(calls) == 1
        assert calls[0].name == "search_memories"
        assert calls[0].arguments == {"query": "preferences"}
        assert calls[0].tool_call_id == "call_123"
        assert content == "I'll search for that."

    def test_parse_string_response(self, handler):
        response = "Just a regular response, no tools needed."

        calls, content = handler.parse_response(response)

        assert len(calls) == 0
        assert content == response

    def test_parse_multiple_tool_calls(self, handler):
        response = {
            "content": "",
            "tool_calls": [
                {"id": "call_1", "function": {"name": "tool_a", "arguments": "{}"}},
                {"id": "call_2", "function": {"name": "tool_b", "arguments": '{"x": 1}'}},
            ]
        }

        calls, _ = handler.parse_response(response)

        assert len(calls) == 2
        assert calls[0].name == "tool_a"
        assert calls[1].name == "tool_b"
        assert calls[1].arguments == {"x": 1}

    def test_format_result_returns_dict(self, handler):
        result = handler.format_result(
            "search_memories",
            {"found": True},
            tool_call_id="call_123"
        )

        assert isinstance(result, dict)
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_123"
        assert "found" in result["content"]

    def test_format_error_result(self, handler):
        result = handler.format_result(
            "search_memories",
            None,
            tool_call_id="call_123",
            error="Database unavailable"
        )

        assert "ERROR:" in result["content"]
        assert "Database unavailable" in result["content"]

    def test_sanitize_is_passthrough(self, handler):
        response = "Any response text"
        assert handler.sanitize_response(response) == response

    def test_sanitize_removes_leaked_tool_protocol(self, handler):
        response = (
            "to=functions.self commentary blah\n"
            '{"tool_uses":[{"recipient_name":"functions.self","parameters":{"action":"update"}}]}\n'
            "Final response"
        )
        sanitized = handler.sanitize_response(response)
        assert "to=functions" not in sanitized
        assert '"tool_uses"' not in sanitized
        assert "Final response" in sanitized

    def test_get_tools_schema(self, handler):
        # Create mock tool objects
        class MockParam:
            def __init__(self, name, type_, description, required=True, default=None):
                self.name = name
                self.type = type_
                self.description = description
                self.required = required
                self.default = default

        class MockTool:
            def __init__(self, name, description, parameters):
                self.name = name
                self.description = description
                self.parameters = parameters

        tools = [
            MockTool(
                "search_memories",
                "Search stored memories",
                [
                    MockParam("query", "string", "Search query", required=True),
                    MockParam("limit", "integer", "Max results", required=False, default=10),
                ]
            )
        ]

        schema = handler.get_tools_schema(tools)

        assert len(schema) == 1
        assert schema[0]["type"] == "function"
        assert schema[0]["function"]["name"] == "search_memories"
        assert "query" in schema[0]["function"]["parameters"]["properties"]
        assert "query" in schema[0]["function"]["parameters"]["required"]
        assert "limit" not in schema[0]["function"]["parameters"]["required"]


class TestRegressionDebugTurnCase:
    """Regression tests for the debug_turn.txt failure case.

    The original bug: user saw raw <tool_call> tags in response because
    parsing failed but the response was returned without sanitization.
    """

    def test_xml_tags_never_shown_to_user_react_handler(self):
        """ReAct handler should sanitize XML tags even when they appear."""
        handler = ReActFormatHandler()

        # Simulate a model that outputs XML format when ReAct was expected
        response = '''Done. Next I'm going to delete...
<tool_call>
{"name":"delete_user_attribute","arguments":{"query":"qwen"}}
</tool_call>'''

        sanitized = handler.sanitize_response(response)

        assert "<tool_call>" not in sanitized
        assert "</tool_call>" not in sanitized

    def test_native_handler_removes_leaked_tool_protocol(self):
        """Native handler strips leaked internal routing/protocol fragments."""
        handler = NativeOpenAIFormatHandler()

        response = (
            "to=multi_tool_use.parallel commentary ...\n"
            '{"tool_uses":[{"recipient_name":"functions.exec","parameters":{"command":"echo hi"}}]}\n'
            "Just a regular text response"
        )
        sanitized = handler.sanitize_response(response)

        assert "to=multi_tool_use" not in sanitized
        assert '"tool_uses"' not in sanitized
        assert "Just a regular text response" in sanitized


class TestEdgeCases:
    """Edge case tests."""

    def test_react_empty_response(self):
        handler = ReActFormatHandler()
        calls, remaining = handler.parse_response("")
        assert calls == []
        assert remaining == ""

    def test_react_none_like_response(self):
        handler = ReActFormatHandler()
        calls, remaining = handler.parse_response("")
        assert calls == []

    def test_native_empty_response(self):
        handler = NativeOpenAIFormatHandler()
        calls, remaining = handler.parse_response("")
        assert calls == []
        assert remaining == ""

    def test_react_nested_json(self):
        handler = ReActFormatHandler()
        response = '''Thought: Complex query
Action: search_memories
Action Input: {"query": "test", "filters": {"category": "preferences", "tags": ["a", "b"]}}'''

        calls, _ = handler.parse_response(response)

        assert len(calls) == 1
        assert calls[0].arguments["filters"]["category"] == "preferences"
        assert calls[0].arguments["filters"]["tags"] == ["a", "b"]

    def test_react_json_with_escaped_quotes(self):
        handler = ReActFormatHandler()
        response = '''Thought: Search for quote
Action: search_memories
Action Input: {"query": "He said \\"hello\\""}'''

        calls, _ = handler.parse_response(response)

        assert len(calls) == 1
        assert 'hello' in calls[0].arguments["query"]


class TestMultiFormatParsing:
    """Tests for multi-format tool call parsing (ReAct + alternatives)."""

    @pytest.fixture
    def handler(self):
        return ReActFormatHandler()

    def test_parse_alt_tool_format_simple(self, handler):
        """Parse # Tool: / # Arguments: format (Dolphin-style)."""
        response = '''# Tool: internet_search
# Arguments: {"num_results": 10}'''
        
        calls, remaining = handler.parse_response(response)
        
        assert len(calls) == 1
        # Should normalize tool name
        assert calls[0].name == "web_search"
        # Should normalize argument name
        assert calls[0].arguments == {"n_results": 10}

    def test_parse_alt_tool_format_with_code_block(self, handler):
        """Parse Dolphin-style format inside code block."""
        response = '''Sure, let me retrieve that.

```
# Tool: internet_search
# Arguments: {
  "num_results": 10
}
```

Here's what I found.'''
        
        calls, remaining = handler.parse_response(response)
        
        assert len(calls) == 1
        assert calls[0].name == "web_search"
        assert calls[0].arguments == {"n_results": 10}

    def test_parse_json_code_block_format(self, handler):
        """Parse JSON code block with name/arguments keys."""
        response = '''I will search for that.

```json
{"name": "search_memories", "arguments": {"query": "test"}}
```'''
        
        calls, remaining = handler.parse_response(response)
        
        assert len(calls) == 1
        assert calls[0].name == "search_memories"
        assert calls[0].arguments == {"query": "test"}

    def test_tool_name_normalization(self, handler):
        """Various tool name variations should normalize to actual tool names."""
        test_cases = [
            ("internet_search", "web_search"),
            ("google", "web_search"),
            ("current_model", "get_current_model"),
            ("show_models", "list_models"),
            # MCP names pass through; legacy memory aliases no longer exist.
            ("memory_search", "memory_search"),
            ("search_memory", "search_memory"),
        ]
        
        for input_name, expected_name in test_cases:
            response = f'''# Tool: {input_name}
# Arguments: {{"query": "test"}}'''
            
            calls, _ = handler.parse_response(response)
            assert len(calls) == 1, f"Failed for {input_name}"
            assert calls[0].name == expected_name, f"Expected {expected_name} for {input_name}, got {calls[0].name}"

    def test_argument_normalization(self, handler):
        """Various argument name variations should normalize."""
        test_cases = [
            ("num_results", "n_results"),
            ("max_results", "n_results"),
            ("search_query", "query"),
            ("q", "query"),
        ]
        
        for input_arg, expected_arg in test_cases:
            response = f'''Action: search
Action Input: {{"{input_arg}": 10}}'''
            
            calls, _ = handler.parse_response(response)
            assert len(calls) == 1
            assert expected_arg in calls[0].arguments, f"Expected {expected_arg} for {input_arg}"

    def test_mcp_tool_arguments_are_never_renamed(self, handler):
        """Discovered MCP tools keep their server-defined argument names."""
        for name in ("tool_search", "memory_search", "messages_find"):
            response = f'''Action: {name}
Action Input: {{"query": "tea", "max_results": 3, "text": "x"}}'''
            calls, _ = handler.parse_response(response)
            assert calls[0].name == name
            assert calls[0].arguments == {"query": "tea", "max_results": 3, "text": "x"}

    def test_sanitize_alt_format(self, handler):
        """Sanitization removes alternative format markers."""
        response = '''Sure, let me check.

# Tool: search_memories
# Arguments: {"query": "test"}

This will search for test.'''
        
        sanitized = handler.sanitize_response(response)
        
        assert "# Tool:" not in sanitized
        assert "# Arguments:" not in sanitized
        assert "Sure, let me check." in sanitized
        assert "This will search for test." in sanitized

    def test_stop_sequences_only_observation(self, handler):
        """Stop sequences include Observation markers and post-action continuation guards.

        The '}\\nThought' and '}\\nFinal Answer' sequences prevent the model from
        hallucinating continuations after it closes an Action Input JSON block.
        They only fire when immediately following a closing brace, so they don't
        block a normal Thought or Final Answer at the start of a turn.
        """
        stops = handler.get_stop_sequences()

        # Should stop before hallucinated observations
        assert any("Observation" in s for s in stops)

        # Should NOT stop at bare Tool: markers
        assert not any("Tool:" in s for s in stops)

        # Post-action Final Answer and Thought stops are intentional (prevent
        # hallucinating content after closing the Action Input JSON)
        final_stops = [s for s in stops if "Final Answer" in s]
        thought_stops = [s for s in stops if "Thought" in s and "Observation" not in s]
        # All such stops must be prefixed with '}' (post-JSON-close guards only)
        assert all(s.startswith("}") for s in final_stops)
        assert all(s.startswith("}") for s in thought_stops)
