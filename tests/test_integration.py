"""Integration tests for full tool calling flow.

These tests verify end-to-end behavior with mocked LLM responses
but real parsing, execution, and streaming logic.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Iterator

from llm_bawt.tools.formats import ToolFormat
from llm_bawt.tools.formats.react import ReActFormatHandler
from llm_bawt.tools.loop import ToolLoop
from llm_bawt.tools.streaming import stream_with_tools


class MockMessage:
    """Simple mock message."""
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self):
        return {"role": self.role, "content": self.content}


# =============================================================================
# ReAct Format Edge Cases
# =============================================================================

class TestReActNoArgumentTools:
    """Test tools that don't require Action Input."""

    def test_parse_action_without_input(self):
        """Tools without required params should parse with just Action:"""
        handler = ReActFormatHandler()
        
        # Model omits Action Input for no-arg tools
        response = """Thought: I need to get the user's profile
Action: get_user_profile"""

        calls, remaining = handler.parse_response(response)

        assert len(calls) == 1
        assert calls[0].name == "get_user_profile"
        assert calls[0].arguments == {}

    def test_parse_action_without_input_with_trailing_newline(self):
        """Handle trailing whitespace after action name."""
        handler = ReActFormatHandler()
        
        response = """Thought: Getting profile
Action: get_user_profile
"""

        calls, remaining = handler.parse_response(response)

        assert len(calls) == 1
        assert calls[0].name == "get_user_profile"

    def test_parse_action_with_empty_input(self):
        """Handle explicit empty JSON object."""
        handler = ReActFormatHandler()
        
        response = """Thought: Getting profile
Action: get_user_profile
Action Input: {}"""

        calls, remaining = handler.parse_response(response)

        assert len(calls) == 1
        assert calls[0].arguments == {}


class TestReActStopSequences:
    """Test that stop sequences don't break normal flow."""

    def test_stop_sequences_dont_include_final_answer(self):
        """Bare 'Final Answer' should NOT be a stop sequence — it's our output.

        Post-action guards like '}\\nFinal Answer' are intentional (they only fire
        after a closing JSON brace) and are therefore allowed.
        """
        handler = ReActFormatHandler()
        stops = handler.get_stop_sequences()

        # The bare string "Final Answer:" should never be a stop (would break output)
        assert "\nFinal Answer:" not in stops
        assert "Final Answer" not in stops  # no exact-match entry

        # Any Final Answer stops must be post-JSON-close guards (prefixed with '}')
        final_stops = [s for s in stops if "Final Answer" in s]
        assert all(s.startswith("}") for s in final_stops)

    def test_stop_sequences_only_observation(self):
        """Observation variants are the primary stop sequences.

        Post-action guards ('}\\nThought', '}\\nFinal Answer') are also present to
        prevent the model hallucinating continuations after a closed Action Input
        JSON block. They only fire after '}' so they don't interfere with normal
        Thought or Final Answer output.
        """
        handler = ReActFormatHandler()
        stops = handler.get_stop_sequences()

        # Should stop before hallucinated observations
        assert any("Observation" in s for s in stops)

        # Post-action Thought stops must be '}'-prefixed (not free-standing)
        thought_stops = [s for s in stops if "Thought" in s and "Observation" not in s]
        assert all(s.startswith("}") for s in thought_stops)


# =============================================================================
# Tool Loop Integration
# =============================================================================

class TestToolLoopFullFlow:
    """Test complete tool execution flow."""

    def test_tool_with_no_args_executes(self):
        """Tools without arguments should execute properly."""
        # Mock client that returns a tool call, then final answer
        responses = [
            "Thought: Getting profile\nAction: get_user_profile",
            "Thought: I have the profile\nFinal Answer: Your name is Operator."
        ]
        call_idx = [0]
        
        def mock_query(messages, stop=None, **kwargs):
            resp = responses[min(call_idx[0], len(responses) - 1)]
            call_idx[0] += 1
            return resp

        client = Mock()
        client.query = mock_query
        client.supports_native_tools = Mock(return_value=False)

        # Mock executor to avoid real tool execution
        with patch('llm_bawt.tools.loop.ToolExecutor') as mock_executor_class:
            mock_executor = Mock()
            mock_executor.execute.return_value = "Profile: name=Operator"
            mock_executor_class.return_value = mock_executor

            loop = ToolLoop(user_id="test", tool_format=ToolFormat.REACT)
            result = loop.run([MockMessage("user", "show profile")], client)

            # Tool was executed
            assert mock_executor.execute.called
            # Got final answer
            assert "Operator" in result

    def test_tool_loop_continues_after_execution(self):
        """After tool execution, loop should query LLM again for response."""
        responses = [
            "Thought: Need to search\nAction: search_memories\nAction Input: {\"query\": \"test\"}",
            "Thought: Found it\nFinal Answer: Here's what I found."
        ]
        call_idx = [0]
        
        def mock_query(messages, stop=None, **kwargs):
            resp = responses[min(call_idx[0], len(responses) - 1)]
            call_idx[0] += 1
            return resp

        client = Mock()
        client.query = mock_query
        client.supports_native_tools = Mock(return_value=False)

        with patch('llm_bawt.tools.loop.ToolExecutor') as mock_executor_class:
            mock_executor = Mock()
            mock_executor.execute.return_value = "Found: test data"
            mock_executor_class.return_value = mock_executor

            loop = ToolLoop(user_id="test", tool_format=ToolFormat.REACT)
            result = loop.run([MockMessage("user", "search")], client)

            # LLM was called twice (tool call + final answer)
            assert call_idx[0] == 2
            assert "Here's what I found" in result


# =============================================================================
# Streaming Integration
# =============================================================================

class TestStreamingToolLoop:
    """Test streaming tool loop behavior."""

    def test_streaming_yields_final_response(self):
        """Streaming should yield content after tool execution."""
        
        def mock_stream_fn(messages, stop_sequences=None):
            # First call: tool call
            if len(messages) == 1:
                yield "Thought: Getting profile\n"
                yield "Action: get_user_profile"
            else:
                # Second call: final response
                yield "Your profile shows name=Operator"

        with patch('llm_bawt.tools.streaming.ToolExecutor') as mock_executor_class:
            mock_executor = Mock()
            mock_executor.execute.return_value = "name=Operator, occupation=developer"
            mock_executor_class.return_value = mock_executor

            result_chunks = list(stream_with_tools(
                messages=[MockMessage("user", "show profile")],
                stream_fn=mock_stream_fn,
                user_id="test",
                bot_id="proto",
                tool_format=ToolFormat.REACT,
            ))

            # Should have yielded the final response
            full_result = "".join(result_chunks)
            assert "Operator" in full_result

    def test_streaming_no_tool_yields_directly(self):
        """Non-tool responses should stream through immediately."""
        
        def mock_stream_fn(messages, stop_sequences=None):
            yield "Hello! "
            yield "How can I help?"

        result_chunks = list(stream_with_tools(
            messages=[MockMessage("user", "hi")],
            stream_fn=mock_stream_fn,
            user_id="test",
            bot_id="proto",
            tool_format=ToolFormat.REACT,
        ))

        full_result = "".join(result_chunks)
        assert "Hello" in full_result
        assert "How can I help" in full_result

    def test_streaming_skips_stop_sequences_after_tool(self):
        """After tool execution, stop sequences should be disabled."""
        stop_seq_on_calls = []
        
        def mock_stream_fn(messages, stop_sequences=None):
            stop_seq_on_calls.append(stop_sequences)
            if len(messages) == 1:
                yield "Thought: Need profile\nAction: get_user_profile"
            else:
                yield "Final Answer: Got it!"

        with patch('llm_bawt.tools.streaming.ToolExecutor') as mock_executor_class:
            mock_executor = Mock()
            mock_executor.execute.return_value = "profile data"
            mock_executor_class.return_value = mock_executor

            list(stream_with_tools(
                messages=[MockMessage("user", "profile")],
                stream_fn=mock_stream_fn,
                user_id="test",
                bot_id="proto",
                tool_format=ToolFormat.REACT,
            ))

            # First call should have stop sequences
            assert stop_seq_on_calls[0] is not None
            # Second call (after tool) should have None (no stop sequences)
            assert stop_seq_on_calls[1] is None

    def test_streaming_collects_tool_call_details(self):
        """Tool call details should be collected for turn-log persistence."""
        details = []

        def mock_stream_fn(messages, stop_sequences=None):
            if len(messages) == 1:
                yield "Thought: Need profile\nAction: get_user_profile\nAction Input: {}"
            else:
                yield "Final Answer: Done"

        with patch('llm_bawt.tools.streaming.ToolExecutor') as mock_executor_class:
            mock_executor = Mock()
            mock_executor.execute.return_value = "name=Operator"
            mock_executor_class.return_value = mock_executor

            list(stream_with_tools(
                messages=[MockMessage("user", "profile")],
                stream_fn=mock_stream_fn,
                user_id="test",
                bot_id="proto",
                tool_format=ToolFormat.REACT,
                tool_call_details=details,
            ))

        assert len(details) == 1
        assert details[0]["tool"] == "get_user_profile"
        assert details[0]["parameters"] == {}
        assert details[0]["result"] == "name=Operator"
        assert details[0]["iteration"] == 1


# =============================================================================
# Response Sanitization
# =============================================================================

class TestResponseSanitization:
    """Test that tool markers are stripped from final output."""

    def test_thought_markers_removed(self):
        """Thought: lines should be stripped from final response."""
        handler = ReActFormatHandler()
        
        response = """Thought: I now have the information
Final Answer: Here is your answer with all the details."""

        sanitized = handler.sanitize_response(response)
        
        assert "Thought:" not in sanitized
        assert "Here is your answer" in sanitized

    def test_action_markers_removed(self):
        """Action markers should be stripped."""
        handler = ReActFormatHandler()
        
        response = """Action: some_tool
Action Input: {}
Final Answer: The result is 42."""

        sanitized = handler.sanitize_response(response)
        
        assert "Action:" not in sanitized
        assert "Action Input:" not in sanitized
        assert "42" in sanitized

    def test_final_answer_prefix_removed(self):
        """Final Answer: prefix should be stripped."""
        handler = ReActFormatHandler()
        
        response = "Final Answer: This is the actual answer."
        sanitized = handler.sanitize_response(response)
        
        assert sanitized == "This is the actual answer."


# =============================================================================
# Run with: pytest tests/test_integration.py -v
# =============================================================================
