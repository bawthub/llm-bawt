"""Streaming tool loop for LLM conversations.

Handles tool calling while maintaining streaming output to the client.
Uses format handlers with stop sequences to properly detect tool calls.
"""

import logging
from typing import TYPE_CHECKING, Iterator, Callable

from .executor import ToolExecutor
from .formats import ToolFormat, get_format_handler
from ..adapters import ModelAdapter, DefaultAdapter

if TYPE_CHECKING:
    from ..models.message import Message
    from ..mcp_server.client import MemoryClient
    from ..profiles import ProfileManager
    from ..search.base import SearchClient
    from ..integrations.ha_mcp.client import HomeAssistantMCPClient, HomeAssistantNativeClient
    from ..core.model_lifecycle import ModelLifecycleManager
    from ..utils.config import Config
    from ..utils.history import HistoryManager

# Use service logger if available, otherwise standard logging
try:
    from ..service.logging import ServiceLogger
    log = ServiceLogger(__name__)
except ImportError:
    log = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 20

# How many characters to buffer before deciding if response is tool call or text
DECISION_THRESHOLD = 80


def _looks_like_tool_call_start(text: str) -> bool:
    """Check if text looks like the start of a tool call in any format."""
    stripped = text.strip().lower()
    if not stripped:
        return False
    
    # ReAct format markers - check both directions for partial matches
    tool_prefixes = ("thought:", "action:", "# tool:", "tool:")
    for prefix in tool_prefixes:
        # Either text starts with prefix, or prefix starts with text (partial match)
        if stripped.startswith(prefix) or prefix.startswith(stripped):
            # Debug log
            try:
                from ..service.logging import ServiceLogger
                log = ServiceLogger(__name__)
                log.debug(f"_looks_like_tool_call_start: '{stripped[:20]}' matches '{prefix}' -> True")
            except Exception:
                pass
            return True
    
    # JSON code block with tool call
    if stripped.startswith("```") and "tool" in stripped:
        return True
    return False


def _looks_like_regular_text(text: str) -> bool:
    """Check if text clearly looks like regular prose."""
    stripped = text.strip()
    if not stripped:
        return False
    
    # Need minimum text to decide - tool calls start with specific patterns
    # that might not be complete yet in short text
    if len(stripped) < 10:
        return False

    first_char = stripped[0]
    # Starts with letter or common punctuation = regular text
    if first_char.isalpha():
        # But not if it's "Thought" or "Action" starting (or partial match)
        first_word = stripped.split()[0].lower() if stripped.split() else ""
        # Check for partial matches of tool call patterns
        tool_prefixes = ("thought", "action", "tool", "#")
        if any(first_word.startswith(p) or p.startswith(first_word) for p in tool_prefixes):
            return False
        return True
    if first_char in '!?.,;:()[]"\'—–-':
        return True
    # Emoji or unicode that's not a tool marker
    if first_char not in '{<`#':
        return True
    return False


# Patterns that indicate ReAct junk at end of response
_REACT_LINE_MARKERS = (
    "thought:",
    "action:",
    "action input:",
    "observation:",
    "final answer:",
)


def _contains_react_marker(text: str) -> bool:
    """Check if text contains a ReAct marker that should stop streaming."""
    lower = text.lower()
    for marker in _REACT_LINE_MARKERS:
        if marker in lower:
            return True
    return False


def stream_with_tools(
    messages: list["Message"],
    stream_fn: Callable[[list["Message"], list[str] | None], Iterator[str]],
    memory_client: "MemoryClient | None" = None,
    profile_manager: "ProfileManager | None" = None,
    search_client: "SearchClient | None" = None,
    home_client: "HomeAssistantMCPClient | None" = None,
    ha_native_client: "HomeAssistantNativeClient | None" = None,
    model_lifecycle: "ModelLifecycleManager | None" = None,
    config: "Config | None" = None,
    user_id: str = "",
    bot_id: str = "",
    max_iterations: int | None = None,
    tool_format: ToolFormat | str = ToolFormat.REACT,
    adapter: "ModelAdapter | None" = None,
    history_manager: "HistoryManager | None" = None,
    tool_call_details: list[dict] | None = None,
) -> Iterator[str]:
    """Stream LLM response with tool calling support.

    Uses format handlers with stop sequences to reliably detect tool calls.
    Stop sequences prevent the model from generating content after the tool call,
    which solves the "hallucinated observation" problem.

    Streaming strategy:
    1. Buffer initial tokens to detect if response is a tool call
    2. If it looks like regular text -> buffer everything, clean at end, then yield
    3. If it looks like a tool call -> buffer everything, execute, loop

    Args:
        messages: Initial conversation messages.
        stream_fn: Function to stream from LLM. Signature: (messages, stop_sequences) -> Iterator[str]
        memory_client: Memory client for tool execution.
        profile_manager: Profile manager for profile tools.
        search_client: Search client for web search tools.
        model_lifecycle: Model lifecycle manager for model switching tools.
        config: Application config (used for lazy search setup).
        user_id: Current user ID (required).
        bot_id: Current bot ID.
        max_iterations: Max tool iterations per turn.
        tool_format: Tool format to use (determines stop sequences and parsing).
        adapter: Model adapter for model-specific cleaning and stop sequences.
        history_manager: History manager for recall operations.
        tool_call_details: Optional mutable list to collect per-call tool details
            for downstream logging/persistence.

    Yields:
        Text chunks from the LLM response (cleaned by adapter and handler).
    """
    if not user_id:
        raise ValueError("user_id is required for stream_with_tools")
    from ..models.message import Message

    # Use config value if not explicitly provided (0 = unlimited, cap at reasonable max)
    if max_iterations is None:
        config_max = getattr(config, 'MAX_TOOL_CALLS_PER_TURN', DEFAULT_MAX_ITERATIONS) if config else DEFAULT_MAX_ITERATIONS
        max_iterations = config_max if config_max > 0 else 100

    if adapter is None:
        log.warning("No adapter provided, using DefaultAdapter")
        adapter = DefaultAdapter()
    else:
        log.debug(f"Using adapter: {adapter.name}")

    handler = get_format_handler(tool_format)
    if handler is None:
        raise ValueError(f"Tool streaming requires a valid tool format, got: {tool_format}")

    # Combine stop sequences from both handler and adapter
    handler_stops = handler.get_stop_sequences()
    adapter_stops = adapter.get_stop_sequences()
    stop_sequences = list(set(handler_stops + adapter_stops))

    # Log stop sequences compactly (repr escapes newlines, join keeps it single-line)
    log.info(f"🛑 Tool streaming with {len(stop_sequences)} stop sequences")

    executor = ToolExecutor(
        memory_client=memory_client,
        profile_manager=profile_manager,
        search_client=search_client,
        home_client=home_client,
        ha_native_client=ha_native_client,
        model_lifecycle=model_lifecycle,
        config=config,
        user_id=user_id,
        bot_id=bot_id,
        history_manager=history_manager,
    )
    current_messages = messages.copy()
    has_executed_tools = False  # Track if we've executed tools

    for iteration in range(1, max_iterations + 1):
        if iteration > 1:
            log.info(f"🔄 Tool loop iteration {iteration}/{max_iterations}")

        # After executing tools, don't use stop sequences - let model complete naturally
        current_stop_sequences = None if has_executed_tools else stop_sequences
        
        # Buffer for initial detection
        initial_buffer = ""
        initial_chunks: list[str] = []
        is_tool_call = False  # True if response looks like a tool call
        full_response = ""

        log.debug(f"Starting stream_fn with {len(current_messages)} messages")
        chunk_count = 0
        for chunk in stream_fn(current_messages, current_stop_sequences):
            chunk_count += 1
            if chunk_count == 1:
                log.debug(f"Got first chunk: {repr(chunk[:50]) if len(chunk) > 50 else repr(chunk)}")
            full_response += chunk
            
            if is_tool_call:
                # Already decided it's a tool call - keep buffering
                continue
            
            # Still deciding - buffer
            initial_buffer += chunk
            initial_chunks.append(chunk)
            stripped = initial_buffer.strip()

            if _looks_like_tool_call_start(stripped):
                # Definitely a tool call - keep buffering
                log.debug("Response looks like tool call - buffering")
                is_tool_call = True
                continue

            if _looks_like_regular_text(stripped):
                # Regular text - will buffer and clean at end
                # log.debug(f"Response looks like regular text - buffering for cleaning")
                is_tool_call = False
                continue

            # Haven't decided yet - check threshold
            if len(stripped) >= DECISION_THRESHOLD:
                # Buffered enough without tool markers - assume it's text
                log.debug("Decision threshold reached - treating as text")
                is_tool_call = False

        log.debug(f"Stream ended: {chunk_count} chunks, {len(full_response)} chars")

        # Parse for tool calls using format handler
        tool_calls, remaining_text = handler.parse_response(full_response)

        if not tool_calls:
            # No tool call found - clean and yield the final response
            log.debug("No tool calls found - cleaning and yielding response")
            log.debug(f"Raw response before cleaning ({len(full_response)} chars): {repr(full_response[:200])}")
            # Apply adapter cleaning FIRST, then format sanitization
            cleaned = adapter.clean_output(full_response)
            if cleaned != full_response:
                log.info(f"Adapter '{adapter.name}' cleaned response: {len(full_response)} -> {len(cleaned)} chars")
                log.debug(f"Cleaned response: {repr(cleaned[:200])}")
            sanitized = handler.sanitize_response(cleaned)

            # If sanitized is empty but original had tool-like content, the model
            # tried to call a tool but failed (invalid name, bad JSON, etc.)
            if not sanitized.strip() and ("action:" in full_response.lower()):
                log.warning(f"Model attempted invalid tool call: {repr(full_response)}")
                yield f"[Model tried to use a tool but the format was invalid. Raw output: {full_response}]"
                return

            yield sanitized
            return

        # Tool call detected - execute it
        tc = tool_calls[0]  # Process one tool at a time
        log.info(f"🔧 Calling tool: {tc.name} with args: {tc.arguments}")

        from .parser import ToolCall, tool_result_is_error
        tool_call =ToolCall(name=tc.name, arguments=tc.arguments, raw_text=tc.raw_text or "")
        tool_result = executor.execute(tool_call)
        has_executed_tools = True  # Mark that we've executed tools
        log.debug(f"Tool result: {len(tool_result)} chars")
        if tool_call_details is not None:
            tool_call_details.append(
                {
                    "tool": tc.name,
                    "parameters": tc.arguments if isinstance(tc.arguments, dict) else {"raw": tc.arguments},
                    "result": tool_result,
                    "is_error": tool_result_is_error(tool_result),
                    "iteration": iteration,
                }
            )

        # Format the result for the model
        formatted_result = handler.format_result(
            tc.name,
            tool_result,
            tool_call_id=getattr(tc, "tool_call_id", None),
        )

        # Build continuation messages - use only the tool call portion, not hallucinated continuation
        # tc.raw_text contains just the Action/Action Input block
        # We prepend any text before the tool call (e.g., "Thought: ...")
        tool_call_text = tc.raw_text or ""
        if not tool_call_text:
            # Fallback: extract just up to end of JSON object
            action_end = full_response.find("Action Input:")
            if action_end > 0:
                # Find the end of the JSON object after Action Input:
                brace_start = full_response.find("{", action_end)
                if brace_start > 0:
                    depth = 0
                    for i, ch in enumerate(full_response[brace_start:], brace_start):
                        if ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                tool_call_text = full_response[:i+1]
                                break
        if not tool_call_text:
            tool_call_text = full_response  # Last resort fallback
            
        current_messages.append(Message(role="assistant", content=tool_call_text))

        # For ReAct format, the observation goes in a user message
        # Append guidance to help model provide final answer instead of looping
        observation_content = ""
        if isinstance(formatted_result, dict):
            observation_content = formatted_result.get("content", str(tool_result))
        else:
            observation_content = str(formatted_result)
        
        # Add continuation prompt after observation to guide toward final answer
        # Use iteration == 1 since has_executed_tools is already True at this point
        if iteration == 1:
            observation_content += "\n\nBased on this information, provide your answer to the user's question."
        
        if isinstance(formatted_result, dict):
            current_messages.append(Message(
                role=formatted_result.get("role", "user"),
                content=observation_content,
                tool_call_id=formatted_result.get("tool_call_id"),
            ))
        else:
            current_messages.append(Message(role="user", content=observation_content))

        # Continue to next iteration for follow-up response

    log.warning(f"Max tool iterations ({max_iterations}) reached")
