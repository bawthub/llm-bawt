"""Shared tool-call records and result formatting."""

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    """A normalized tool call ready for execution."""

    name: str
    arguments: dict[str, Any]
    raw_text: str

    def __str__(self) -> str:
        return f"ToolCall({self.name}, {self.arguments})"


_TOOL_RESULT_RE = re.compile(
    r'^<tool_result\b[^>]*>\n?(.*?)\n?</tool_result>',
    re.DOTALL,
)


_TOOL_RESULT_ERROR_RE = re.compile(r'^<tool_result\b[^>]*\bstatus="error"')


def tool_result_is_error(text: Any) -> bool:
    """Whether an executor result is a ``format_tool_result`` error envelope.

    The status lives only in the wrapper tag, so check before any caller
    strips it with :func:`strip_tool_result_tags`.
    """
    return isinstance(text, str) and _TOOL_RESULT_ERROR_RE.match(text) is not None


def strip_tool_result_tags(text: str) -> str:
    """Remove ``<tool_result>`` wrapper tags, returning the inner content.

    Used by the native tool-calling path where results go into structured
    ``function_call_output`` items and the XML wrapper is wasted tokens.
    Also strips the ``[IMPORTANT: ...]`` error suffix injected for text-based formats.
    """
    m = _TOOL_RESULT_RE.match(text)
    if m:
        inner = m.group(1)
        # Drop the text-format error instruction suffix if present
        remainder = text[m.end():]
        if remainder.startswith("\n"):
            remainder = remainder[1:]
        if remainder.startswith("[IMPORTANT:"):
            remainder = ""
        return (inner + remainder).strip()
    return text


def format_tool_result(tool_name: str, result: Any, error: str | None = None) -> str:
    """Format a tool result for injection back into the conversation.
    
    Args:
        tool_name: Name of the tool that was called.
        result: The result from the tool execution.
        error: Optional error message if tool failed.
        
    Returns:
        Formatted string to inject as a message.
    """
    if error:
        return (
            f"<tool_result name=\"{tool_name}\" status=\"error\">\n"
            f"{error}\n"
            f"</tool_result>\n"
            f"[IMPORTANT: The tool failed. Report this error to the user accurately.]"
        )
    
    # Format result nicely
    if isinstance(result, (dict, list)):
        result_str = json.dumps(result, indent=2, default=str)
    else:
        result_str = str(result)
    
    return (
        f"<tool_result name=\"{tool_name}\" status=\"success\">\n"
        f"{result_str}\n"
        f"</tool_result>"
    )
