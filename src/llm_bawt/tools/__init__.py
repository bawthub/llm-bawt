"""Tool calling support for LLMs.

Local tools: search (web/news), home (Home Assistant), model (list/switch)
and time. Memory, message history, profiles and bot traits come from BawtHub
MCP through deferred discovery (``tool_search``), as do tools from connected
MCP servers.
"""

from .definitions import (
    # Tool instances
    SEARCH_TOOL,
    HOME_TOOL,
    MODEL_TOOL,
    TIME_TOOL,
    # Tool categories
    CORE_TOOLS,
    SEARCH_TOOLS,
    HOME_TOOLS,
    MODEL_TOOLS,
    ALL_TOOLS,
    # Legacy mapping
    LEGACY_TOOL_MAP,
    normalize_legacy_tool_call,
    # Functions
    get_tools_prompt,
    get_tools_list,
    get_tool_by_name,
    # Types
    Tool,
    ToolParameter,
)
from .parser import ToolCall, format_tool_result, strip_tool_result_tags
from .executor import ToolExecutor
from .loop import ToolLoop, query_with_tools
from .streaming import stream_with_tools

__all__ = [
    # Tool instances
    "SEARCH_TOOL",
    "HOME_TOOL",
    "MODEL_TOOL",
    "TIME_TOOL",
    # Tool categories
    "CORE_TOOLS",
    "SEARCH_TOOLS",
    "HOME_TOOLS",
    "MODEL_TOOLS",
    "ALL_TOOLS",
    # Legacy mapping
    "LEGACY_TOOL_MAP",
    "normalize_legacy_tool_call",
    # Functions
    "get_tools_prompt",
    "get_tools_list",
    "get_tool_by_name",
    # Types
    "Tool",
    "ToolParameter",
    # Parser/result formatting
    "ToolCall",
    "format_tool_result",
    "strip_tool_result_tags",
    # Executor
    "ToolExecutor",
    # Loop
    "ToolLoop",
    "query_with_tools",
    # Streaming
    "stream_with_tools",
]
