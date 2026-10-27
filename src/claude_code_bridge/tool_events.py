"""Claude Agent SDK tool-result normalization.

Kept out of bridge orchestration so provider block handling and lossless result
serialization have one focused, testable boundary.
"""

from __future__ import annotations

from typing import Any

from agent_bridge.tool_results import ToolResultPayload


def normalize_tool_result(content: Any) -> ToolResultPayload:
    """Serialize an SDK ToolResultBlock without destructively slicing it."""
    return ToolResultPayload.from_producer(content, complete=True)
