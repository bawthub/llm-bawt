"""BawtHub MCP server (formerly named 'llm-bawt-memory').

Exposes memory, messages, fact-extraction, inter-bot, and agent task tools
via Model Context Protocol (FastMCP). The server identity is now ``bawthub``
to reflect that it serves the broader llm-bawt/BawtHub platform, not just
memory. Also provides MemoryClient for unified in-process or remote
(HTTP JSON-RPC) access.
"""

from typing import TYPE_CHECKING

# Client imports are always available (no mcp dependency)
from .client import MemoryClient, MemoryResult, MessageResult, get_memory_client

# Server imports require mcp package - lazy load
if TYPE_CHECKING:
    from .server import mcp, run_server

__all__ = [
    "mcp",
    "run_server",
    "MemoryClient",
    "MemoryResult",
    "MessageResult",
    "get_memory_client",
]


def __getattr__(name: str):
    """Lazy import server components that require mcp package."""
    if name in ("mcp", "run_server"):
        try:
            from .server import mcp, run_server
            return mcp if name == "mcp" else run_server
        except ImportError as e:
            raise ImportError(
                f"The '{name}' component requires the 'mcp' package. "
                "Install it with: uv sync --extra mcp"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
