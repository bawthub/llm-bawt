"""Local model bridge — runs local GPU inference out of the main app process.

Reads ``chat.send`` commands (and the ``chat.abort`` RPC) from the shared
Redis ``agent:commands`` stream, loads the requested local model (gguf via
llama-cpp-python, or vLLM), streams the generated text back over the
per-run Redis response stream as ``AgentEvent``s, and serializes all CUDA
work onto a single-worker thread pool.

The main app sees identical events regardless of which agent bridge handled
the request.

Architecture (TASK-276): isolating local inference in a separate process
means a CUDA abort() — which calls abort() on the whole process — can only
kill this bridge, never the main app / MCP server / live agent sessions.
"""

from .bridge import LocalModelBridge

__all__ = ["LocalModelBridge"]
