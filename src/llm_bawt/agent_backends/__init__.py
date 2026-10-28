"""Agent backend plugin system for llm-bawt.

Agent backends allow bots to delegate chat handling to external agent systems
(e.g. Claude Code, Codex) instead of using the built-in LLM pipeline.

Third-party packages register backends via entry points::

    [project.entry-points."llm_bawt.agent_backends"]
    myagent = "mypackage.agents:MyAgentBackend"
"""

from .base import AgentBackend
from .registry import get_backend, list_backends, register_backend

__all__ = [
    "AgentBackend",
    "get_backend",
    "list_backends",
    "register_backend",
]
