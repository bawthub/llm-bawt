"""Abstract base class for agent backend plugins.

Agent backends handle chat requests by delegating to external agent systems
(e.g. OpenClaw, Claude Code, custom agents) instead of using llm-bawt's
internal LLM pipeline.

Third-party packages can implement this interface and register via entry points:

    [project.entry-points."llm_bawt.agent_backends"]
    myagent = "mypackage.agents:MyAgentBackend"
"""

from abc import ABC, abstractmethod


class AgentBackend(ABC):
    """Plugin interface for external agent chat systems."""

    name: str = "base"

    #: TASK-1318: the harness records the system prompt once per session and
    #: ignores a changed prompt on resume, so llm-bawt pins the session's
    #: prompt and sends those exact bytes every turn (service/system_prompt_pin).
    #: False keeps the backend's historical per-turn render.
    pins_system_prompt: bool = False

    def compose_system_prompt(self, system_prompt: str, config: dict) -> str:
        """Return the complete app-side system prompt for one turn.

        Backends that wrap llm-bawt's rendered prompt (runtime context,
        provider guidance) do it here, so a pinned session captures the exact
        bytes the bridge receives. The default sends the render unchanged.
        """
        return system_prompt

    def system_prompt_delivery(self, is_boundary: bool) -> str | None:
        """How the bridge should deliver a pinned prompt (TASK-1318).

        None: the bridge sends the prompt as-is every turn (the harness keeps
        its own session snapshot). Backends whose harness stores the prompt in
        thread history return a mode the bridge understands instead.
        """
        return None

    @abstractmethod
    async def chat(
        self,
        prompt: str,
        config: dict,
        stream: bool = False,
    ) -> str:
        """Handle a chat request and return the response text.

        Args:
            prompt: The user's message.
            config: Backend-specific configuration from bot's
                ``agent_backend_config`` in bot_profiles DB.
            stream: Whether streaming is requested (reserved for future use).

        Returns:
            The agent's response as a string.
        """
        ...

    async def health_check(self, config: dict) -> bool:
        """Check if the backend is reachable.

        Args:
            config: Backend-specific configuration.

        Returns:
            True if the backend is healthy and reachable.
        """
        return True
