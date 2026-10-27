"""Codex agent backend.

Uses the shared agent-bridge Redis command/event protocol, but routes
to ``codex-bridge`` (OpenAI Codex SDK / ChatGPT-mode OAuth) instead of
the openclaw-bridge (WebSocket gateway) or claude-code-bridge (Claude
Agent SDK).

Each bot+user pair gets its own Codex thread — the persisted ``session_key``
in ``agent_backend_config`` is the Codex ``thread_id`` written by the
bridge after thread/started fires for the first turn.
"""

from __future__ import annotations

from .agent_bridge import AgentBridgeBackend


class CodexBackend(AgentBridgeBackend):
    """Agent backend for OpenAI Codex via the codex-bridge.

    Inherits the full Redis command/event protocol from AgentBridgeBackend.
    Session keys are bot+user scoped so each user gets an independent
    Codex thread for a given bot.

    Configuration keys (``agent_backend_config`` in bot profile):
        session_key: Codex ``thread_id`` (managed by the bridge — do not
                     set manually; cleared by ``/new`` or model change)
        model:       Codex model id (default ``gpt-5.4``)
        timeout_seconds: Max wait for response (default: 600)
        codex_transport: ``exec`` (default) or ``app_server``. Only an
                     app-server run can be steered mid-turn (TASK-1243).
    """

    name = "codex"

    # TASK-1318: the bridge pastes the prompt into the thread, so thread
    # history carries it. Pin it per session and paste it only when it takes
    # effect, instead of appending a copy to every turn.
    pins_system_prompt = True

    def steer_capabilities(self, config: dict) -> frozenset[str]:
        # TASK-1243: app-server accepts ``turn/steer`` into the running turn,
        # which is a ``next`` steer. It has no queued-command handle, so no
        # ``now``, escalate or cancel. ``codex exec`` cannot steer at all.
        from ..steer_capabilities import NEXT, NO_STEER_CAPS

        if str(config.get("codex_transport") or "").strip() == "app_server":
            return frozenset({NEXT})
        return NO_STEER_CAPS

    def system_prompt_delivery(self, is_boundary: bool) -> str | None:
        # "now": the pin moved this turn, paste it even on a resumed thread.
        # "cold_start": paste only if the bridge opens a fresh thread (its own
        # reset or fresh-session retry included); resumed threads already hold it.
        return "now" if is_boundary else "cold_start"

    def _resolve_session_key(self, config: dict) -> str:
        # Route by bot + user so each user gets an independent Codex thread
        # for a given bot. The session_key in agent_backend_config is the
        # Codex thread_id written by the bridge and should not be used
        # directly as a routing key — match the claude-code semantics.
        bot_id = str(config.get("bot_id") or "main").strip() or "main"
        user_id = str(config.get("user_id") or "default").strip() or "default"
        return f"{bot_id}:{user_id}"
