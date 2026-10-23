"""Claude Code agent backend.

Uses the shared agent-bridge Redis command/event protocol, but routes
to the claude-code-bridge (Agent SDK) instead of the openclaw-bridge
(WebSocket gateway).

Each user gets their own Claude conversation thread — the session key
is scoped by user_id (e.g. ``claude-code:main:nick``).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from claude_code_bridge.tool_policy import (
    CLAUDE_CODE_DISALLOWED_TOOLS_KEY,
    configured_disallowed_tools,
)

from .agent_bridge import AgentBridgeBackend


class ClaudeCodeBackend(AgentBridgeBackend):
    """Agent backend for Claude Code via the Agent SDK bridge.

    Inherits the full Redis command/event protocol from AgentBridgeBackend.
    Session keys are user-scoped so each user gets their own conversation.

    Configuration keys (``agent_backend_config`` in bot profile):
        session_key: Base session key (default: "claude-code:main")
        timeout_seconds: Max wait for response (default: 120)
        model: Claude model SDK ID — REQUIRED, no fallback.
    """

    name = "claude-code"

    # The Claude Code CLI records the system prompt on a session's first
    # request and reuses it on every resume until compaction (TASK-1318).
    pins_system_prompt = True

    @staticmethod
    def _require_model(config: dict) -> str:
        # Hard-require an explicit model. No silent fallback to Sonnet (or
        # any other model) — if the bot's agent_backend_config is missing
        # ``model``, fail loudly with a message that names the bot so the
        # operator can fix the config.
        model = str(config.get("model") or "").strip()
        if not model:
            bot_id = str(config.get("bot_id") or "?").strip() or "?"
            raise ValueError(
                f"Claude Code backend: bot={bot_id} has no 'model' in "
                f"agent_backend_config. Set it on the bot's profile — the "
                f"bridge will not fall back to a default."
            )
        return model

    def compose_system_prompt(self, system_prompt: str, config: dict) -> str:
        model = self._require_model(config)
        # Inject the resolved model into the system prompt so the agent has
        # a ground-truth reference instead of confabulating from environment
        # variables or training-time defaults.  The Claude Agent SDK does
        # not surface its own model id inside the agent's context window —
        # the bridge logs `Actual model: ...` to itself, but the agent
        # never sees that line.  Without this block, "what model are you"
        # gets a guess (often wrong) instead of the truth.
        sp = (system_prompt or "").rstrip()
        provider_system_prompt = str(
            config.get("provider_system_prompt") or ""
        ).strip()
        endpoint_id = config.get("endpoint_id")
        harness = str(config.get("harness") or "").strip() or None
        if endpoint_id is not None and harness:
            try:
                current = self._config.resolve_model(
                    int(endpoint_id), harness=harness, default={}
                )
                resolved_prompt = current.get("provider_system_prompt")
                provider_system_prompt = (
                    resolved_prompt.strip()
                    if isinstance(resolved_prompt, str)
                    else ""
                )
            except Exception:
                # Keep the construction-time value if a transient catalog reload
                # fails; the turn still has a valid, previously resolved config.
                pass
        # TASK-490: body comes from the registry (agents.runtime_context_template),
        # bot-overridable, with the constant as the default/fallback.
        from ..prompt_registry import RUNTIME_CONTEXT_TEMPLATE, get_prompt_resolver
        try:
            resolved = get_prompt_resolver().resolve("agents.runtime_context_template")
            template = resolved.body if (resolved and resolved.body) else RUNTIME_CONTEXT_TEMPLATE
            model_block = template.format(model=model)
        except Exception:
            model_block = RUNTIME_CONTEXT_TEMPLATE.format(model=model)
        augmented_sp = f"{model_block}\n\n{sp}" if sp else model_block
        if provider_system_prompt:
            augmented_sp = f"{augmented_sp}\n\n{provider_system_prompt}"
        return augmented_sp

    def stream_raw(
        self,
        prompt: str,
        config: dict,
        attachments: list | None = None,
        trigger_message_id: str | None = None,
    ) -> Iterator[str | dict[str, Any]]:
        self._require_model(config)
        # TASK-1318: a pinned session prompt is already the composed, exact
        # bytes for this session. Re-composing would let a catalog or registry
        # edit change the prompt mid-session.
        if config.get("system_prompt_pinned"):
            augmented_sp = str(config.get("system_prompt") or "")
        else:
            augmented_sp = self.compose_system_prompt(
                config.get("system_prompt") or "", config,
            )

        # Resolve this system-wide policy at dispatch time so DB edits apply to
        # the next command without teaching the long-running bridge to poll the
        # app. A resolver with no bot deliberately ignores bot-scoped rows.
        from ..runtime_setting_resolution import resolve_global_runtime_setting

        configured = resolve_global_runtime_setting(
            self._config,
            CLAUDE_CODE_DISALLOWED_TOOLS_KEY,
        )
        disallowed_tools = configured_disallowed_tools(configured)
        augmented_config = {
            **config,
            "system_prompt": augmented_sp,
            "disallowed_tools": disallowed_tools,
        }
        return super().stream_raw(
            prompt,
            augmented_config,
            attachments=attachments,
            trigger_message_id=trigger_message_id,
        )

    def _resolve_session_key(self, config: dict) -> str:
        # Route by bot + user so each user gets an independent Claude session
        # for a given bot. The session_key in agent_backend_config is an SDK
        # session UUID written by the bridge and should not be used directly
        # as a routing key.
        bot_id = str(config.get("bot_id") or "main").strip() or "main"
        user_id = str(config.get("user_id") or "default").strip() or "default"
        return f"{bot_id}:{user_id}"
