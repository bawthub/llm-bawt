"""Declared prompt manifest (TASK-489).

The system-prompt shape used to be imperative logic split across two methods
(``_init_system_prompt`` for the stable, cacheable base and
``_assemble_system_builder`` for the per-turn sections). This module makes the
shape DATA: one ordered list of :class:`SectionSpec` that a single walk function
iterates. Each section's actual rendering logic lives in a ``_sec_<name>``
method on ``BaseLLMBawt`` (defined in ``prompt_sections.py``, referenced by
name here) — so this file is the one
place to read the entire prompt shape top-to-bottom, while the heterogeneous
per-section logic stays where it can reach ``self``.

Two stages, matching the cache architecture (TASK-288):
- ``STABLE``   sections ride ``self._prompt_builder`` — the cached base prefix.
- ``PER_TURN`` sections ride a ``.copy()`` per turn — never mutate the cache.

``applies_to`` DECLARES bot-type applicability (chat vs agent) instead of
burying it in inline ``_is_agent_backend`` conditionals: e.g. tts_output is
chat-only, agent_global_prompt is agent-only.
"""

from dataclasses import dataclass

STABLE = "stable"
PER_TURN = "per_turn"

BOT_TYPES_ALL = ("chat", "agent")


@dataclass(frozen=True)
class SectionSpec:
    """One prompt section, declared as data.

    Attributes:
        name: Section identifier (matches the PromptBuilder section name).
        stage: STABLE (cached base) or PER_TURN (rides the per-turn copy).
        method: Name of the ``BaseLLMBawt`` method that renders it. The method
            takes ``(builder, prompt)`` and performs the gated ``add_section``.
        applies_to: Bot types this section applies to. The walk skips sections
            that don't apply to the current bot's type — declarative gating.
    """

    name: str
    stage: str
    method: str
    applies_to: tuple[str, ...] = BOT_TYPES_ALL


# THE prompt shape, in position order (top of prompt → bottom). This ordered
# list is the whole map; the builder's integer positions enforce final order.
PROMPT_MANIFEST: tuple[SectionSpec, ...] = (
    # --- stable base (cached prefix) ---
    SectionSpec("user_context", STABLE, "_sec_user_context"),
    SectionSpec("bot_traits", STABLE, "_sec_bot_traits"),
    SectionSpec("base_prompt", STABLE, "_sec_base_prompt"),
    SectionSpec("global_instructions", STABLE, "_sec_global_instructions"),
    # --- per-turn (rides the copy) ---
    # Native agent harnesses (Claude Code, Codex) own their tool
    # schemas and calling protocol. App-side tool prompts are chat-only; injecting
    # them into an agent system prompt duplicates and contradicts the harness.
    SectionSpec("tools", PER_TURN, "_sec_tools", applies_to=("chat",)),
    SectionSpec("client_context", PER_TURN, "_sec_client_context"),
    SectionSpec("cold_start_memory", PER_TURN, "_sec_cold_start_memory"),
    SectionSpec("tts_output", PER_TURN, "_sec_tts_output", applies_to=("chat",)),
    SectionSpec("agent_global_prompt", PER_TURN, "_sec_agent_global_prompt", applies_to=("agent",)),
    # TASK-1249: same position as agent_global_prompt; walk order puts it after.
    SectionSpec("tag_context", PER_TURN, "_sec_tag_context", applies_to=("agent",)),
    # TASK-1309: same position again; walk order puts it after tag_context.
    SectionSpec("injected_skills", PER_TURN, "_sec_injected_skills", applies_to=("agent",)),
    SectionSpec("working_memory", PER_TURN, "_sec_working_memory", applies_to=("agent",)),
)
