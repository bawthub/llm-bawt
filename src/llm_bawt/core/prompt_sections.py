"""Prompt section renderers for BaseLLMBawt (TASK-1249 A1).

Each ``_sec_<name>`` method renders one section declared in
``prompt_manifest.PROMPT_MANIFEST`` (referenced there by name) and performs its
own gated ``builder.add_section``. ``BaseLLMBawt._walk_manifest`` drives them.
They live here, split out of ``base.py`` without behavior change, so the
orchestrator stays under the monolith gate as new sections are added.
"""

import logging

from rich.console import Console

from ..tools import get_tools_prompt
from .prompt_builder import GLOBAL_SYSTEM_PROMPT, PromptBuilder, SectionPosition

logger = logging.getLogger(__name__)
console = Console()


class PromptSectionsMixin:
    """Section renderers; mixed into ``BaseLLMBawt``, which supplies state."""

    # --- stable section renderers (walked by _walk_manifest, STABLE) --------
    def _sec_user_context(self, builder: PromptBuilder, prompt: str) -> None:
        """Section 1: User profile context."""
        if self._db_available and self.profile_manager:
            try:
                user_context = self.profile_manager.get_user_profile_summary(self.user_id)
                if user_context:
                    builder.add_section(
                        "user_context",
                        f"## About the User\n{user_context}",
                        position=SectionPosition.USER_CONTEXT,
                        metadata={"source": "profile_db:user", "gate": "db_available & profile"},
                    )
                    if self.verbose:
                        console.print(f"[dim]─── User Profile ({self.user_id}) ───[/dim]")
                        console.print(f"[cyan]{user_context}[/cyan]")
                        console.print(f"[dim]{'─' * 30}[/dim]")
            except Exception as e:
                logger.warning(f"Failed to load user profile: {e}")

    def _sec_bot_traits(self, builder: PromptBuilder, prompt: str) -> None:
        """Section 2: Bot personality traits."""
        if self._db_available and self.profile_manager:
            try:
                bot_context = self.profile_manager.get_bot_profile_summary(self.bot_id)
                if bot_context:
                    builder.add_section(
                        "bot_traits",
                        f"## Your Developed Traits\n{bot_context}",
                        position=SectionPosition.BOT_TRAITS,
                        metadata={"source": "profile_db:bot", "gate": "db_available & profile"},
                    )
                    if self.verbose:
                        console.print(f"[dim]─── Bot Profile ({self.bot_id}) ───[/dim]")
                        console.print(f"[magenta]{bot_context}[/magenta]")
                        console.print(f"[dim]{'─' * 30}[/dim]")
            except Exception as e:
                logger.warning(f"Failed to load bot profile: {e}")

    def _sec_base_prompt(self, builder: PromptBuilder, prompt: str) -> None:
        """Section 3: Base bot prompt (or active persona override, TASK-477)."""
        base_prompt = self._resolve_base_prompt()
        _base_src = (
            "persona_override" if getattr(self.bot, "prompt_override_id", None)
            else "bot.system_prompt"
        )
        if base_prompt:
            builder.add_section(
                "base_prompt",
                base_prompt,
                position=SectionPosition.BASE_PROMPT,
                metadata={"source": _base_src, "gate": "always"},
            )

    def _sec_global_instructions(self, builder: PromptBuilder, prompt: str) -> None:
        """Section 6: Global behavioral instructions (memory-enabled bots only).

        TASK-490: body comes from the registry (chat.global_recall_guidance,
        bot-overridable); the GLOBAL_SYSTEM_PROMPT constant is the default and
        the safety fallback.
        """
        if self.bot.requires_memory:
            resolved = self.config_resolver.resolve_body(
                "chat.global_recall_guidance", scope_type="bot", scope_id=self.bot_id
            )
            body = (resolved.body if resolved else None) or GLOBAL_SYSTEM_PROMPT
            src = f"registry:{resolved.source}" if resolved else "code_default:GLOBAL_SYSTEM_PROMPT"
            builder.add_section(
                "global_instructions",
                body,
                position=SectionPosition.GLOBAL_INSTRUCTIONS,
                metadata={"source": src, "gate": "requires_memory"},
            )

    # --- per-turn section renderers (walked by _walk_manifest, PER_TURN) -----
    def _sec_tools(self, builder: PromptBuilder, prompt: str) -> None:
        """Chat-model tool instructions (native agent harnesses own theirs)."""
        if self.tool_format != "none":
            if self.bot.uses_tools:
                tool_definitions = self._get_tool_definitions()
                if tool_definitions:
                    tools_prompt = get_tools_prompt(
                        tools=tool_definitions,
                        tool_format=self.tool_format,
                    )
                    builder.add_section(
                        "tools",
                        tools_prompt,
                        position=SectionPosition.TOOLS,
                        metadata={"source": "generated:tools", "gate": "uses_tools"},
                    )
            elif self.memory:
                # Non-tool memory bots get a read-only memory search tool
                from ..tools.definitions import MEMORY_TOOL
                tools_prompt = get_tools_prompt(
                    tools=[MEMORY_TOOL],
                    tool_format=self.tool_format,
                )
                builder.add_section(
                    "tools",
                    tools_prompt,
                    position=SectionPosition.TOOLS,
                    metadata={"source": "generated:memory_tool", "gate": "memory_readonly"},
                )

    def _sec_client_context(self, builder: PromptBuilder, prompt: str) -> None:
        """Client-supplied system context (e.g. HA device list)."""
        if self._client_system_context:
            builder.add_section(
                "client_context",
                f"## Client Context\n{self._client_system_context}",
                position=SectionPosition.CLIENT_CONTEXT,
                metadata={"source": "client_runtime", "gate": "client_system_context"},
            )

    def _sec_cold_start_memory(self, builder: PromptBuilder, prompt: str) -> None:
        """Cold-start memory priming: inject top memories when history is thin.

        Skip for HA-mode — device commands don't need semantic memories.
        """
        if self.memory and not self._ha_mode:
            history_count = len(self.history_manager.messages)
            if history_count <= 3:
                cold_start_context = self._retrieve_cold_start_memories(prompt)
                if cold_start_context:
                    builder.add_section(
                        "cold_start_memory",
                        cold_start_context,
                        position=SectionPosition.MEMORY_CONTEXT,
                        metadata={"source": "memory_search", "gate": "cold_start(history<=3)"},
                    )

    def _sec_tts_output(self, builder: PromptBuilder, prompt: str) -> None:
        """TTS output instructions (chat bots only — manifest applies_to gates
        out agent backends, which use a per-turn user-message voice prefix
        instead; injecting TTS into both system prompt AND user message would
        double-dose the constraint and cripple tool chaining)."""
        if self._tts_mode:
            resolved = self.config_resolver.resolve_body("chat.tts_output_instructions")
            tts_body = resolved.body if resolved else None
            if tts_body:
                builder.add_section(
                    "tts_output",
                    tts_body,
                    position=SectionPosition.CUSTOM,
                    metadata={
                        "source": f"registry:{resolved.source}",
                        "gate": "tts_mode & not agent_backend",
                    },
                )

    def _sec_agent_global_prompt(self, builder: PromptBuilder, prompt: str) -> None:
        """Agent global prompt (agent backends only — manifest applies_to gates
        out chat bots). Opt-in per bot via the `agent_global_prompt_enabled`
        runtime setting. Rides the cacheable prefix; bot-scoped resolve falls
        back to the global default; an empty body injects nothing."""
        if bool(self._resolve_setting("agent_global_prompt_enabled", False)):
            resolved = self.config_resolver.resolve_body(
                "agents.global_prompt", scope_type="bot", scope_id=self.bot_id
            )
            agent_global_body = resolved.body if resolved else None
            if agent_global_body:
                builder.add_section(
                    "agent_global_prompt",
                    agent_global_body,
                    position=SectionPosition.GLOBAL_INSTRUCTIONS,
                    metadata={
                        "source": f"registry:{resolved.source}",
                        "gate": "agent_backend & agent_global_prompt_enabled",
                    },
                )

    def _sec_tag_context(self, builder: PromptBuilder, prompt: str) -> None:
        """TASK-1249 bot-tag section (agent backends only): this bot's role-tag
        instructions and the family roster, rendered into the operator's
        ``agents.tag_context`` template. The default body is empty, so nothing
        is read or injected until an operator writes one. A template or data
        failure skips the section; it never fails the prompt build."""
        from ..bot_tags.context import TEMPLATE_KEY, load_tag_context

        try:
            resolved = self.config_resolver.resolve_body(
                TEMPLATE_KEY, scope_type="bot", scope_id=self.bot_id
            )
        except Exception:
            logger.warning("tag context: template %s unreadable; section skipped",
                           TEMPLATE_KEY, exc_info=True)
            return
        template = resolved.body if resolved else None
        if not template or not template.strip():
            return
        try:
            from ..utils.db import get_shared_engine

            def instructions(key: str) -> str | None:
                found = self.config_resolver.resolve_body(key)
                return found.body if found else None

            context = load_tag_context(
                get_shared_engine(self.config), self.config, self.bot_id, instructions
            )
            body = template.format(**context.variables())
        except (KeyError, IndexError, ValueError) as exc:
            logger.warning("tag context: template %s does not render: %s", TEMPLATE_KEY, exc)
            return
        except Exception:
            logger.warning("tag context: facts unavailable; section skipped", exc_info=True)
            return
        if not body.strip():
            return
        metadata = {
            "source": f"registry:{resolved.source}",
            "gate": "agent_backend & agents.tag_context template",
        }
        if context.flags:
            metadata["provenance"] = list(context.flags)
        builder.add_section(
            "tag_context",
            body,
            position=SectionPosition.GLOBAL_INSTRUCTIONS,
            metadata=metadata,
        )

    def _sec_injected_skills(self, builder: PromptBuilder, prompt: str) -> None:
        """TASK-1309 injected skills (agent backends only): the SKILL.md bodies
        assigned to this bot in 'inject' mode, admitted under the token caps.
        Fails closed: any error omits the section; it never fails the build.
        The resolution is kept as ``_skill_resolution`` so the turn pins the
        same manifest this section rendered (None when it failed).
        ``_injected_skill_refs`` names the skills whose bodies were added; it
        stays empty whenever the section is not (TASK-1691)."""
        from ..skills.prompts import ENTRY_KEY, PREAMBLE_KEY, render_section
        from ..skills.resolver import INJECT
        from ..skills.service import SkillService
        from ..utils.db import get_shared_engine

        self._skill_resolution = None
        self._injected_skill_refs = ()
        try:
            engine = get_shared_engine(self.config)
            resolution = SkillService.from_config(engine, self.config).resolve_bot(self.bot_id)
            self._skill_resolution = resolution
            injected = resolution.injected if resolution is not None else ()
            if not injected:
                return
            preamble = self.config_resolver.resolve_body(
                PREAMBLE_KEY, scope_type="bot", scope_id=self.bot_id)
            entry = self.config_resolver.resolve_body(
                ENTRY_KEY, scope_type="bot", scope_id=self.bot_id)
            body = render_section(
                preamble.body if preamble else "", entry.body if entry else "",
                [(e.ref, e.base_dir, e.body_text) for e in injected])
            manifest = resolution.manifest()
        except Exception:
            logger.warning("injected skills unavailable; section skipped", exc_info=True)
            return
        if not body.strip():
            return
        builder.add_section(
            "injected_skills",
            body,
            position=SectionPosition.GLOBAL_INSTRUCTIONS,
            metadata={
                "source": "skills.resolve",
                "gate": "agent_backend & inject exceptions",
                "refs": [e.ref for e in injected],
                "tokens": resolution.injected_tokens,
                "manifest_sha": manifest["manifest_sha"] if manifest else None,
                # Injected assignments left out of this render, and why.
                "skipped": [{"ref": e.ref, "reason": e.reason} for e in resolution.entries
                            if e.mode == INJECT and not e.applied],
            },
        )
        self._injected_skill_refs = tuple(e.ref for e in injected)

    def _sec_working_memory(self, builder: PromptBuilder, prompt: str) -> None:
        """TASK-1281 working-memory section, rendered per turn by the dispatch path."""
        if self._working_memory_section:
            builder.add_section(
                "working_memory",
                self._working_memory_section,
                position=SectionPosition.WORKING_MEMORY,
                metadata={"source": "working_memory.render", "gate": "agent_backend & working_memory policy"},
            )
