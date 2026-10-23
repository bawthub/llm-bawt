"""Base class for LLM orchestration.

BaseLLMBawt provides the shared logic for both CLI (LLMBawt) and service
(ServiceLLMBawt) modes, including:
- Bot initialization
- Memory client initialization
- Profile management
- Pipeline execution

Subclasses override _initialize_client() for their specific client types.
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any

from rich.console import Console

from ..clients import LLMClient
from ..tools import get_tools_prompt, get_tools_list, query_with_tools
from ..utils.paths import resolve_log_dir
from ..utils.history import HistoryManager, Message, scope_flags
from .prompt_builder import GLOBAL_SYSTEM_PROMPT, PromptBuilder, SectionPosition
from .prompt_manifest import PROMPT_MANIFEST, STABLE, PER_TURN
from .initialization import OrchestratorInitializationMixin

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
console = Console()


# Slash commands (e.g. /new) are app-level signals consumed by the bridges
# before any agent sees them. Match leading slash + alpha word so we can skip
# the agent user-prefix wrap, which otherwise breaks the bridge's literal
# startswith("/new") check. Conservative: only matches /<letter><word-chars>,
# so a stray URL path-shaped user message is also passed through unwrapped
# (acceptable — those are rare and the LLM handles them fine without prefix).
_SLASH_COMMAND_RE = re.compile(r"^\s*/[a-zA-Z][\w-]*\b")




class BaseLLMBawt(OrchestratorInitializationMixin, ABC):
    """Base class for LLM orchestration with memory and tools.
    
    Provides common initialization and query logic. Subclasses implement
    _initialize_client() for their specific model backends.
    """
    
    @abstractmethod
    def _initialize_client(self) -> LLMClient:
        """Initialize the LLM client. Implemented by subclasses."""
        pass
    
    def _resolve_base_prompt(self) -> str:
        """Return the effective base prompt for this bot (TASK-477).

        If the bot has an active persona override (``prompt_override_id``) that
        resolves to a persona template, its body REPLACES the bot's stored
        ``system_prompt``. If the override id is missing/unresolvable, fall back
        to ``system_prompt`` and log — a stale override must never break a turn.
        """
        override_id = getattr(self.bot, "prompt_override_id", None)
        if not override_id:
            return self.bot.system_prompt

        try:
            from ..prompt_registry import PERSONA_CATEGORY, PromptTemplateStore

            store = PromptTemplateStore(self.config)
            row = store.get_by_id(int(override_id))
            if row is not None and row.category == PERSONA_CATEGORY and (row.body or "").strip():
                if self.verbose:
                    console.print(
                        f"[dim]─── Persona override: {row.title or row.key} "
                        f"(id={override_id}) ───[/dim]"
                    )
                return row.body
            logger.warning(
                "Bot %s persona override id=%s unresolved (missing/non-persona/empty); "
                "falling back to base system_prompt",
                self.bot_id, override_id,
            )
        except Exception as e:
            logger.warning(
                "Failed to resolve persona override id=%s for bot %s: %s; using base prompt",
                override_id, self.bot_id, e,
            )
        return self.bot.system_prompt

    @property
    def _prompt_bot_type(self) -> str:
        """'agent' for agent-backend bots, else 'chat' — drives manifest gating (TASK-489)."""
        return (
            "agent"
            if self.model_definition.get("type") in ("agent_backend", "claude-code")
            else "chat"
        )

    def _walk_manifest(self, builder: "PromptBuilder", prompt: str, stage: str) -> None:
        """Render the declared PROMPT_MANIFEST sections for `stage` that apply to
        this bot's type. The single assembly driver (TASK-489) — both the stable
        base and the per-turn copy are built by walking the same ordered list."""
        bot_type = self._prompt_bot_type
        for spec in PROMPT_MANIFEST:
            if spec.stage != stage or bot_type not in spec.applies_to:
                continue
            getattr(self, spec.method)(builder, prompt)

    def _init_system_prompt(self):
        """Build the stable (cacheable) system prompt by walking the manifest."""
        builder = PromptBuilder()

        # Walk the STABLE manifest entries — the cached base prefix (TASK-489).
        self._walk_manifest(builder, "", STABLE)

        # We store the builder, not the final string, so we can add context later
        self._prompt_builder = builder
        
        # For backward compatibility, also set config.SYSTEM_MESSAGE
        # This will be the base prompt without per-query memory context
        self.config.SYSTEM_MESSAGE = builder.build()
        
        if self.verbose:
            console.print(builder.get_verbose_summary())

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

    def _init_history(self):
        """Initialize history manager."""
        self.history_manager = HistoryManager(
            client=self.client,
            config=self.config,
            db_backend=self.memory.get_short_term_manager() if self.memory else None,
            bot_id=self.bot_id,
            settings_getter=self._resolve_setting,
        )
        self.load_history()
    
    def load_history(self, since_minutes: int | None = None) -> list[dict]:
        """Load conversation history.

        With budget-driven summarization, all messages are loaded by default.
        The token budget in get_context_messages() handles windowing.
        """
        self.history_manager.load_history(since_minutes=since_minutes)
        return [msg.to_dict() for msg in self.history_manager.messages]

    def _resolve_setting(self, key: str, fallback: Any = None) -> Any:
        """Resolve a runtime setting for this bot (via the unified resolver)."""
        if self.settings is None:
            return fallback
        return self.config_resolver.resolve_scalar_value(key=key, fallback=fallback)

    def _get_generation_kwargs(self) -> dict[str, Any]:
        """Resolve per-bot generation parameters for LLM client calls."""
        return {
            "temperature": float(self._resolve_setting("temperature", self.config.TEMPERATURE)),
            "top_p": float(self._resolve_setting("top_p", self.config.TOP_P)),
            "max_tokens": int(self.config.get_model_max_tokens(getattr(self.client, "model_alias", None))),
        }
    
    def query(self, prompt: str, plaintext_output: bool = False, stream: bool = True) -> str:
        """Send a query to the LLM and return the response.
        
        This is the main entry point for queries. It builds the context
        messages directly via ``_build_context_messages`` (the sole live
        assembly path) and executes them, optionally through the tool loop.

        Args:
            prompt: User's question/message
            plaintext_output: If True, skip rich formatting
            stream: If True, stream the response

        Returns:
            Assistant's response string
        """
        try:
            # Add user message to history first
            self.history_manager.add_message("user", prompt)

            # Build the outbound context messages (system prompt + history)
            context_messages = self._build_context_messages(prompt)
            
            # Use tool loop if bot has tools enabled (full tools or read-only memory)
            # Disable tools if tool_format is "none" (explicitly configured no tool support)
            if self.tool_format == "none":
                use_tools = False
                if self.bot.uses_tools and self.config.VERBOSE:
                    logger.info(f"Tool calling disabled for model {self.resolved_model_alias} (tool_format=none)")
            else:
                use_tools = (self.bot.uses_tools and (self.memory is not None or self.home_client is not None or self.ha_native_client is not None)) or (
                    self.memory and not self.bot.uses_tools
                )
            # Resolve per-bot generation parameters once for this query
            gen_kwargs = self._get_generation_kwargs()

            if use_tools:
                if self.bot.uses_tools:
                    tool_definitions = self._get_tool_definitions()
                    if not tool_definitions:
                        use_tools = False
                else:
                    # Non-tool memory bots get read-only memory search only
                    from ..tools.definitions import MEMORY_TOOL
                    tool_definitions = [MEMORY_TOOL]
                if use_tools:
                    assistant_response, tool_context, tool_call_details = query_with_tools(
                        messages=context_messages,
                        client=self.client,
                        memory_client=self.memory,
                        profile_manager=self.profile_manager,
                        search_client=self.search_client if self.bot.uses_tools else None,
                        home_client=self.home_client if self.bot.uses_tools else None,
                        ha_native_client=self.ha_native_client if self.bot.uses_tools else None,
                        news_client=self.news_client if self.bot.uses_tools else None,
                        web_fetch_client=self.web_fetch_client if self.bot.uses_tools else None,
                        model_lifecycle=self.model_lifecycle if self.bot.uses_tools else None,
                        config=self.config,
                        user_id=self.user_id,
                        bot_id=self.bot_id,
                        stream=stream,
                        tool_format=self.tool_format,
                        tools=tool_definitions,
                        adapter=self.adapter,
                        history_manager=self.history_manager,
                        generation_kwargs=gen_kwargs,
                    )
                else:
                    # Pass adapter stop sequences even without tools
                    adapter_stops = self.adapter.get_stop_sequences()
                    assistant_response = self.client.query(
                        context_messages,
                        plaintext_output=plaintext_output,
                        stream=stream,
                        stop=adapter_stops or None,
                        **gen_kwargs,
                    )
                    # Apply adapter output cleaning as safety net
                    if assistant_response:
                        cleaned = self.adapter.clean_output(assistant_response)
                        if cleaned != assistant_response:
                            logger.debug(
                                f"Adapter '{self.adapter.name}' cleaned response: "
                                f"{len(assistant_response)} -> {len(cleaned)} chars"
                            )
                            assistant_response = cleaned
                    tool_call_details = []
                    tool_context = ""

                # Render the final response (tool loop returns raw text, never renders)
                if assistant_response:
                    if not plaintext_output:
                        self.client._print_assistant_message(assistant_response)
                    else:
                        print(assistant_response)

                # Save tool context to history
                if tool_context:
                    from datetime import datetime
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
                    self.history_manager.add_message("system", f"[Tool Results @ {ts}]\n{tool_context}")
            else:
                # Pass adapter stop sequences even without tools
                adapter_stops = self.adapter.get_stop_sequences()
                assistant_response = self.client.query(
                    context_messages,
                    plaintext_output=plaintext_output,
                    stream=stream,
                    stop=adapter_stops or None,
                    **gen_kwargs,
                )
                # Apply adapter output cleaning as safety net
                if assistant_response:
                    cleaned = self.adapter.clean_output(assistant_response)
                    if cleaned != assistant_response:
                        logger.debug(
                            f"Adapter '{self.adapter.name}' cleaned response: "
                            f"{len(assistant_response)} -> {len(cleaned)} chars"
                        )
                        assistant_response = cleaned
                tool_call_details = []

            if assistant_response:
                self.history_manager.add_message("assistant", assistant_response)

            # Write debug turn log if enabled
            if self.debug:
                self._write_debug_turn_log(context_messages, prompt, assistant_response, tool_call_details)

            return assistant_response
        
        except KeyboardInterrupt:
            console.print("[bold red]Query interrupted.[/bold red]")
            self.history_manager.remove_last_message_if_partial("assistant")
            return ""
        except Exception as e:
            console.print(f"[bold red]Error during query:[/bold red] {e}")
            logger.exception(f"Error during query: {e}")
            self.history_manager.remove_last_message_if_partial("assistant")
            return ""
    
    def _assemble_system_builder(self, prompt: str = "") -> PromptBuilder:
        """Assemble the per-turn system-prompt builder (TASK-487).

        Extracted verbatim from ``_build_context_messages`` so there is exactly
        ONE assembly path: the live turn and the effective-config inspector both
        call this, guaranteeing the inspector can never report a prompt that
        differs from what is actually sent. Pure w.r.t. state that matters to the
        cached prefix — it starts from a ``.copy()`` of the stable builder and
        adds only per-turn sections, so the cached base prefix is untouched.

        Each per-turn section is tagged with ``metadata`` {source, gate} purely
        for inspection; metadata does not affect ``build()`` output.
        """
        # Start with a copy of the stable prompt builder (TASK-288: per-turn
        # sections ride the copy, never the cached base).
        builder = self._prompt_builder.copy()

        # NOTE: temporal grounding (build_temporal_context @ SectionPosition.DATETIME)
        # was removed here (TASK-288). It changed every minute and sat near the TOP
        # of the system prompt — the cacheable prefix — so it busted the prompt cache
        # for agent bots on every turn and added per-turn variance for chatbots too.
        # Relative-time grounding needs a proper redesign off the cacheable prefix
        # (a separate future task); do NOT re-add it to the system prompt.

        # Walk the PER_TURN manifest entries onto the copy (TASK-489).
        self._walk_manifest(builder, prompt, PER_TURN)
        return builder

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

    def describe_effective_config(self, prompt: str = "") -> dict[str, Any]:
        """Inspect the same effective prompt/config path a live turn uses."""
        from .effective_config import describe_effective_config

        return describe_effective_config(self, prompt)

    def _build_context_messages(self, prompt: str, context_suffix: str | None = None) -> list[Message]:
        """Build messages list with system prompt, memory context, and history.

        ``context_suffix`` (TASK-391) is per-turn text appended to the OUTBOUND
        user message only — it is model-visible but never persisted (the caller
        already stored the clean ``prompt``). Used for the agent attachment
        manifest; see ``prepare_messages_for_query``.
        """
        messages = []

        # Assemble the per-turn system-prompt builder. Extracted into one method
        # (TASK-487) so the effective-config inspector drives the SAME code path
        # and can never drift from what a live turn actually sends.
        builder = self._assemble_system_builder(prompt)

        # Response-style instruction derived from keywords in the user's message.
        # Lets the user shape any bot's answer inline without extra config.
        #
        # TASK-288: this is per-turn, user-message-driven variance. It must NOT be
        # baked into the system prompt — doing so (a) is backwards coupling (the
        # user's text mutating the system block) and (b) busts the cacheable
        # system-prompt prefix on agent bots whenever a keyword fires. Compute it
        # here, then stamp it onto the OUTBOUND COPY of the user message below
        # (mirroring the agent_user_prefix pattern) so it shapes the answer
        # without ever landing in stored history/memory or the cached prefix.
        # Matched on word boundaries so it doesn't trip on substrings inside
        # unrelated words.
        # TASK-490: bodies come from the registry (chat.response_style.*),
        # bot-overridable. Keyword detection stays here; only the text moved.
        response_style: str | None = None
        if prompt:
            _p = prompt.lower()
            _style_key: str | None = None
            if re.search(r"\btldr\b", _p):
                _style_key = "chat.response_style.tldr"
            elif re.search(r"\beli5\b", _p):
                _style_key = "chat.response_style.eli5"
            elif re.search(r"\bdeep[ -]?dive\b", _p):
                _style_key = "chat.response_style.deep_dive"
            if _style_key:
                resolved = self.config_resolver.resolve_body(_style_key)
                response_style = resolved.body if resolved else None

        # Build final system message
        system_content = builder.build()
        if system_content:
            messages.append(Message(role="system", content=system_content))
        
        if self.debug:
            logger.debug(f"System message: {len(system_content)} chars")
            logger.debug(f"Sections: {[s.name for s in builder.enabled_sections]}")
        
        # Agent backends manage their own conversation history —
        # only include the current user message, skip old history.
        #
        # Per-turn user-message prefixes: some content is per-turn or mode-
        # dependent (e.g. the voice tts_output_instructions when a session began
        # in text mode) and we deliberately keep it OFF the system prompt so the
        # cacheable system-prompt prefix stays byte-stable across turns
        # (TASK-288). Such content is stamped on the user message instead — the
        # single implementation point for every agent backend.
        #
        # NOTE: the Claude Code CLI DOES freeze the system prompt per session
        # (recorded at session start and after a manual /compact; a changed
        # prompt on resume is ignored). llm-bawt pins the prompt per session and
        # sends the pinned bytes (TASK-1318, service/system_prompt_pin.py), so a
        # system-prompt edit lands at the next session boundary. Per-turn content
        # must therefore ride the user message: it would never reach a resumed
        # session through the system prompt.
        #
        # Slash commands (e.g. /new) are passed through untouched because the
        # bridges consume them via a literal startswith("/new") check before
        # the agent ever sees them.
        if self.model_definition.get("type") in ("agent_backend", "claude-code"):
            user_content = prompt
            # Per-turn prefixes that ride on the user message (kept off the cached
            # system prefix). Anything mode-dependent or session-configurable is
            # stamped per-turn here.
            #
            #   - chat.agent_user_prefix  fires whenever _inject_user_prefix is
            #     True (controlled by the chat composer's "Agent prefix" toggle
            #     via the inject_user_prefix request flag). Mode-agnostic.
            #   - chat.agent_voice_prefix fires whenever _tts_mode is True,
            #     regardless of the user-prefix toggle. Voice-only.
            #
            # If both fire, they stack with the user_prefix first (broader
            # "session-wide" guidance), then voice_prefix (the narrow voice
            # constraint), then a `---` separator, then the actual user text.
            # Slash commands (e.g. /new) are passed through untouched because
            # the bridges parse them with a literal startswith() before the
            # agent ever sees them.
            if not _SLASH_COMMAND_RE.match(prompt or ""):
                from ..prompt_registry import get_prompt_resolver
                resolver = get_prompt_resolver(self.config)
                prefix_parts: list[str] = []
                applied_keys: list[tuple[str, int, str]] = []

                if self._inject_user_prefix:
                    resolved_user = resolver.resolve("chat.agent_user_prefix")
                    body_user = (resolved_user.body if resolved_user else "").strip()
                    if body_user:
                        prefix_parts.append(body_user)
                        applied_keys.append(("chat.agent_user_prefix", len(body_user), resolved_user.source))

                if self._tts_mode:
                    resolved_voice = resolver.resolve("chat.agent_voice_prefix")
                    body_voice = (resolved_voice.body if resolved_voice else "").strip()
                    if body_voice:
                        prefix_parts.append(body_voice)
                        applied_keys.append(("chat.agent_voice_prefix", len(body_voice), resolved_voice.source))

                # TASK-288: inline response-style (tldr/eli5/deep dive) rides on
                # the user message, not the system prompt, so it never busts the
                # cached system-prompt prefix. Last so it's the narrowest framing.
                if response_style:
                    prefix_parts.append(response_style)
                    applied_keys.append(("response_style", len(response_style), "inline-keyword"))

                if prefix_parts:
                    combined = "\n\n".join(prefix_parts)
                    user_content = f"{combined}\n\n---\n\n{prompt}"
                    if self.debug:
                        applied_repr = ", ".join(
                            f"{key}(len={n}, src={src})" for key, n, src in applied_keys
                        )
                        logger.debug(f"Agent user-message prefixes applied: {applied_repr}")
            # TASK-391: per-turn attachment manifest — appended to the outbound
            # user message only (model-visible, never persisted). Kept AFTER the
            # user text so the manifest reads as trailing context.
            if context_suffix:
                user_content = f"{user_content}\n\n{context_suffix}"
            messages.append(Message(role="user", content=user_content))
            return messages

        # Always include history — two-layer architecture handles context overflow.
        # TASK-615: the legacy max_context_tokens total-budget override is GONE
        # (that key was renamed to history_tokens with a different meaning: the
        # Tier-3 raw-bucket cap, applied inside the assembler ladder). The total
        # prompt budget has exactly one source: Tier-2 resolve_context_budget.
        max_context_tokens = 0
        if self.client:
            _, _, max_context_tokens = self.config.resolve_context_budget(
                getattr(self.client, "model_alias", None)
            )

        if self._ha_mode:
            # HA-mode: drop summaries, keep only last 6 user/assistant/tool-result msgs.
            # Device commands are stateless — long history causes hallucinated tool calls.
            history = self.history_manager.get_context_messages(
                max_tokens=max_context_tokens
            )
            ha_msgs = []
            for msg in history:
                if msg.role in ("user", "assistant"):
                    ha_msgs.append(msg)
                elif msg.role == "system" and msg.content and (
                    msg.content.startswith("[Tool Results]")
                    or msg.content.startswith("[Tools used:")
                ):
                    ha_msgs.append(msg)
                # Skip 'summary' role entirely
            messages.extend(ha_msgs[-6:])
        else:
            # TASK-493/518: chat history assembly goes through the ONE shared
            # handler (build_context_payload) — the same function the agent seed
            # path uses. history_scope decodes into two INDEPENDENT flags
            # (include_history, include_summaries); continuity is the coarse master
            # (a mirror of scope != "none"), ANDed in so a legacy row that turned
            # continuity off still carries nothing. The legacy per-request
            # include_summaries flag is a compat shim: it can only force summaries
            # OFF for this turn, never on.
            continuity = bool(
                self.config_resolver.resolve_config_setting(
                    "session_memory_continuity"
                ).value
            )
            scope = str(
                self.config_resolver.resolve_config_setting("history_scope").value
                or "inline+summaries"
            )
            include_history, include_summaries = scope_flags(scope)
            include_history = include_history and continuity
            include_summaries = include_summaries and continuity and self._include_summaries
            payload = self.history_manager.build_context_payload(
                include_history=include_history,
                include_summaries=include_summaries,
                delivery="inline",
                max_tokens=max_context_tokens,
            )
            messages.extend(payload.inline_history)

        # TASK-288: stamp the inline response-style directive onto the OUTBOUND
        # COPY of the latest user message. We replace the list entry with a new
        # Message rather than mutating it in place — the entries are references to
        # the persisted history objects, so mutating would leak the directive into
        # stored history/memory.
        if response_style:
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].role == "user":
                    original = messages[i].content or ""
                    messages[i] = Message(
                        role="user",
                        content=f"{response_style}\n\n---\n\n{original}",
                    )
                    break

        return messages

    def _get_tool_definitions(self) -> list:
        include_search = self.search_client is not None
        include_news = self.news_client is not None
        include_web_fetch = self.web_fetch_client is not None
        include_home = self.home_client is not None
        include_models = self.model_lifecycle is not None

        # Convert HA native tools if available
        ha_native_tools = None
        if self.ha_native_client and self.ha_native_client.initialized:
            from ..tools.definitions import ha_tools_to_tool_definitions
            ha_native_tools = ha_tools_to_tool_definitions(self.ha_native_client.tools)

        tools = get_tools_list(
            include_search_tools=include_search,
            include_news_tools=include_news,
            include_web_fetch_tools=include_web_fetch,
            include_home_tools=include_home,
            include_model_tools=include_models,
            ha_native_tools=ha_native_tools,
        )
        if self.memory is None:
            disallowed = {"memory", "history", "profile", "self"}
            tools = [t for t in tools if t.name not in disallowed]
        return tools

    def _retrieve_cold_start_memories(self, prompt: str) -> str:
        """Retrieve a small set of high-importance memories for cold-start context.

        Called when a tool-enabled bot has sparse history (≤3 messages).
        Injects a few key facts about the user into the system prompt so the
        model has immediate context without needing to call the memory tool first.

        Args:
            prompt: Current user prompt (used for relevance search)

        Returns:
            Formatted memory context string, or empty string if nothing found
        """
        if not self.memory:
            return ""

        try:
            # Fetch a small number of high-relevance memories
            n_results = max(
                1,
                min(
                    10,
                    int(self._resolve_setting("memory_n_results", 3) or 3),
                ),
            )
            results = self.memory.search(
                prompt,
                n_results=n_results,
                min_relevance=float(
                    self._resolve_setting("memory_min_relevance", self.config.MEMORY_MIN_RELEVANCE)
                ),
            )

            if not results:
                return ""

            memories = [
                {
                    "content": m.content,
                    "relevance": m.relevance,
                    "tags": m.tags,
                    "importance": m.importance,
                }
                for m in results
            ]

            from ..memory.context_builder import build_memory_context_string

            from ..profiles import EntityType

            user_name = None
            if self.profile_manager:
                try:
                    profile, _ = self.profile_manager.get_or_create_profile(
                        EntityType.USER, self.user_id
                    )
                    user_name = profile.display_name
                except Exception:
                    pass

            context = build_memory_context_string(memories, user_name=user_name)
            if context:
                logger.debug(
                    f"Cold-start memory priming: injected {len(memories)} memories "
                    f"({len(context)} chars) into system prompt"
                )
            return context

        except Exception as e:
            logger.warning(f"Cold-start memory retrieval failed: {e}")
            return ""

    def _write_debug_turn_log(
        self,
        context_messages: list[Message],
        user_prompt: str,
        assistant_response: str,
        tool_calls: list[dict] | None = None,
    ):
        """Write the current turn's request/response data to a debug log file.

        Only called when --debug is enabled. Overwrites the file on each turn
        to show the most recent request/response for review.

        Args:
            context_messages: Full list of messages sent to the LLM
            user_prompt: The user's input for this turn
            assistant_response: The assistant's response
            tool_calls: Per-call tool details from the tool loop
        """
        try:
            logs_dir = resolve_log_dir()
            logs_dir.mkdir(parents=True, exist_ok=True)

            log_file = logs_dir / "debug_turn.txt"

            # Build the log content
            lines = []
            lines.append("=" * 80)
            lines.append(f"DEBUG TURN LOG - {datetime.now().isoformat()}")
            lines.append(f"Model: {self.resolved_model_alias}")
            lines.append(f"Bot: {self.bot_id} ({self.bot.name})")
            lines.append(f"User: {self.user_id}")
            lines.append("=" * 80)
            lines.append("")

            # Request data - all context messages
            lines.append("─" * 40)
            lines.append("REQUEST MESSAGES")
            lines.append("─" * 40)
            for i, msg in enumerate(context_messages):
                lines.append(f"\n[{i}] Role: {msg.role}")
                lines.append(f"    Timestamp: {msg.timestamp}")
                lines.append(f"    Content ({len(msg.content)} chars):")
                lines.append("    " + "─" * 36)
                # Indent content for readability
                for content_line in msg.content.split("\n"):
                    lines.append(f"    {content_line}")
                lines.append("")

            # Tool calls section (between request and response)
            if tool_calls:
                total_calls = len(tool_calls)
                iterations = max((tc.get("iteration", 1) for tc in tool_calls), default=1)
                lines.append("─" * 40)
                lines.append(f"TOOL CALLS ({total_calls} call{'s' if total_calls != 1 else ''} across {iterations} iteration{'s' if iterations != 1 else ''})")
                lines.append("─" * 40)
                for idx, tc in enumerate(tool_calls, 1):
                    lines.append("")
                    lines.append(f"[{idx}] Tool: {tc.get('tool', 'unknown')}")
                    params = tc.get('parameters', {})
                    if isinstance(params, dict):
                        for pk, pv in params.items():
                            lines.append(f"    {pk}: {pv}")
                    else:
                        lines.append(f"    Parameters: {params}")
                    result = tc.get('result', '')
                    lines.append(f"    Result ({len(result)} chars):")
                    lines.append("    " + "─" * 36)
                    for result_line in str(result)[:2000].split("\n"):
                        lines.append(f"    {result_line}")
                lines.append("")

            # Response data
            lines.append("─" * 40)
            lines.append("RESPONSE")
            lines.append("─" * 40)
            lines.append(f"Length: {len(assistant_response)} chars")
            lines.append("")
            lines.append(assistant_response)
            lines.append("")

            # Also dump as JSON for machine parsing
            lines.append("─" * 40)
            lines.append("JSON FORMAT (for parsing)")
            lines.append("─" * 40)
            json_data = {
                "timestamp": datetime.now().isoformat(),
                "model": self.resolved_model_alias,
                "bot_id": self.bot_id,
                "user_id": self.user_id,
                "request": [msg.to_dict() for msg in context_messages],
                "tool_calls": tool_calls or [],
                "response": assistant_response,
            }
            lines.append(json.dumps(json_data, indent=2, ensure_ascii=False))

            # Write to file (overwrite)
            log_file.write_text("\n".join(lines), encoding="utf-8")
            logger.debug(f"Debug turn log written to: {log_file}")

        except Exception as e:
            logger.warning(f"Failed to write debug turn log: {e}")
    
    def refine_prompt(self, prompt: str, history: list | None = None) -> str:
        """Refine the user's prompt using context."""
        try:
            from ..bots import get_system_prompt
            
            messages = []
            refine_prompt = get_system_prompt("refine") or "You are a prompt refinement assistant."
            messages.append(Message(role="system", content=refine_prompt))
            
            if history:
                history_text = []
                for msg in history:
                    if hasattr(msg, 'role') and hasattr(msg, 'content'):
                        history_text.append(f"{msg.role}: {msg.content}")
                    elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                        history_text.append(f"{msg['role']}: {msg['content']}")
                
                if history_text:
                    context = f"Previous conversation:\n{chr(10).join(history_text)}\n\nUser's prompt: {prompt}"
                    messages.append(Message(role="user", content=context))
                else:
                    messages.append(Message(role="user", content=f"User's prompt: {prompt}"))
            else:
                messages.append(Message(role="user", content=f"User's prompt: {prompt}"))
            
            return self.client.query(messages, plaintext_output=True, stream=False)
        except Exception as e:
            logger.warning(f"Prompt refinement failed: {e}")
            return prompt
