"""Service-side LLMBawt class - supports loading local models (GGUF, etc.)

This is separate from the CLI's core.py which only supports OpenAI API calls.
The service runs on the server and can load models directly.
"""

import logging
import time
from typing import TYPE_CHECKING

from rich.console import Console

from ..clients import LLMClient, GrokClient
from ..clients.openai_client import OpenAIClient
from ..core.base import BaseLLMBawt
from ..utils.config import Config, has_database_credentials
from ..utils.history import Message
from ..tools import query_with_tools
from .logging import get_service_logger

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
slog = get_service_logger(__name__)
console = Console()


class ServiceLLMBawt(BaseLLMBawt):
    """Server-side LLM class that can load local models directly.
    
    Supports:
    - OpenAI-compatible API
    - GGUF models via llama-cpp-python
    - Memory augmentation when database is available
    """
    
    def __init__(
        self,
        resolved_model_alias: str,
        config: Config,
        local_mode: bool = False,
        bot_id: str = "nova",
        user_id: str = "",  # Required - must be passed explicitly
        verbose: bool = False,
        debug: bool = False,
        existing_client: "LLMClient | None" = None,
    ):
        """Initialize ServiceLLMBawt.
        
        Args:
            resolved_model_alias: Model alias from models.yaml
            config: Application configuration
            local_mode: Skip database features
            bot_id: Bot personality to use
            user_id: User profile ID (required)
            verbose: Enable verbose logging (--verbose)
            debug: Enable debug logging (--debug)
            existing_client: Reuse this LLM client instead of loading model again
        """
        super().__init__(
            resolved_model_alias=resolved_model_alias,
            config=config,
            local_mode=local_mode,
            bot_id=bot_id,
            user_id=user_id,
            verbose=verbose,
            debug=debug,
            existing_client=existing_client,
        )
        self._last_history_load_at: float = time.time()
        from ..clients.agent_backend_client import AgentBackendClient
        if isinstance(self.client, AgentBackendClient):
            # TASK-618: one DB-backed global controls both the MCP tool's
            # synchronous wait and Claude's client-side MCP timeout. Add 30s of
            # transport/cleanup headroom before sending milliseconds downstream.
            from ..runtime_setting_resolution import resolve_global_runtime_setting

            bot_wait_seconds = int(resolve_global_runtime_setting(
                self.config, "bot_send_wait_seconds", fallback=300,
            ) or 300)
            self.client._bot_config["mcp_tool_timeout_ms"] = (
                max(1, bot_wait_seconds) + 30
            ) * 1000

    def _init_bot(self, config: Config):
        """Initialize bot, then patch AgentBackendClient with bot-specific config."""
        super()._init_bot(config)
        # The virtual model 'openclaw' is registered once with the first bot's
        # config.  Now that self.bot is loaded, override the client's bot_config
        # so each bot uses its own session_key / settings.
        from ..clients.agent_backend_client import AgentBackendClient
        if isinstance(self.client, AgentBackendClient):
            self.client._bot_config = dict(self.bot.agent_backend_config or {})
            # Include bot + user identifiers so agent backends can scope
            # per-bot and per-user sessions deterministically.
            self.client._bot_config["bot_id"] = self.bot_id
            self.client._bot_config["user_id"] = self.user_id
            # Canonical model injection: the bot's default_model is THE model
            # reference for agent bots.  Resolve it through the catalog and
            # inject the SDK model_id into the bridge config — this overrides
            # any legacy ``agent_backend_config.model`` (which is migrated to
            # ``session_model`` and no longer user-facing).
            from ..bot_types import agent_backend_for_model_def
            from ..model_catalog import resolve_model_config
            default_alias = self._agent_model_ref()
            if default_alias:
                model_def = resolve_model_config(
                    self.config,
                    default_alias,
                    harness=getattr(self.bot, "harness", None),
                    default={},
                )
                if (
                    agent_backend_for_model_def(model_def)
                    == getattr(self.bot, "agent_backend", None)
                    and model_def.get("model_id")
                ):
                    self.client._bot_config["model"] = model_def["model_id"]
                for field in ("endpoint_id", "harness", "access_path"):
                    if model_def.get(field) is not None:
                        self.client._bot_config[field] = model_def[field]
                # Per-endpoint reasoning policy belongs to the normalized model
                # catalog, not provider/model-name branches in bridge transport.
                # Preserve an explicit per-bot override when present; otherwise
                # feed the catalog value into the bridge's existing per-turn
                # effort field.
                configured_effort = model_def.get("reasoning_effort")
                if (
                    "effort" not in self.client._bot_config
                    and isinstance(configured_effort, str)
                    and configured_effort.strip()
                ):
                    self.client._bot_config["effort"] = configured_effort.strip().lower()
                responses_transport = model_def.get("responses_transport")
                if isinstance(responses_transport, str) and responses_transport.strip():
                    self.client._bot_config["responses_transport"] = (
                        responses_transport.strip().lower()
                    )
                provider_system_prompt = model_def.get("provider_system_prompt")
                if isinstance(provider_system_prompt, str) and provider_system_prompt:
                    self.client._bot_config["provider_system_prompt"] = (
                        provider_system_prompt
                    )

                # TASK-609: resolve the catalog context window app-side and
                # inject it so the bridge reports the true window for
                # proxy-routed models the Claude CLI defaults to 200k. The app
                # owns catalog resolution (the single Tier-2 authority); the
                # bridge consumes this scalar — no bridge-side window table.
                # Resolve from the bot's pinned endpoint_id: a bare alias is
                # ambiguous when a model has >1 endpoint (grok-4.5 ->
                # xai-chat + xai-responses) even WITH the harness, and
                # resolve_model only swallows ModelNotFoundError — an
                # Ambiguous/Incompatible error would propagate. The int
                # endpoint id is a single-row lookup, so it can't be ambiguous;
                # the bot's endpoint is harness-compatible by DB invariant.
                # Fall back to the harness-resolved alias def when no endpoint
                # is pinned (e.g. direct-Anthropic bots).
                try:
                    window = None
                    compact_threshold = None
                    ep_id = model_def.get("endpoint_id") or getattr(self.bot, "endpoint_id", None)
                    if ep_id is not None:
                        ep_def = self.config.resolve_model(
                            ep_id,
                            harness=getattr(self.bot, "harness", None),
                            default={},
                        )
                        if isinstance(ep_def, dict):
                            window = ep_def.get("context_window")
                            compact_threshold = ep_def.get("compact_threshold")
                    if window is None:
                        window = model_def.get("context_window")
                    if compact_threshold is None:
                        compact_threshold = model_def.get("compact_threshold")
                    if window and int(window) > 0:
                        self.client._bot_config["context_window"] = int(window)
                    # Claude Code auto-compact window: catalog per-model value,
                    # else the global agent_compact_threshold setting.
                    if compact_threshold is None:
                        from ..runtime_setting_resolution import resolve_global_runtime_setting

                        compact_threshold = resolve_global_runtime_setting(
                            self.config, "agent_compact_threshold"
                        )
                    if compact_threshold and int(compact_threshold) > 0:
                        self.client._bot_config["compact_threshold"] = int(compact_threshold)
                except Exception:
                    logger.debug(
                        "TASK-609: could not resolve context_window "
                        "(alias=%s endpoint=%s)",
                        default_alias,
                        getattr(self.bot, "endpoint_id", None),
                    )

            # TASK-546: subagent_model is stored in agent_backend_config (like
            # effort/max_turns) so it flows through _bot_config automatically.
            # No resolution needed — agent_bridge.py reads it directly.

            # TASK-276: local GPU models are ordinary chat bots whose
            # model_definition was rewritten to type=agent_backend/backend=local
            # in _initialize_client.  The routing key the local-model-bridge
            # needs is the catalog ALIAS (it re-resolves repo_id/filename via
            # /v1/models).  The block above only fires for true agent bots
            # (agent_backend set), so set the alias here for the local case —
            # and forward the nested original definition as a resolution
            # fallback for the bridge.
            client_def = getattr(self.client, "model_definition", {}) or {}
            if str(client_def.get("backend") or "").strip() == "local":
                self.client._bot_config["model"] = (
                    client_def.get("model_id") or self.resolved_model_alias
                )
                local_def = client_def.get("local_model_definition")
                if isinstance(local_def, dict):
                    self.client._bot_config["local_model_definition"] = local_def

    def _agent_model_ref(self):
        from ..model_catalog import bot_model_ref
        return bot_model_ref(self.config, self.bot)

    def _history_manager_type(self):
        from ..utils.history import HistoryManager
        return HistoryManager

    def _init_history(self):
        """Ensure history always persists to PostgreSQL, even in local_mode.

        The base implementation uses ``self.memory`` for the DB backend,
        which is None in local_mode.  Here we create a lightweight memory
        client solely for history persistence so that ``local_mode`` only
        controls *prompt augmentation*, not *persistence*.
        """
        db_backend = None
        if self.memory:
            db_backend = self.memory.get_short_term_manager()
        elif has_database_credentials(self.config):
            try:
                from ..mcp_server.client import get_memory_client
                _history_client = get_memory_client(
                    config=self.config,
                    bot_id=self.bot_id,
                    user_id=self.user_id,
                    server_url=self.config.MCP_SERVER_URL,
                )
                db_backend = _history_client.get_short_term_manager()
                self._db_available = True
                logger.debug("History-only DB backend initialized for %s (local_mode=%s)", self.bot_id, self.local_mode)
            except Exception as e:
                logger.warning("Failed to init history DB backend: %s", e)

        self.history_manager = self._history_manager_type()(
            client=self.client,
            config=self.config,
            db_backend=db_backend,
            bot_id=self.bot_id,
            settings_getter=self._resolve_setting,
        )
        self.load_history()

    def _init_memory(self, config: Config):
        """Initialize memory client and profile manager for service-side operation."""
        if self.local_mode:
            logger.debug("Local mode enabled - skipping memory augmentation")
            return

        if not has_database_credentials(config):
            logger.debug("Database credentials not configured")
            return

        if not self.bot.requires_memory:
            logger.debug(f"Bot '{self.bot.name}' has requires_memory=false")
            self._db_available = True  # DB available but not needed
            return

        try:
            from ..mcp_server.client import get_memory_client
            from ..profiles import ProfileManager

            self.memory = get_memory_client(
                config=config,
                bot_id=self.bot_id,
                user_id=self.user_id,
                server_url=config.MCP_SERVER_URL,
            )
            self._db_available = True
            logger.debug(f"Memory client initialized for bot: {self.bot_id}")

            # Initialize profile manager
            self.profile_manager = ProfileManager(config)

        except Exception as e:
            self.profile_manager = None
            # Log without full traceback for connection errors
            error_str = str(e)
            if "could not translate host name" in error_str or "Connection refused" in error_str:
                logger.error(f"Failed to initialize memory: {e}")
            else:
                logger.exception(f"Failed to initialize memory: {e}")
            if self.verbose:
                console.print(f"[yellow]Warning:[/yellow] Memory client failed: {e}")

    def _refresh_history_if_stale(self) -> None:
        """Refresh history from DB only when cache is stale."""
        ttl = max(
            0.0,
            float(
                self._resolve_setting(
                    "history_reload_ttl_seconds",
                    getattr(self.config, "HISTORY_RELOAD_TTL_SECONDS", 0.0),
                )
            ),
        )
        now = time.time()
        is_empty = not bool(getattr(self.history_manager, "messages", []))
        is_stale = ttl == 0.0 or (now - self._last_history_load_at) >= ttl
        if is_empty or is_stale:
            self.load_history()
            self._last_history_load_at = now

    def invalidate_history_cache(self) -> None:
        """Force a history refresh on the next request."""
        self._last_history_load_at = 0.0
    
    def _initialize_client(self) -> LLMClient:
        """Initialize client based on model type - supports local models."""
        model_type = self.model_definition.get("type")
        model_id = self.model_definition.get("model_id")
        
        # Log model loading
        slog.model_loading(self.resolved_model_alias, model_type)
        start_time = time.perf_counter()
        
        if model_type == "openai":
            if not model_id:
                raise ValueError(
                    f"Missing 'model_id' in definition for '{self.resolved_model_alias}'"
                )
            base_url = self.model_definition.get("base_url")
            api_key = self.model_definition.get("api_key")
            client = OpenAIClient(
                model_id,
                config=self.config,
                base_url=base_url,
                api_key=api_key,
                model_definition=self.model_definition,
            )
            load_time_ms = (time.perf_counter() - start_time) * 1000
            slog.model_loaded(self.resolved_model_alias, model_type, load_time_ms)
            return client
        
        elif model_type == "grok":
            if not model_id:
                raise ValueError(
                    f"Missing 'model_id' in definition for '{self.resolved_model_alias}'"
                )
            # Explicit per-model key only; GrokClient resolves CredentialStore
            # first, then legacy env/config fallbacks.
            api_key = self.model_definition.get("api_key")
            client = GrokClient(
                model_id,
                config=self.config,
                api_key=api_key,
                model_definition=self.model_definition,
            )
            load_time_ms = (time.perf_counter() - start_time) * 1000
            slog.model_loaded(self.resolved_model_alias, model_type, load_time_ms)
            return client
        
        elif model_type in ("gguf", "vllm"):
            # TASK-276: local GPU inference (gguf via llama-cpp + vLLM) no
            # longer runs in this process.  A CUDA abort() in local inference
            # used to take down the whole app/MCP/agent session host; it now
            # runs in the standalone `local_model_bridge` process and talks
            # over Redis like the codex / claude-code bridges.
            #
            # We DON'T construct LlamaCppClient/VLLMClient here anymore — those
            # modules (and their heavy llama_cpp/vllm/torch deps) live in the
            # bridge package.  Instead we hand back an AgentBackendClient
            # routed to the "local" backend.  The TOP-LEVEL type must be
            # "agent_backend" so the executor/streaming branches treat it as a
            # remote bridge call (in-process GPU inference no longer exists; it
            # moved to local_model_bridge in TASK-276/278).  The
            # original local fields are preserved nested under
            # ``local_model_definition`` so the bridge can still resolve them;
            # the bridge ALSO re-resolves the alias against /v1/models, so this
            # is belt-and-suspenders.
            from ..clients.agent_backend_client import AgentBackendClient

            bot_config = {
                "bot_id": self.bot_id,
                "user_id": self.user_id,
                # The bridge's send_command reads config["model"] to tell the
                # local-model-bridge which GGUF/vLLM alias to load.
                "model": self.resolved_model_alias,
            }
            bridge_model_definition = {
                "type": "agent_backend",
                "backend": "local",
                # Carry the alias through so the bridge resolves the same
                # catalog entry the app saw.
                "model_id": self.resolved_model_alias,
                # Preserve the original local model fields verbatim for the
                # bridge (and for any app-side introspection).
                "local_model_definition": dict(self.model_definition),
            }
            # Surface context_window / max_tokens at the top level too so the
            # base LLMClient's effective_* helpers keep working app-side.
            if self.model_definition.get("context_window") is not None:
                bridge_model_definition["context_window"] = self.model_definition["context_window"]
            if self.model_definition.get("max_tokens") is not None:
                bridge_model_definition["max_tokens"] = self.model_definition["max_tokens"]

            # Update self.model_definition so _build_context_messages()
            # takes the agent_backend path (system + user only, no history).
            # Without this, self.model_definition still says "gguf" while
            # self.client.model_definition says "agent_backend", and the
            # context builder falls through to the regular history path
            # which can produce messages the AgentBackendClient doesn't
            # expect.
            self.model_definition = bridge_model_definition

            client = AgentBackendClient(
                backend_name="local",
                config=self.config,
                bot_config=bot_config,
                model_definition=bridge_model_definition,
            )
            load_time_ms = (time.perf_counter() - start_time) * 1000
            slog.model_loaded(self.resolved_model_alias, "agent_backend", load_time_ms)
            return client

        elif model_type == "ollama":
            if not model_id:
                raise ValueError(
                    f"Missing 'model_id' in definition for '{self.resolved_model_alias}'"
                )
            base_url = self.model_definition.get("base_url") or (
                f"{self.config.OLLAMA_URL.rstrip('/')}/v1"
            )
            client = OpenAIClient(
                model_id,
                config=self.config,
                base_url=base_url,
                api_key=self.model_definition.get("api_key") or "ollama",
                model_definition=self.model_definition,
            )
            load_time_ms = (time.perf_counter() - start_time) * 1000
            slog.model_loaded(self.resolved_model_alias, model_type, load_time_ms)
            return client

        elif model_type in ("agent_backend", "claude-code"):
            from ..bot_types import agent_backend_for_model_def
            from ..clients.agent_backend_client import AgentBackendClient
            # Derive the backend from the model definition shape.  The alias
            # fallback only existed for virtual backend-name aliases (e.g.
            # 'claude-code'); real catalog aliases (e.g. 'opus-4-7') resolve
            # via agent_backend_for_model_def.
            backend_name = (
                self.model_definition.get("backend")
                or agent_backend_for_model_def(self.model_definition)
                or self.resolved_model_alias
            )
            bot_config = {
                **(self.model_definition.get("bot_config", {}) or {}),
                "bot_id": self.bot_id,
                "user_id": self.user_id,
            }
            client = AgentBackendClient(
                backend_name=backend_name,
                config=self.config,
                bot_config=bot_config,
                model_definition=self.model_definition,
            )
            load_time_ms = (time.perf_counter() - start_time) * 1000
            slog.model_loaded(self.resolved_model_alias, "agent_backend", load_time_ms)
            return client

        else:
            raise ValueError(
                f"Unsupported model type: '{model_type}'. "
                f"Supported types: openai, grok, gguf, vllm, ollama, agent_backend, claude-code"
            )
    
    # =========================================================================
    # Service API methods - used by api.py for request handling
    # =========================================================================
    
    def prepare_messages_for_query(
        self,
        prompt: str,
        user_attachments: list[dict] | None = None,
        message_id: str | None = None,
        attachments: list[dict] | None = None,
        context_suffix: str | None = None,
        author=None,
    ) -> list[Message]:
        """Prepare messages for query including history and memory context.

        Called by the service API before sending to the LLM.

        Args:
            prompt: User's prompt
            user_attachments: Optional list of image attachments in the format
                [{"mimeType": "image/png", "content": "<base64>"}].
                When present, the last user message will use a multimodal
                content array so the LLM can see the images. This is the
                LLM-call-boundary inline form — it never touches the DB.
            message_id: Optional client-supplied UUID for the persisted user
                message.  When supplied, it becomes the ID in chat history
                and matches turn_log.trigger_message_id, enabling clean joins
                between history rows and tool-call events.
            attachments: TASK-225 — optional tiny JSONB payload to persist
                on the user-message row's ``attachments`` column, e.g.
                ``[{"asset_id": "ma_xxx", "kind": "image"}, ...]``.  The
                full asset metadata lives in ``media_assets`` and is
                fetched on read.  Distinct from ``user_attachments``,
                which carries the bytes for the LLM call.
            context_suffix: TASK-391 — optional per-turn text appended to the
                OUTBOUND user message only (rides the model-visible message
                like the agent prefixes). NEVER persisted: ``prompt`` above is
                what lands in history. Used to hand the agent a curlable
                attachment manifest without polluting the stored message.

        Returns:
            List of messages ready for LLM query
        """
        # TASK-251: explicit thread selection. When the request carried a
        # session_id, this turn assembles context from THAT thread's pool
        # (raw scoped to the thread + rolling summaries) and persists to it.
        # The scoped pool is strictly per-turn state on this cached instance —
        # invalidate the history cache so the next continuous turn reloads the
        # continuous pool instead of inheriting the scoped one.
        _thread_override = (getattr(self, "_session_id_override", None) or "").strip() or None
        if _thread_override:
            self.history_manager.load_history(session_id=_thread_override)
            self.invalidate_history_cache()
        else:
            # Refresh from DB only when stale to reduce per-turn read pressure.
            # Set HISTORY_RELOAD_TTL_SECONDS=0 to force legacy "reload every request".
            self._refresh_history_if_stale()

        # Add user message to history first (with optional attachment refs).
        self.history_manager.add_message(
            "user", prompt, message_id=message_id, attachments=attachments,
            session_id=_thread_override, author=author,
        )

        # Build context with system prompt, memory, and history. context_suffix
        # (e.g. the TASK-391 attachment manifest) rides the OUTBOUND user
        # message only — the clean ``prompt`` was already persisted above.
        messages = self._build_context_messages(prompt, context_suffix=context_suffix)

        # TASK-225 LLM-boundary inlining: ``user_attachments`` is the
        # in-memory ``{mimeType, content=naked-b64}`` contract produced by
        # the chat_streaming user-message resolver.  Both intake paths
        # converge on the same shape before this point:
        #
        #   1. NEW STYLE ``attachment_ids=[...]`` -> MediaStore.read_original
        #      -> {mimeType, content}.
        #   2. LEGACY STYLE inline ``image_url`` data URL -> {mimeType,
        #      content} AND auto-upload to MediaStore (persisted ref is
        #      passed in via ``attachments`` above).
        #
        # This is the ONLY place we materialize base64 in the request — it
        # is consumed by the LLM client and never written to the DB. The
        # ``attachments`` kwarg above is what gets persisted.
        if user_attachments:
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].role == "user":
                    image_parts = []
                    for att in user_attachments:
                        mime = att.get("mimeType", "image/png")
                        b64 = att.get("content", "")
                        image_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        })
                    messages[i].content_parts = image_parts
                    break

        return messages
    
    def execute_llm_query(
        self,
        messages: list[Message],
        plaintext_output: bool = False,
        stream: bool = False,
        inject_messages: list | None = None,
        thread_binding: dict | None = None,
        bridge_request_id: str | None = None,
        bridge_timeout_seconds: float | None = None,
        bridge_event_callback=None,
    ) -> tuple[str, str, list[dict]]:
        """Execute the LLM query and return response.
        
        Handles tool calling loop if bot has tools enabled.
        Called by the service API to get the response.
        
        Args:
            messages: Prepared messages from prepare_messages_for_query
            plaintext_output: Skip rich formatting
            stream: Enable streaming output
            
        Returns:
            Tuple of (response, tool_context, tool_call_details).
            tool_context is empty if no tools used.
            tool_call_details is a list of per-call dicts for debug logging.
        """
        # TASK-501: only agent backends (AgentBackendClient, **kwargs) accept
        # inject_messages; guard so plain clients never receive an unknown kwarg.
        _q_kwargs = {"inject_messages": inject_messages} if inject_messages else {}
        # TASK-252: request-local per-thread SDK binding (explicit session_id
        # turns) — rides the same kwarg channel as inject_messages so it can
        # never leak between concurrent turns via shared instance state.
        if thread_binding:
            _q_kwargs["thread_binding"] = thread_binding
        if bridge_request_id:
            _q_kwargs["bridge_request_id"] = bridge_request_id
        if bridge_timeout_seconds:
            _q_kwargs["bridge_timeout_seconds"] = bridge_timeout_seconds
        if bridge_event_callback is not None:
            _q_kwargs["bridge_event_callback"] = bridge_event_callback

        # Agent backends (claude-code/openclaw) execute tools in their OWN
        # bridge/runtime — they must NOT go through llm-bawt's tool loop. The
        # streaming path already enforces this (not is_agent_backend); mirror it
        # here so non-streaming stays consistent AND so inject_messages reaches
        # the single client.query below instead of being dropped in the tool
        # branch.
        _is_agent_backend = str(
            (getattr(self.client, "model_definition", None) or {}).get("type", "")
        ).strip() in ("agent_backend", "claude-code")

        tool_format_value = str(getattr(self.tool_format, "value", self.tool_format)).strip().lower()
        tools_enabled_for_model = tool_format_value != "none"

        # Use tool loop if bot has tools enabled, model supports tool format,
        # and at least one tool backend is available.
        if not _is_agent_backend and self.bot.uses_tools and tools_enabled_for_model and (
            self.memory
            or self.search_client
            or self.news_client
            or self.web_fetch_client
            or self.home_client
            or self.ha_native_client
            or self.model_lifecycle
        ):
            tool_definitions = self._get_tool_definitions()
            if not tool_definitions:
                response = self.client.query(
                    messages,
                    plaintext_output=plaintext_output,
                    stream=stream,
                )
                return response, "", []
            return query_with_tools(
                messages=messages,
                client=self.client,
                memory_client=self.memory,
                profile_manager=self.profile_manager,
                search_client=self.search_client,
                home_client=self.home_client,
                ha_native_client=self.ha_native_client,
                news_client=self.news_client,
                web_fetch_client=self.web_fetch_client,
                model_lifecycle=self.model_lifecycle,
                config=self.config,
                user_id=self.user_id,
                bot_id=self.bot_id,
                stream=stream,
                tool_format=self.tool_format,
                tools=tool_definitions,
                adapter=self.adapter,
                history_manager=self.history_manager,
                ha_mode=self._ha_mode,
            )
        
        response = self.client.query(
            messages,
            plaintext_output=plaintext_output,
            stream=stream,
            **_q_kwargs,
        )
        return response, "", []
    
    def finalize_response(
        self,
        response: str,
        tool_context: str = "",
        attachments: list[dict] | None = None,
        reasoning: str | None = None,
        message_id: str | None = None,
    ):
        """Finalize the response by saving to history.

        Called by the service API after receiving the response.

        Args:
            response: The assistant's response
            tool_context: Optional tool results to save to history
            attachments: Optional {asset_id, kind} media refs persisted during
                the turn (e.g. Playwright screenshots) to attach to the reply.
            reasoning: Optional model reasoning ("thinking") accumulated during
                the turn, persisted on the assistant row for display-only
                replay (TASK-301). Never re-fed into LLM context.
            message_id: Optional client-supplied UUID for the persisted ASSISTANT
                row. When set, the live streaming bubble (which the frontend
                minted this same UUID for) and the reloaded history row share one
                id, so they upsert-merge into a SINGLE bubble instead of the two
                that the content-fingerprint dedup used to (partially) reconcile.
                This closes the assistant-identity gap left open by EPIC TASK-217
                (see makeMessageId.ts). None → server mints a fresh UUID (legacy
                / server-originated turns like inter-bot, cron, agent dispatch).
        """
        # TASK-251: an explicit-thread turn persists its assistant/tool rows to
        # the SAME requested thread as the user row — never the active one.
        _thread_override = (getattr(self, "_session_id_override", None) or "").strip() or None

        # Save tool context first so it appears before the response in history
        if tool_context:
            self.history_manager.add_message(
                "system", f"[Tool Results]\n{tool_context}",
                session_id=_thread_override,
            )

        if response:
            from ..message_authorship import AuthorReference

            self.history_manager.add_message(
                "assistant", response, message_id=message_id,
                attachments=attachments, reasoning=reasoning,
                session_id=_thread_override,
                author=AuthorReference.bot(self.bot_id),
            )
    
    def refine_prompt(self, prompt: str, history: list | None = None) -> str:
        """Refine the user's prompt using context.
        
        Overrides base to add service-specific logging.
        """
        slog.debug(f"Refining prompt: {prompt[:50]}...")
        return super().refine_prompt(prompt, history)
