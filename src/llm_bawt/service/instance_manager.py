"""LLM client and bot instance management mixin.

Extracted from background_service.py — handles model resolution,
LLMBawt caching, memory clients, and background client lifecycle.
"""

from __future__ import annotations

from typing import Any

from ..bot_types import agent_backend_for_model_def
from ..bots import BotManager
from ..model_catalog import bot_model_ref, resolve_model_config
from .logging import get_service_logger

log = get_service_logger(__name__)


class InstanceManagerMixin:
    """Mixin providing model/bot instance management for BackgroundService."""

    def _on_model_unloaded(self, model_alias: str):
        """Called when a model is unloaded - clears related caches."""
        log.debug(f"Model '{model_alias}' unloaded - clearing caches")

        # Clear client cache for this model
        if model_alias in self._client_cache:
            del self._client_cache[model_alias]

        # Clear all LLMBawt instances that use this model
        keys_to_remove = [
            key for key in self._llm_bawt_cache
            if key[0] == model_alias
        ]
        for key in keys_to_remove:
            del self._llm_bawt_cache[key]

        log.debug(f"Cleared {len(keys_to_remove)} cached instances for model '{model_alias}'")

    def invalidate_bot_instances(self, bot_id: str) -> int:
        """Invalidate cached ServiceLLMBawt instances for a bot across models/users."""
        normalized_bot_id = (bot_id or "").strip().lower()
        if not normalized_bot_id:
            return 0
        keys_to_remove = [
            key for key in self._llm_bawt_cache
            if key[1] == normalized_bot_id
        ]
        for key in keys_to_remove:
            del self._llm_bawt_cache[key]
        if keys_to_remove:
            log.debug(
                "Cleared %s cached instances for bot '%s'",
                len(keys_to_remove),
                normalized_bot_id,
            )
        return len(keys_to_remove)

    def invalidate_all_instances(self) -> int:
        """Invalidate all cached ServiceLLMBawt instances."""
        cleared = len(self._llm_bawt_cache)
        self._llm_bawt_cache.clear()
        if cleared:
            log.debug("Cleared all cached instances (%s)", cleared)
        return cleared

    def clear_session_model_overrides(self, bot_id: str | None = None, user_id: str | None = None) -> int:
        """Clear session model overrides, optionally scoped by bot and/or user."""
        if not self._session_model_overrides:
            return 0

        normalized_bot = (bot_id or "").strip().lower() if bot_id is not None else None
        normalized_user = (user_id or "").strip() if user_id is not None else None

        keys_to_remove: list[tuple[str, str]] = []
        for key in self._session_model_overrides:
            key_bot, key_user = key
            if normalized_bot is not None and key_bot != normalized_bot:
                continue
            if normalized_user is not None and key_user != normalized_user:
                continue
            keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._session_model_overrides[key]

        if keys_to_remove:
            scope = []
            if normalized_bot is not None:
                scope.append(f"bot={normalized_bot}")
            if normalized_user is not None:
                scope.append(f"user={normalized_user}")
            detail = " ".join(scope) if scope else "all sessions"
            log.info("Cleared %s session model override(s) for %s", len(keys_to_remove), detail)

        return len(keys_to_remove)

    def _load_available_models(self):
        """Load list of available models from config.

        Builds the canonical model catalog for the service:
        1. All models from models.yaml / DB
        2. Virtual model aliases for agent-backend bots
        3. A mapping from bot slug → backend model for agent-backend bots
           (stored separately — does NOT mutate bot objects)
        """
        models = self.config.defined_models.get("models", {})
        self._available_models = list(models.keys())

        # Agent-backend bot → virtual model mapping (no bot mutation)
        self._agent_backend_models: dict[str, str] = {}

        bot_manager = BotManager(self.config)
        for bot in bot_manager.list_bots():
            backend_name = getattr(bot, "agent_backend", None)
            if not backend_name:
                continue
            backend_config = getattr(bot, "agent_backend_config", {}) or {}
            # Register the virtual model definition if not already a real model
            if backend_name not in models:
                models[backend_name] = {
                    "type": "agent_backend",
                    "backend": backend_name,
                    "bot_config": backend_config,
                    "tool_support": "none",
                }
                if backend_name not in self._available_models:
                    self._available_models.append(backend_name)
                log.info(
                    "Registered virtual model '%s' for agent backend (bot=%s)",
                    backend_name,
                    bot.slug,
                )
            # Track the mapping without mutating the bot object
            self._agent_backend_models[bot.slug] = backend_name

        # Resolve default model: bot default → config default → first available
        default_bot = bot_manager.get_bot(self._default_bot) or bot_manager.get_default_bot()
        if default_bot:
            backend_name = self._agent_backend_models.get(default_bot.slug)
            if backend_name:
                # Agent default bot: prefer its real catalog model when it
                # resolves to a compatible entry; else the virtual alias.
                default_alias = bot_model_ref(self.config, default_bot)
                # Canonical endpoint refs (``model@access-path``) are valid
                # resolver inputs even when the compatibility model list exposes
                # only the unqualified key. Catalog resolution is authoritative.
                try:
                    resolved_backend = agent_backend_for_model_def(
                        resolve_model_config(
                            self.config,
                            default_alias,
                            harness=getattr(default_bot, "harness", None),
                            default={},
                        )
                    )
                except Exception:  # noqa: BLE001
                    resolved_backend = None
                if default_alias and resolved_backend == backend_name:
                    self._default_model = default_alias
                else:
                    self._default_model = backend_name
            elif default_bot.default_model and default_bot.default_model in self._available_models:
                self._default_model = default_bot.default_model
        if not self._default_model:
            config_default = getattr(self.config, "DEFAULT_MODEL_ALIAS", None)
            if config_default and config_default in self._available_models:
                self._default_model = config_default
        if not self._default_model and self._available_models:
            self._default_model = self._available_models[0]

        log.debug(
            "Loaded %d models (default=%s, agent_backends=%s)",
            len(self._available_models), self._default_model,
            list(self._agent_backend_models.values()),
        )

    def _resolve_request_model(
        self,
        requested_model: str | None,
        bot_id: str,
        local_mode: bool,
    ) -> tuple[str, list[str]]:
        """Resolve the model alias for a request.

        This is the single authoritative resolution point. The CLI in service
        mode sends only the user's explicit --model (or None).

        Priority:
        1. Agent-backend bots are locked to their backend model
        2. Explicit requested model (if available on service)
        3. Bot's default_model (from bot_profiles DB)
        4. Service default model
        5. First available model (with warning)

        Returns:
            Tuple of (resolved_model_alias, list_of_warnings)
        """
        warnings: list[str] = []
        bot_manager = BotManager(self.config)
        bot = bot_manager.get_bot(bot_id) or bot_manager.get_default_bot()

        # 1. Agent-backend bots are locked to their bot-configured model.
        #    Prefer the REAL catalog alias from bot.default_model (canonical
        #    model identity — shows the actual model in turn logs / SSE /
        #    task attribution).  Fall back to the legacy virtual backend
        #    alias (e.g. 'claude-code') only when default_model is missing
        #    or incompatible.
        backend_name = self._agent_backend_models.get(bot.slug if bot else bot_id)
        if backend_name:
            if requested_model and requested_model != backend_name:
                warnings.append(
                    f"Bot '{bot_id}' uses agent backend; ignoring requested model '{requested_model}'"
                )
            try:
                default_alias = bot_model_ref(self.config, bot) if bot else None
            except Exception:  # noqa: BLE001
                default_alias = getattr(bot, "default_model", None) if bot else None
            # ``bot_model_ref`` deliberately returns the canonical endpoint
            # ref, which need not exist in the legacy compatibility key list.
            try:
                resolved_backend = agent_backend_for_model_def(
                    resolve_model_config(
                        self.config,
                        default_alias,
                        harness=getattr(bot, "harness", None),
                        default={},
                    )
                )
            except Exception:  # noqa: BLE001
                resolved_backend = None
            if default_alias and resolved_backend == backend_name:
                return default_alias, warnings
            log.warning(
                "Agent bot '%s' has missing/incompatible default_model=%r for "
                "backend '%s'; falling back to virtual alias '%s'. Set the "
                "bot's Model to a %s catalog entry.",
                bot_id, default_alias, backend_name, backend_name, backend_name,
            )
            if backend_name in self._available_models:
                return backend_name, warnings

        # 2. Explicit requested model
        model = requested_model.strip() if requested_model else None
        if model:
            if model in self._available_models:
                return model, warnings
            # Requested model not available — fall through with warning
            warnings.append(f"Model '{model}' not available on service")

        # 3. Bot's pinned endpoint (or legacy default_model without a pin).
        #    A multi-endpoint model's bare key is absent from the compatibility
        #    map, so checking only default_model silently selects the service
        #    default instead. Never fall back to a different provider for a
        #    broken pin; catalog resolution must fail visibly.
        if bot and getattr(bot, "endpoint_id", None) is not None:
            ensure_catalog = getattr(self.config, "ensure_model_catalog", None)
            if not callable(ensure_catalog) or ensure_catalog() is None:
                raise ValueError(f"No model catalog available for bot '{bot_id}' endpoint pin")
            bot_default = bot_model_ref(self.config, bot)
            if warnings:
                warnings[-1] += f", using bot default '{bot_default}'"
            return bot_default, warnings
        if bot and bot.default_model and bot.default_model in self._available_models:
            if warnings:
                warnings[-1] += f", using bot default '{bot.default_model}'"
            return bot.default_model, warnings

        # 4. Service default model
        if self._default_model and self._default_model in self._available_models:
            if warnings:
                warnings[-1] += f", using service default '{self._default_model}'"
            return self._default_model, warnings

        # 5. First available
        if self._available_models:
            fallback = self._available_models[0]
            msg = f"No preferred model available, using '{fallback}'"
            log.warning(msg)
            warnings.append(msg)
            return fallback, warnings

        raise ValueError("No models available on service.")

    def get_memory_client(self, bot_id: str, user_id: str | None = None):
        """Get or create memory client for a bot/user pair."""
        cache_key = (bot_id, user_id)

        cached = self._memory_clients.get(cache_key)
        if cache_key in self._memory_clients:
            return cached

        with self._memory_clients_lock:
            if cache_key in self._memory_clients:
                return self._memory_clients[cache_key]
            try:
                from ..mcp_server.client import get_memory_client
                self._memory_clients[cache_key] = get_memory_client(
                    config=self.config,
                    bot_id=bot_id,
                    user_id=user_id,
                    server_url=self.config.MCP_SERVER_URL,
                )
                log.memory_operation("client_init", bot_id, details="MemoryClient created")
            except Exception as e:
                log.warning(f"Memory client unavailable for {bot_id}: {e}")
                self._memory_clients[cache_key] = None
            return self._memory_clients[cache_key]

    def _get_llm_bawt(self, model_alias: str, bot_id: str, user_id: str, local_mode: bool = False):
        """Get or create an LLMBawt instance with caching.

        This method enforces single-model loading through the ModelLifecycleManager.
        If a different model is requested, the current model will be unloaded first.

        Model selection priority:
        1. Pending model switch (from switch_model tool) - becomes the new session model
        2. Current session model override (from previous switch)
        3. Model from API request
        """
        from .core import ServiceLLMBawt

        # Session key for model overrides (per bot+user)
        session_key = (bot_id, user_id)

        # Check for pending model switch (from switch_model tool)
        pending = self._model_lifecycle.clear_pending_switch()
        if pending:
            log.info(f"🔄 Switching to model: {pending}")
            # Store as session override so subsequent requests use this model
            self._session_model_overrides[session_key] = pending
            model_alias = pending
        elif session_key in self._session_model_overrides:
            # Use the session model override from a previous switch
            model_alias = self._session_model_overrides[session_key]
            log.debug(f"Using session model override: {model_alias}")

        cache_key = (model_alias, bot_id, user_id)

        # Check if we need to switch models (different model requested).
        # Only unload when the PREVIOUS model holds local resources
        # (GGUF/vLLM) — mirrors ModelLifecycleManager.register_client.
        # API clients and agent backends are stateless; unloading them
        # would needlessly thrash caches now that agent bots resolve to
        # distinct real aliases (e.g. opus-4-7 vs gpt-5.4-codex).
        current_model = self._model_lifecycle.current_model
        if current_model and current_model != model_alias:
            prev_def = resolve_model_config(self.config, current_model, default={})
            if prev_def.get("type") in ("gguf", "vllm", "llamacpp"):
                log.info(f"🔄 Model: {current_model} → {model_alias}")
                # Unloading triggers _on_model_unloaded which clears caches
                self._model_lifecycle.unload_current_model()
            else:
                log.debug(f"Model switch: {current_model} → {model_alias} (no unload needed)")

        if cache_key in self._llm_bawt_cache:
            cached = self._llm_bawt_cache[cache_key]
            # If cached instance has no memory (local_mode=True) but this request
            # needs memory (local_mode=False), discard the stale instance so we
            # create a fresh memory-enabled one.
            #
            # The reverse (memory-enabled instance reused for a local_mode=True
            # request) is harmless — extra context won't break anything, and
            # memory extraction is gated by request.extract_memory separately.
            cached_local = getattr(cached, "local_mode", False)
            discard = False
            if cached_local and not local_mode:
                log.info(
                    "Upgrading cached local-mode instance → memory-enabled for %s/%s/%s",
                    model_alias, bot_id, user_id,
                )
                discard = True
            elif not local_mode and not cached_local:
                # Non-local instance whose memory init failed (e.g. transient DB
                # outage).  _db_available=False + memory=None means the init
                # exception path was hit — discard so we retry the connection.
                # (requires_memory=False sets _db_available=True, so won't match.)
                from ..utils.config import has_database_credentials
                has_memory = getattr(cached, "memory", None) is not None
                db_available = getattr(cached, "_db_available", False)
                if not has_memory and not db_available and has_database_credentials(self.config):
                    log.info(
                        "Discarding cached instance with failed memory init for %s/%s/%s — retrying",
                        model_alias, bot_id, user_id,
                    )
                    discard = True

            if discard:
                del self._llm_bawt_cache[cache_key]
            else:
                log.cache_hit("llm_bawt", f"{model_alias}/{bot_id}/{user_id}")
                log.debug(f"Reusing cached ServiceLLMBawt instance for {cache_key}")
                return cached

        log.cache_miss("llm_bawt", f"{model_alias}/{bot_id}/{user_id}")

        # Get model type for logging and cache decisions
        resolution_bot = BotManager(self.config).get_bot(bot_id)
        model_def = resolve_model_config(
            self.config,
            model_alias,
            harness=getattr(resolution_bot, "harness", None),
            default={},
        )
        model_type = model_def.get("type", "unknown")

        # Check if we already have a client for this model in the client cache
        # If so, reuse it to avoid reloading GGUF models into VRAM.
        # Skip for agent_backend models — they're lightweight and hold per-bot
        # state (_bot_config with session_key) that must not be shared.
        existing_client = None
        if model_type not in ("agent_backend", "claude-code"):
            existing_client = self._client_cache.get(model_alias)

        # Only log model loading if we don't have the client cached
        if not existing_client:
            log.model_loading(model_alias, model_type, cached=False)
        else:
            log.model_loading(model_alias, model_type, cached=True)
        try:
            # Create a copy of config for each ServiceLLMBawt instance
            # This is necessary because it modifies config.SYSTEM_MESSAGE
            # based on the bot's system prompt
            instance_config = self.config.model_copy(deep=True)
            llm_bawt = ServiceLLMBawt(
                resolved_model_alias=model_alias,
                config=instance_config,
                local_mode=local_mode,
                bot_id=bot_id,
                user_id=user_id,
                existing_client=existing_client,  # Reuse cached client if available
            )
            self._llm_bawt_cache[cache_key] = llm_bawt

            # Cache reusable chat clients. Botless background jobs use their
            # separate pool and never borrow a bot's client or identity.
            if model_alias not in self._client_cache and model_type not in ("agent_backend", "claude-code"):
                self._client_cache[model_alias] = llm_bawt.client
            # Note: BaseLLMBawt.__init__ already registers with lifecycle manager

        except Exception as e:
            log.model_error(model_alias, str(e))
            raise

        return self._llm_bawt_cache[cache_key]

    def get_client(self, model_alias: str):
        """Get a botless extraction client, never a bot's cached chat client."""
        client, resolved_alias = self._get_background_client(model_override=model_alias, job="mcp_extraction")
        if client is None or resolved_alias is None:
            raise ValueError(f"Model '{model_alias}' has no botless background client")
        return client

    def _maintenance_reasoning_effort(self) -> str | None:
        """Global ``maintenance_reasoning_effort``, read per call (TASK-1729)."""
        from ..runtime_setting_resolution import resolve_global_runtime_setting

        value = resolve_global_runtime_setting(self.config, "maintenance_reasoning_effort")
        return (str(value).strip() or None) if value else None

    def _get_background_client(
        self, model_override: str | None = None, *, job: str = "background",
        allow_fallback: bool = False,
    ) -> tuple[Any | None, str | None]:
        """Background client wrapped so every call's outcome is recorded for ``job``.

        TASK-1704: a job with no usable client is recorded as a failure too, so
        "fell back to heuristics" stops being silent. See ``maintenance_health``.

        ``allow_fallback`` is set only by callers whose model came from settings
        (not one a caller named): then a provider spending limit replays the
        call on ``maintenance_model_fallback``. See ``maintenance_fallback``.
        """
        from .maintenance_health import UNAVAILABLE, MonitoredClient, maintenance_health

        client, model_alias, model_def = self._resolve_background_endpoint(model_override)
        provider, credential = self._background_credential_labels(model_alias, model_def)
        if client is None:
            maintenance_health().record_failure(
                job, UNAVAILABLE, "no usable background client for the configured job model",
                model=model_alias or model_override, provider=provider, credential=credential,
            )
            return None, None
        monitored = MonitoredClient(client, job=job, model=model_alias, provider=provider,
                                    credential=credential,
                                    reasoning_effort=self._maintenance_reasoning_effort)
        if not allow_fallback or not provider:
            return monitored, model_alias
        return self._with_maintenance_fallback(monitored, job, model_alias, provider), model_alias

    def _with_maintenance_fallback(self, primary: Any, job: str, model_alias: str,
                                   provider: str) -> Any:
        """Wrap ``primary`` so a spending limit reaches the configured backup (slice B)."""
        from ..runtime_settings import resolve_job_model
        from .maintenance_fallback import (
            COOLDOWN_SETTING, FALLBACK_SETTING, Backup, FallbackClient,
            default_state_health, default_store,
        )
        from .maintenance_health import MonitoredClient, maintenance_health

        def incompatible(alias: str | None, why: str) -> None:
            maintenance_health().record_failure(
                job, "fallback_unavailable", f"backup {FALLBACK_SETTING}: {why}",
                model=alias, provider=None, credential=None)

        def open_backup(primary_provider: str) -> Backup | None:
            alias = resolve_job_model(self.config, FALLBACK_SETTING)
            if not alias:
                incompatible(None, "not set")
                return None
            client, backup_alias, backup_def = self._resolve_background_endpoint(alias)
            if client is None or backup_alias is None:
                incompatible(alias, "not in the catalog or no API client")
                return None
            backup_provider, credential = self._background_credential_labels(backup_alias, backup_def)
            if not backup_provider or backup_provider == primary_provider:
                incompatible(backup_alias, "must be on a different provider than the primary")
                return None
            return Backup(client=MonitoredClient(client, job=job, model=backup_alias,
                                                 provider=backup_provider, credential=credential,
                                                 reasoning_effort=self._maintenance_reasoning_effort),
                          model=backup_alias, provider=backup_provider)

        def cooldown() -> float:
            from ..runtime_settings import RuntimeSettingsResolver
            from ..setting_definitions import SETTING_DEFINITIONS

            value = RuntimeSettingsResolver(config=self.config, bot=None).resolve(
                COOLDOWN_SETTING, fallback=SETTING_DEFINITIONS[COOLDOWN_SETTING].default)
            return max(0.0, float(value or 0))

        return FallbackClient(primary, job=job, model=model_alias, provider=provider,
                              open_backup=open_backup, store=default_store(self.config),
                              cooldown_seconds=cooldown, state_health=default_state_health(),
                              on_incompatible=incompatible)

    def _background_credential_labels(
        self, model_alias: str | None, model_def: dict | None = None,
    ) -> tuple[str | None, str | None]:
        """Provider id and stored credential *name* (never the secret) for a job model.

        Pass the ``model_def`` the client was built from: re-resolving a bare
        model key can pick a different endpoint or fail as ambiguous.
        """
        if not model_alias:
            return None, None
        try:
            from .background_clients import provider_label, resolve_background_model

            if model_def is None:
                model_def = resolve_background_model(self.config, model_alias) or {}
            provider = model_def.get("provider") or provider_label(self.config, model_def)
            if not provider:
                return None, None
            from .providers.base import CredentialStore
            record = CredentialStore(self.config).load(str(provider))
            return str(provider), (record.account or record.auth_method) if record else None
        except Exception:  # noqa: BLE001 - labels are best-effort, never fail the job
            return None, None

    def _resolve_background_client(self, model_override: str | None = None) -> tuple[Any | None, str | None]:
        """Get or create an API client for background tasks (summarization, extraction).

        This client is completely isolated from the main chat model lifecycle —
        it never triggers model switches or unloads the active chat model.

        Which endpoints are servable: ``service.background_clients`` (TASK-1726).

        Returns:
            ``(client, model_alias)`` or ``(None, None)`` if unavailable.
        """
        client, model_alias, _ = self._resolve_background_endpoint(model_override)
        return client, model_alias

    def _resolve_background_endpoint(
        self, model_override: str | None = None,
    ) -> tuple[Any | None, str | None, dict | None]:
        """:meth:`_resolve_background_client` plus the endpoint definition it built from."""
        from ..runtime_settings import resolve_job_model

        preferred = (
            model_override
            or resolve_job_model(self.config, "extraction_model")
            or resolve_job_model(self.config, "maintenance_model")
        )
        if not preferred:
            log.warning("No background job model configured; falling back to heuristics.")
            return None, None, None

        # TASK-857: background jobs are GLOBAL (TASK-522), resolved with no bot
        # identity. TASK-1726: the ``background`` harness (after ``chat``) also
        # admits api-key/no-auth anthropic-messages and responses endpoints;
        # ``background_clients`` picks the client by protocol, never by name.
        from .background_clients import (
            UnsupportedBackgroundModel, build_background_client, resolve_background_model,
        )

        try:
            model_def = resolve_background_model(self.config, preferred)
        except Exception as e:
            log.error(f"Background model '{preferred}' cannot serve background jobs: {e}")
            return None, None, None
        if not model_def:
            log.error(f"Background model '{preferred}' not found in catalog")
            return None, None, None

        # Keyed pool: reuse this model's client if already built. Each resolved
        # model gets its own stable, reused client — created once, then only
        # read — so concurrent background jobs resolving different models no
        # longer thrash a single slot (TASK-281). Keyed by the catalog's
        # canonical model key, not the raw preferred string. Lock-free read;
        # dict.get is atomic under the GIL.
        model_alias = str(model_def.get("model_key") or preferred)
        # One client per endpoint: the same model on two access paths (a
        # primary and its backup) must not share a client.
        cache_key = f"{model_alias}@{model_def['access_path']}" if model_def.get("access_path") else model_alias
        cached = self._bg_client_cache.get(cache_key)
        if cached is not None:
            return cached, model_alias, model_def

        try:
            client, kind = build_background_client(self.config, model_def, model_alias)
        except UnsupportedBackgroundModel as e:
            log.warning("Background model '%s' is unsupported: %s", preferred, e)
            return None, None, None
        except Exception as e:
            log.error("Failed to create background client for '%s': %s", model_alias, e)
            return None, None, None
        # Insert under the lock so two concurrent cold-start callers for the
        # same model don't double-build; the loser reuses the winner.
        with self._bg_client_lock:
            client = self._bg_client_cache.setdefault(cache_key, client)
        log.info("Background client ready: %s (%s)", cache_key, kind)
        return client, model_alias, model_def
