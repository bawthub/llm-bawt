"""Orchestrator initialization and optional integration wiring."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..profiles import ProfileManager
    from ..integrations.web_fetch.client import WebFetchClient
from rich.console import Console
from ..bots import Bot, BotManager
from ..clients import LLMClient
from ..runtime_settings import RuntimeSettingsResolver
from ..config_resolver import ConfigResolver
from ..mcp_server.client import MemoryClient
from ..integrations.ha_mcp.client import HomeAssistantMCPClient, HomeAssistantNativeClient
from ..integrations.newsapi.client import NewsAPIClient
from ..search import get_search_client, SearchClient
from ..utils.config import Config
from ..utils.history import HistoryManager
from ..adapters import get_adapter, ModelAdapter
from .model_lifecycle import ModelLifecycleManager, get_model_lifecycle

logger = logging.getLogger(__name__)
console = Console()


class OrchestratorInitializationMixin:
    def __init__(
        self,
        resolved_model_alias: str,
        config: Config,
        local_mode: bool = False,
        bot_id: str = "nova",
        user_id: str = "",  # Required - must be passed explicitly
        verbose: bool = False,
        debug: bool = False,
        existing_client: LLMClient | None = None,
    ):
        """Initialize the LLM orchestrator.
        
        Args:
            resolved_model_alias: The model alias from models.yaml
            config: Application configuration
            local_mode: If True, skip database features
            bot_id: Bot personality to use
            user_id: User profile ID (required)
            verbose: Enable verbose output (--verbose)
            debug: Enable debug output (--debug)
            existing_client: Reuse this client instead of creating a new one
        """
        if not user_id:
            raise ValueError("user_id is required - set LLM_BAWT_DEFAULT_USER or pass --user")
        self.resolved_model_alias = resolved_model_alias
        self.config = config
        resolution_bot = BotManager(config).get_bot(bot_id)
        resolution_harness = getattr(resolution_bot, "harness", None)
        from ..model_catalog import resolve_model_config

        self.model_definition = resolve_model_config(
            self.config,
            resolved_model_alias,
            harness=resolution_harness,
        )
        self.tool_format = self.config.get_tool_format(model_alias=resolved_model_alias, model_def=self.model_definition)
        self.local_mode = local_mode
        self.bot_id = bot_id
        self.user_id = user_id
        self.verbose = verbose
        self.debug = debug
        
        # Will be initialized below
        self.memory: MemoryClient | None = None
        self.profile_manager: ProfileManager | None = None
        self.search_client: SearchClient | None = None
        self.home_client: HomeAssistantMCPClient | None = None
        self.ha_native_client: HomeAssistantNativeClient | None = None
        self.news_client: NewsAPIClient | None = None
        self.web_fetch_client: "WebFetchClient | None" = None
        self.model_lifecycle: ModelLifecycleManager | None = None
        self.client: LLMClient
        self.bot: Bot
        self.history_manager: HistoryManager
        self._db_available: bool = False
        self.adapter: ModelAdapter
        self.settings: RuntimeSettingsResolver | None = None
        self._client_system_context: str | None = None
        self._ha_mode: bool = False
        self._include_summaries: bool = True
        self._tts_mode: bool = False
        # When True, prepend chat.agent_user_prefix body to every user message
        # for agent backends. Independent of tts_mode (which gates the voice
        # prefix) — both can fire and stack.
        self._inject_user_prefix: bool = False
        # TASK-251: per-turn explicit thread selection. When set (request
        # carried session_id), the turn persists to AND assembles context from
        # that thread. None (the default and the primary mode) = continuous
        # history. Set fresh on every turn by the dispatch paths — never
        # sticky across turns on a cached instance.
        self._session_id_override: str | None = None
        # TASK-1281: this turn's working-memory section text (agent bots the
        # policy allows; None otherwise). Set fresh every turn by the dispatch
        # paths, like the session override above.
        self._working_memory_section: str | None = None
        # TASK-1309: this turn's skill resolution, set by the injected-skills
        # section on every agent build; the session pin replays its manifest.
        self._skill_resolution = None

        if not self.model_definition:
            raise ValueError(f"Could not find model definition for: '{resolved_model_alias}'")
        
        # Initialize model lifecycle manager (singleton)
        self.model_lifecycle = self._get_model_lifecycle(config)
        
        # Initialize adapter based on model definition
        self.adapter = get_adapter(
            self.resolved_model_alias,
            self.model_definition
        )
        
        # Initialize LLM client - reuse existing if provided
        if existing_client is not None:
            self.client = existing_client
            logger.debug(f"Reusing existing client for model '{resolved_model_alias}'")
        else:
            logger.debug(f"Creating new client for model '{resolved_model_alias}'")
            self.client = self._initialize_client()
        
        # Log effective per-model configuration
        if self.verbose:
            logger.info(
                f"Model '{resolved_model_alias}' config: "
                f"context_window={self.client.effective_context_window}, "
                f"max_tokens={self.client.effective_max_tokens}, "
                f"type={self.model_definition.get('type', 'unknown')}"
            )
        
        # Register client with lifecycle manager
        if self.model_lifecycle is not None:
            self.model_lifecycle.register_client(resolved_model_alias, self.client)
        
        # Initialize bot
        self._init_bot(config)
        self.settings = RuntimeSettingsResolver(config=config, bot=self.bot, bot_id=self.bot_id)
        # TASK-488: single resolver over scalars (runtime_settings) + bodies
        # (prompt_templates). Wraps self.settings (shares its cache) so the live
        # prompt path and the inspector resolve everything through one API with
        # uniform provenance. Other call sites keep the legacy resolvers.
        self.config_resolver = ConfigResolver(
            config=config, bot=self.bot, bot_id=self.bot_id, settings=self.settings
        )
        
        # Initialize memory and profiles
        self._init_memory(config)
        
        # Initialize search client
        self._init_search(config)

        # Initialize Home Assistant client
        self._init_home_assistant(config)

        # Initialize NewsAPI client
        self._init_newsapi()

        # Initialize web fetch client (Crawl4AI)
        self._init_web_fetch()

        # Build system prompt
        self._init_system_prompt()
        
        # Initialize history manager
        self._init_history()
    
    def _get_model_lifecycle(self, config):
        """Ordinary clients register globally; isolated automation overrides this."""
        return get_model_lifecycle(config)

    def _init_bot(self, config: Config):
        """Initialize bot configuration."""
        bot_manager = BotManager(config)
        bot = bot_manager.get_bot(self.bot_id)
        
        if not bot:
            logger.warning(f"Bot '{self.bot_id}' not found, falling back to default")
            bot = bot_manager.get_default_bot()
            self.bot_id = bot.slug
        
        self.bot = bot
        self.client.bot_name = self.bot.name
        self.client.model_alias = self.resolved_model_alias

        if self.verbose:
            logger.info(f"Using bot: {self.bot.name} ({self.bot_id})")
    
    def _init_memory(self, config: Config):
        """Initialize memory client - base class is a no-op.

        Service-side subclass (ServiceLLMBawt) overrides this to set up
        database connections. CLI never initializes memory directly.
        """
        self._db_available = False
        logger.debug("BaseLLMBawt._init_memory() is a no-op; memory is service-managed")
    
    def _init_search(self, config: Config):
        """Initialize search client if bot uses search."""
        if not getattr(self.bot, 'uses_search', False):
            return
        
        try:
            self.search_client = get_search_client(config)
            if self.search_client and self.search_client.is_available():
                logger.debug(f"Search client initialized: {self.search_client.PROVIDER.value}")
            else:
                logger.debug("Search requested but no provider available")
                self.search_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize search: {e}")
            self.search_client = None
    
    def _init_home_assistant(self, config: Config):
        """Initialize Home Assistant integration.

        Priority: HA native MCP > legacy custom MCP server.
        """
        if not self.bot.uses_tools or not self.bot.uses_home_assistant:
            return

        # Try native MCP first (direct connection to HA's /api/mcp). Its
        # credentials come from the Integrations record first (TASK-1476).
        try:
            client = HomeAssistantNativeClient(config)
            if client.available:
                tools = client.discover_tools()
                if tools:
                    self.ha_native_client = client
                    logger.info(f"HA native MCP initialized with {len(tools)} tools")
                    return
                else:
                    logger.warning("HA native MCP connected but no tools discovered")
        except Exception as e:
            logger.warning(f"Failed to initialize HA native MCP: {e}")

        # Fallback to legacy custom MCP server
        if not getattr(config, "HA_MCP_ENABLED", False):
            return
        try:
            client = HomeAssistantMCPClient(config)
            if client.available:
                self.home_client = client
                logger.debug("Legacy Home Assistant MCP client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize legacy HA MCP client: {e}")

    def _init_newsapi(self) -> None:
        """Initialize NewsAPI client if API key is available."""
        if not self.bot.uses_tools:
            return
        try:
            from ..search.credentials import resolve_key  # noqa: PLC0415

            client = NewsAPIClient(api_key=resolve_key(self.config, "newsapi").value)
            if client.is_available():
                self.news_client = client
                logger.debug("NewsAPI client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize NewsAPI client: {e}")

    def _init_web_fetch(self) -> None:
        """Initialize web fetch client if Crawl4AI service is available."""
        if not self.bot.uses_tools:
            return
        try:
            from ..integrations.web_fetch.client import WebFetchClient
            client = WebFetchClient()
            if client.is_available():
                self.web_fetch_client = client
                logger.debug("Web fetch client initialized (Crawl4AI)")
            else:
                logger.debug("Crawl4AI service not reachable, web_fetch tool disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize web fetch client: {e}")

