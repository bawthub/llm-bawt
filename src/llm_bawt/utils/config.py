import importlib.util
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console

from agent_bridge.chatgpt_models import CHATGPT_MODELS_CLIENT_VERSION

# Default system message fallback - only used if bot config fails to load
DEFAULT_SYSTEM_MESSAGE = """You are Nova, a personal AI assistant running locally via llm-bawt.
You have persistent memory across sessions. Keep responses concise and helpful."""

PROVIDER_OPENAI = "openai"
PROVIDER_CODEX = "codex"
PROVIDER_OLLAMA = "ollama"
PROVIDER_GGUF = "gguf"
PROVIDER_HF = "huggingface"
PROVIDER_VLLM = "vllm"
PROVIDER_GROK = "grok"
PROVIDER_AGENT_BACKEND = "agent_backend"
PROVIDER_UNKNOWN = "Unknown"

# BawtHub production frontend's stable service name on the shared llm-bawt
# network. Internal callers use it directly instead of routing through a
# Traefik hostname alias (legacy echo.* aliases were retired in TASK-1180).
DEFAULT_TASK_API_URL = "http://frontend-prod:3000"

logger = logging.getLogger(__name__)
console = Console()

def is_huggingface_available() -> bool:
    """Checks if Hugging Face dependencies (torch, transformers, bitsandbytes) are available."""
    torch_spec = importlib.util.find_spec("torch")
    transformers_spec = importlib.util.find_spec("transformers")
    # bitsandbytes is often optional but useful for quantization with HF
    _ = importlib.util.find_spec("bitsandbytes") 
    # Adjust logic if bitsandbytes should be strictly required
    return all([torch_spec, transformers_spec]) # Or include bitsandbytes_spec if mandatory

def is_llama_cpp_available() -> bool:
    # Restore original or remove if too obvious
    return importlib.util.find_spec("llama_cpp") is not None

def is_vllm_available() -> bool:
    """Checks if vLLM is available."""
    return importlib.util.find_spec("vllm") is not None

def get_default_config_dir() -> Path:
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    default_config_dir = Path(xdg_config_home) / "llm-bawt"
    default_config_dir.mkdir(parents=True, exist_ok=True)
    return default_config_dir


def get_default_models_yaml_path() -> Path:
    env_path = os.environ.get("LLM_BAWT_MODELS_CONFIG_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    return get_default_config_dir() / "models.yaml"

DEFAULT_MODELS_YAML = get_default_models_yaml_path()


def _env_file_has_llm_bawt_key(path: Path) -> bool:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[len("export "):].strip()
            if "=" not in stripped:
                continue
            key = stripped.split("=", 1)[0].strip()
            if key.startswith("LLM_BAWT_"):
                return True
    except OSError:
        return False
    return False


def get_dotenv_path() -> Path:
    env_override = os.environ.get("LLM_BAWT_ENV_FILE") or os.environ.get("LLM_BAWT_DOTENV_PATH")
    if env_override:
        return Path(env_override).expanduser().resolve()

    cwd_env = Path.cwd() / ".env"
    if cwd_env.is_file() and _env_file_has_llm_bawt_key(cwd_env):
        return cwd_env

    env_models = os.environ.get("LLM_BAWT_MODELS_CONFIG_PATH")
    if env_models:
        return Path(env_models).expanduser().resolve().parent / ".env"

    return get_default_config_dir() / ".env"


DOTENV_PATH = get_dotenv_path()


class RuntimeTunables(BaseModel):
    """Per-bot runtime tunables — introspectable for help text and field metadata.

    These are the settings that can be overridden per-bot via DB or YAML.
    Runtime settings keys are just the lowercase field name (e.g. MEMORY_N_RESULTS -> memory_n_results).
    """

    # --- Context & History --- #
    # TASK-615 clean break: the legacy context/summarization env fields are
    # DELETED, not aliased. Canonical replacements (all DB/registry-resolved,
    # no env): MAX_CONTEXT_TOKENS -> history_tokens (Tier-3 raw-bucket cap;
    # the total budget is Tier-2 resolve_context_budget). MAX_OUTPUT_TOKENS ->
    # the max_output_tokens runtime setting. SUMMARIZATION_SESSION_GAP_SECONDS/
    # _MIN_MESSAGES_PER_SESSION/_TRIGGER_TOKENS -> the Tier-1 summarization_job
    # dict. SUMMARIZATION_MAX_IN_CONTEXT -> summary_count. SUMMARIZATION_
    # COMPACT_CONTEXT -> compact_context. MEMORY_PROTECTED_RECENT_TURNS ->
    # summarization_job.protected_recent_turns.
    HISTORY_RELOAD_TTL_SECONDS: float = Field(default=2.0, description="How long service history cache stays fresh before reloading from DB (0 = every request)")

    # --- Memory Retrieval --- #
    MEMORY_N_RESULTS: int = Field(default=10, description="Semantic-memory results to retrieve on cold start")
    MEMORY_MIN_RELEVANCE: float = Field(default=0.01, description="Minimum cosine-similarity score for a memory hit to be included (0.0-1.0)")
    MEMORY_DEDUP_SIMILARITY: float = Field(default=0.85, description="Similarity threshold for deduplicating near-identical memory hits (0.0-1.0)")

    # --- Generation --- #
    TEMPERATURE: float = Field(default=0.8, description="Sampling temperature (0.0 = deterministic, higher = more creative)")
    TOP_P: float = Field(default=0.95, description="Nucleus sampling cutoff. Lower = less random.")

    # --- Agent Behavior --- #
    AGENT_GLOBAL_PROMPT_ENABLED: bool = Field(default=False, description="For agent backends (Claude Code, Codex): inject the shared 'agents.global_prompt' dynamic prompt into the system prompt (steers planning to the BawtHub task system instead of harness plan mode). Opt-in per bot.")


# Map of runtime settings key -> RuntimeTunables field name (trivial: just uppercase)
def runtime_key_to_config(key: str) -> str:
    """Convert a runtime settings key to its Config field name."""
    return key.upper()


def config_to_runtime_key(field_name: str) -> str:
    """Convert a Config field name to its runtime settings key."""
    return field_name.lower()


class Config(RuntimeTunables, BaseSettings):
    HISTORY_FILE: str = Field(default=os.path.expanduser("~/.cache/llm-bawt/chat-history"))
    OLLAMA_URL: str = Field(default="http://localhost:11434")
    MODEL_CACHE_DIR: str = Field(default=os.path.expanduser("~/.cache/llm-bawt/models"), description="Directory to cache downloaded GGUF models")
    MODELS_CONFIG_PATH: str = Field(default=str(DEFAULT_MODELS_YAML), description="Legacy path to models YAML (deprecated — models load from DB)")
    DEFAULT_MODEL_ALIAS: Optional[str] = Field(default=None, description="Default model alias if --model is not specified")
    DEFAULT_BOT: str = Field(default="nova", description="Default bot to use if --bot is not specified")
    DEFAULT_USER: Optional[str] = Field(
        default=None,
        description="User profile to use. Set via LLM_BAWT_DEFAULT_USER env var.",
    )

    # Optional llm-bawt MCP server (HTTP JSON-RPC). If set, memory/messages/tasks
    # operations route through MCP tool calls instead of direct DB access.
    MCP_SERVER_URL: Optional[str] = Field(
        default=None,
        description="Optional llm-bawt MCP server base URL (Set via LLM_BAWT_MCP_SERVER_URL)",
    )
    HA_MCP_ENABLED: bool = Field(
        default=False,
        description="Enable Home Assistant MCP integration (Set via LLM_BAWT_HA_MCP_ENABLED)",
    )
    HA_MCP_URL: str = Field(
        default="http://localhost:3000",
        description="Home Assistant MCP base URL (Set via LLM_BAWT_HA_MCP_URL)",
    )
    HA_MCP_TIMEOUT: int = Field(
        default=10,
        description="Timeout in seconds for Home Assistant MCP calls (Set via LLM_BAWT_HA_MCP_TIMEOUT)",
    )
    HA_MCP_AUTH_TOKEN: str = Field(
        default="",
        description="Optional bearer token for Home Assistant MCP (Set via LLM_BAWT_HA_MCP_AUTH_TOKEN)",
    )
    HA_NATIVE_MCP_URL: str = Field(
        default="",
        description="Home Assistant native MCP endpoint URL, e.g. http://hass.home:8123/api/mcp (Set via LLM_BAWT_HA_NATIVE_MCP_URL)",
    )
    HA_NATIVE_MCP_TOKEN: str = Field(
        default="",
        description="Home Assistant long-lived access token for native MCP auth (Set via LLM_BAWT_HA_NATIVE_MCP_TOKEN)",
    )
    HA_MCP_TOOL_EXCLUDE: str = Field(
        default="GetDateTime,HassCancelAllTimers,HassBroadcast",
        description="Comma-separated list of HA MCP tool names to exclude (Set via LLM_BAWT_HA_MCP_TOOL_EXCLUDE)",
    )

    # --- PostgreSQL Memory Backend Settings --- #
    POSTGRES_HOST: str = Field(default="localhost", description="PostgreSQL server hostname (Set via LLM_BAWT_POSTGRES_HOST)")
    POSTGRES_PORT: int = Field(default=5432, description="PostgreSQL server port (Set via LLM_BAWT_POSTGRES_PORT)")
    POSTGRES_USER: str = Field(default="llm_bawt", description="PostgreSQL username (Set via LLM_BAWT_POSTGRES_USER)")
    POSTGRES_PASSWORD: str = Field(default="", description="PostgreSQL password (Set via LLM_BAWT_POSTGRES_PASSWORD)")
    POSTGRES_DATABASE: str = Field(default="llm_bawt", description="PostgreSQL database name (Set via LLM_BAWT_POSTGRES_DATABASE)")
    
    # --- Memory Extraction Settings --- #
    MEMORY_EXTRACTION_ENABLED: bool = Field(default=True, description="Enable LLM-based memory extraction from conversations")
    MEMORY_EXTRACTION_MIN_IMPORTANCE: float = Field(default=0.5, description="Minimum importance score for extracted memories to be stored (0.5 = moderately important)")
    MEMORY_EXTRACTION_INTERVAL_MINUTES: int = Field(default=30, description="How often to run automatic memory extraction from unprocessed messages")
    MEMORY_PROFILE_ATTRIBUTE_ENABLED: bool = Field(default=True, description="Create profile attributes from extracted memories")
    MEMORY_PROFILE_ATTRIBUTE_MIN_IMPORTANCE: float = Field(default=0.7, description="Minimum importance for extracted profile attributes")
    MEMORY_EMBEDDING_DIM: int = Field(default=384, description="Dimension of embedding vectors for semantic search (384 for all-MiniLM-L6-v2)")
    MEMORY_EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2", description="Sentence-transformers model for local embeddings")
    # EXTRACTION_MODEL retired -> global runtime setting 'extraction_model' (TASK-522)

    # --- Memory Decay Settings --- #
    MEMORY_DECAY_ENABLED: bool = Field(default=True, description="Enable temporal decay for memory relevance scoring")
    MEMORY_DECAY_HALF_LIFE_DAYS: float = Field(default=90.0, description="Half-life in days for memory decay (90 = memory at 50% weight after 90 days)")
    MEMORY_ACCESS_BOOST_FACTOR: float = Field(default=0.15, description="How much to boost frequently accessed memories (0.15 = +15% per log(access_count))")
    MEMORY_RECENCY_WEIGHT: float = Field(default=0.3, description="Weight given to recency vs base importance (0.3 = 30% recency, 70% base)")
    MEMORY_DIVERSITY_ENABLED: bool = Field(default=True, description="Enable diversity sampling to avoid echo chambers in retrieval")
    
    # --- Memory Consolidation Settings --- #
    MEMORY_CONSOLIDATION_THRESHOLD: float = Field(default=0.75, description="Cosine similarity threshold for clustering similar memories (0.0-1.0)")

    # --- Scheduler Settings --- #
    SCHEDULER_ENABLED: bool = Field(default=True, description="Enable background job scheduler")
    SCHEDULER_CHECK_INTERVAL_SECONDS: int = Field(default=30, description="How often to check for due jobs")
    PROFILE_MAINTENANCE_INTERVAL_MINUTES: int = Field(default=60, description="Run profile summarization every N minutes")
    HISTORY_SUMMARIZATION_INTERVAL_MINUTES: int = Field(
        default=30,
        description="Run history summarization every N minutes",
    )
    # MAINTENANCE_MODEL / PROFILE_MAINTENANCE_MODEL retired -> global runtime
    # settings 'maintenance_model' / 'profile_maintenance_model' (TASK-522)

    # --- Service Settings --- #
    USE_SERVICE: bool = Field(default=False, description="Route queries through the background service by default (same as --service flag)")
    SERVICE_HOST: str = Field(default="127.0.0.1", description="Host for the background service to bind to")
    SERVICE_PORT: int = Field(default=8642, description="Port for the background service to listen on")
    AGENT_ORIGIN: str = Field(
        default="",
        description=(
            "Absolute base URL that agent backends use to fetch chat-upload "
            "assets, e.g. http://app:8642. Used to build curlable /v1/uploads "
            "URLs in the agent-visible 'Attached Images' manifest (TASK-391). "
            "Empty = emit relative /v1/uploads paths. Set via LLM_BAWT_AGENT_ORIGIN."
        ),
    )
    PUBLIC_ORIGIN: str = Field(
        default="https://app.bawthub.com",
        description=(
            "User-facing BawtHub origin used to build clickable links to "
            "uploaded assets (TASK-847): <origin>/api/chat/uploads/<asset_id>. "
            "Returned as public_url from POST /v1/uploads and echoed in the "
            "agent-visible attachment manifest. Empty = omit public links. "
            "Set via LLM_BAWT_PUBLIC_ORIGIN."
        ),
    )

    # --- xAI (Grok) Settings --- #
    XAI_API_KEY: str = Field(default="", description="xAI API key for Grok models (get from x.ai)")

    # --- ChatGPT subscription discovery (TASK-1296) --- #
    CODEX_CLIENT_VERSION: str = Field(
        default=CHATGPT_MODELS_CLIENT_VERSION,
        description=(
            "client_version the ChatGPT Codex backend's /models listing is told. "
            "The backend hides models whose minimal_client_version exceeds it, so "
            "this is the floor a listing may claim. Accepted value: a codex CLI "
            "version string (default: the CLI the agent-bridge image ships, "
            "0.157.1); raise it when a newer model needs a newer client. "
            "Set via LLM_BAWT_CODEX_CLIENT_VERSION."
        ),
    )

    # --- Agent Bridge Transport --- #
    REDIS_URL: str = Field(
        default="",
        description="Redis URL for bridge command/event transport (e.g. redis://redis:6379/0)",
    )

    # --- Agent Task API Settings --- #
    TASK_API_URL: str = Field(
        default=DEFAULT_TASK_API_URL,
        description="Base URL for the BawtHub agent task API (Set via LLM_BAWT_TASK_API_URL)",
    )

    # --- Web Search Settings --- #
    NEWSAPI_API_KEY: str = Field(
        default="",
        description="NewsAPI key for dedicated news tool (newsapi.org)",
    )
    SEARCH_PROVIDER: Optional[str] = Field(
        default=None,
        description=(
            "Search provider: 'duckduckgo' (free), 'tavily' (production), 'brave' "
            "(privacy-focused), or 'reddit' (official Reddit OAuth API). "
            "Auto-selects based on available keys."
        ),
    )
    TAVILY_API_KEY: str = Field(default="", description="Tavily API key for production search (get from tavily.com)")
    BRAVE_API_KEY: str = Field(default="", description="Brave Search API key")
    REDDIT_CLIENT_ID: str = Field(default="", description="Reddit OAuth app client ID for Reddit API search")
    REDDIT_CLIENT_SECRET: str = Field(default="", description="Reddit OAuth app client secret for Reddit API search")
    REDDIT_USER_AGENT: str = Field(default="llm-bawt/0.1", description="User-Agent string required by Reddit API")
    SEARCH_MAX_RESULTS: int = Field(default=5, description="Maximum search results to return per query")
    SEARCH_TIMEOUT: int = Field(default=10, description="Timeout in seconds for search requests (DuckDuckGo/Brave/Reddit)")
    SEARCH_INCLUDE_ANSWER: bool = Field(default=False, description="Include AI-generated answer from Tavily or Brave (uses more credits)")
    SEARCH_DEPTH: str = Field(default="basic", description="Tavily search depth: 'basic' (1 credit) or 'advanced' (2 credits)")
    BRAVE_SAFESEARCH: str = Field(
        default="moderate",
        description="Brave safesearch level: off, moderate, strict"
    )

    # SUMMARIZATION_MODEL retired -> global runtime setting 'summarization_model'
    # (TASK-522); default 'dolphin-qwen-3b' now lives in SETTING_DEFINITIONS.

    # --- Tool Calling Settings --- #
    MAX_TOOL_CALLS_PER_TURN: int = Field(
        default=20,
        description="Maximum number of tool calls per conversation turn (0 = unlimited)"
    )

    # --- LLM Hardware Settings --- #
    LLAMA_CPP_N_CTX: int = Field(default=32768, description="Context size for Llama.cpp models (32K for high VRAM)")
    LLAMA_CPP_N_GPU_LAYERS: int = Field(default=-1, description="Number of layers to offload to GPU (-1 for all possible layers)")
    LLAMA_CPP_N_BATCH: int = Field(default=2048, description="Batch size for prompt processing (higher = faster but more VRAM)")
    LLAMA_CPP_FLASH_ATTN: bool = Field(default=True, description="Enable flash attention for memory efficiency")

    # --- UI/Interaction Settings --- #
    VERBOSE: bool = Field(default=False, description="Verbose mode for debugging")
    DEBUG: bool = Field(default=False, description="Enable raw debug logging output")
    DEBUG_TURN_LOG: bool = Field(default=False, description="Write debug turn log to .logs/debug_turn.txt on each query")
    PLAIN_OUTPUT: bool = Field(default=False, description="Use plain text output without Rich formatting")
    NO_STREAM: bool = Field(default=False, description="Disable streaming output")
    INTERACTIVE_MODE: bool = Field(default=False, description="Whether the app is in interactive mode (set based on args)")

    # --- Nextcloud Talk --- #
    # Per-bot Talk secrets live in bot_profiles (DB), never in env.
    NEXTCLOUD_URL: str = Field(default="https://nextcloud.example.com", description="Nextcloud instance URL")

    # --- Nextcloud Talk Provisioner --- #
    TALK_PROVISIONER_URL: str = Field(default="http://localhost:8790", description="Nextcloud Talk provisioner service URL")
    TALK_PROVISIONER_TOKEN: Optional[str] = Field(default=None, description="Bearer token for provisioner service")

    # SYSTEM_MESSAGE is set at runtime from bot config, this is just the default
    SYSTEM_MESSAGE: str = Field(default=DEFAULT_SYSTEM_MESSAGE)

    defined_models: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    model_catalog: Any = Field(default=None, exclude=True)
    available_ollama_models: List[str] = Field(default_factory=list, exclude=True)
    ollama_checked: bool = Field(default=False, exclude=True)

    model_config = SettingsConfigDict(
        env_prefix="LLM_BAWT_",
        env_file=DOTENV_PATH,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra='ignore'
    )

    def __init__(self, **values: Any):
        super().__init__(**values)
        # Models are loaded exclusively from the database.
        # merge_db_models() is called by the service / ModelManager on startup.
        self.defined_models = {"models": {}}
        self.model_catalog = None

    def install_model_catalog(self, catalog: Any) -> None:
        """Install the normalized endpoint catalog and its iteration shim."""
        self.model_catalog = catalog
        self.defined_models = {"models": catalog.compatibility_mapping()}

    def ensure_model_catalog(self) -> Any:
        """Lazy-load the normalized catalog for CLI/non-service callers."""
        if self.model_catalog is not None:
            return self.model_catalog
        if not has_database_credentials(self):
            return None
        try:
            from ..model_catalog import ModelCatalogStore
            from .db import get_shared_engine

            engine = get_shared_engine(self)
            if engine is None:
                return None
            catalog = ModelCatalogStore(engine).load()
            if len(catalog):
                self.install_model_catalog(catalog)
            return self.model_catalog
        except Exception as exc:
            logger.debug("Normalized model catalog unavailable: %s", exc)
            return None

    def resolve_model(
        self,
        ref: str | int | Any,
        harness: str | None = None,
        default: Any = None,
    ) -> Any:
        """Resolve a model/endpoint ref, with legacy virtual-model fallback."""
        catalog = self.ensure_model_catalog()
        if catalog is not None:
            from ..model_catalog import ModelNotFoundError

            try:
                return catalog.resolve(ref, harness=harness)
            except ModelNotFoundError:
                pass
        return self.defined_models.get("models", {}).get(str(ref), default)

    def merge_db_models(self, db_models: dict[str, Any]) -> None:
        """Load model definitions from the database.

        Despite the name (kept for API compatibility), this is now the
        primary — and only — source for model definitions.
        """
        if not db_models:
            return
        models = dict(self.defined_models.get("models", {}))
        models.update(db_models)
        self.defined_models = {"models": models}

    def _check_ollama_availability(self, force_check: bool = False) -> None:
        if not force_check and self.ollama_checked:
            return

        try:
            import requests # Import lazily
            start_time = time.time()
            response = requests.get(f"{self.OLLAMA_URL}/api/tags", timeout=5)
            response.raise_for_status()
            available_models = {model['name'] for model in response.json().get('models', [])}
            self.available_ollama_models = list(available_models)
            self.ollama_checked = True
            logger.debug(f"Ollama API query took {time.time() - start_time:.2f} ms and found {len(available_models)} models")
        except Exception as e:
            self.available_ollama_models = []
            self.ollama_checked = True
            logger.warning(f"Ollama API check failed: {str(e)}")

    def force_ollama_check(self) -> None:
        self.ollama_checked = False
        self._check_ollama_availability(force_check=True)

    def get_model_options(self) -> List[str]:
        available_options = []
        defined = self.defined_models.get("models", {})
        for alias, model_info in defined.items():
            model_type = model_info.get("type")
            if model_type == PROVIDER_OPENAI:
                available_options.append(alias)
            elif model_type == PROVIDER_GROK:
                # Credential availability is resolved when GrokClient is built
                # (explicit model key → CredentialStore → legacy env/config).
                # Catalog presence determines availability here so DB-only keys
                # are not hidden by an obsolete env check.
                available_options.append(alias)
            elif model_type == PROVIDER_OLLAMA:
                # Do not probe the Ollama server during normal execution.
                # If the alias is defined, treat it as selectable and let actual
                # Ollama usage/install flows perform connectivity/model checks.
                model_id = model_info.get("model_id")
                if model_id:
                    available_options.append(alias)
            elif model_type == PROVIDER_GGUF:
                if is_llama_cpp_available():
                    available_options.append(alias)
            elif model_type == PROVIDER_HF:
                if is_huggingface_available():
                    available_options.append(alias)
            elif model_type == PROVIDER_VLLM:
                if is_vllm_available():
                    available_options.append(alias)
            elif model_type == PROVIDER_AGENT_BACKEND:
                available_options.append(alias)

        return sorted(list(set(available_options))) # Return sorted list of unique aliases

    def get_model_context_window(self, model_alias: str | None = None) -> int:
        """Get the effective context window for a model.

        Resolution order (TASK-602 — model-driven, no type-based guessing, no env):
        1. Per-model catalog `context_window` (model_endpoints.context_window_override
           -> models.default_context_window, surfaced by resolve_model()).
        2. The global ``model_context_window_default`` runtime setting (global
           runtime_settings row -> its declared default in SETTING_DEFINITIONS).

        The catalog is the source of truth for per-model windows; the global
        default only catches models whose catalog window is NULL. There is NO
        type allow-list and NO LLAMA_CPP_N_CTX fallback here — LLAMA_CPP_N_CTX is
        a *local GGUF inference* knob (utils/vram.py at load time), not the
        general model window, and folding it in here is exactly what starved
        claude-code bots (8192 window -> 4096 budget -> zero summaries).
        """
        model_def = self.resolve_model(model_alias or "", default={})
        def_val = model_def.get("context_window")
        if def_val is not None:
            return int(def_val)
        from ..runtime_setting_resolution import resolve_global_runtime_setting
        return int(resolve_global_runtime_setting(self, "model_context_window_default") or 128000)

    def get_model_max_tokens(self, model_alias: str | None = None) -> int:
        """Get the effective max output tokens for a model.

        Resolution order (TASK-602 — settings-driven, no env):
        1. Per-model catalog `max_tokens`.
        2. The global ``max_output_tokens`` runtime setting (global
           runtime_settings row -> its declared default). Retires the direct
           read of the ``MAX_OUTPUT_TOKENS`` BaseSettings env field.
        """
        model_def = self.resolve_model(model_alias or "", default={})
        def_val = model_def.get("max_tokens")
        if def_val is not None:
            return int(def_val)
        from ..runtime_setting_resolution import resolve_global_runtime_setting
        return int(resolve_global_runtime_setting(self, "max_output_tokens") or 4096)

    def resolve_context_budget(
        self, model_alias: str | None = None
    ) -> tuple[int, int, int]:
        """THE single Tier-2 budget authority (TASK-602/609).

        Returns ``(context_window, effective_reserve, prompt_budget)``:
        - ``context_window`` — the model's ceiling (catalog per-model -> global
          ``model_context_window_default``), via ``get_model_context_window``.
        - ``effective_reserve`` — ``min(request_output_reserve, model max-output
          capability)``. The reserve is the centrally configured, GLOBAL-ONLY
          policy for how much of the window to hold back for the reply; it is
          bounded by what the model can actually emit.
        - ``prompt_budget`` — ``max(0, context_window - effective_reserve)`` (0
          when the window is non-positive). What is left for system prompt +
          history + memory.

        This is the ONLY place a prompt budget is computed. Every chat, seed,
        client, and summarization caller routes through it — no duplicated
        ``window - max_output`` math, no hardcoded/type-based windows, no env
        fallbacks. ``request_output_reserve`` resolves to its declared default
        (4096) when no global row exists; bot rows are ignored (global-only).
        """
        context_window = self.get_model_context_window(model_alias)
        max_output_capability = self.get_model_max_tokens(model_alias)
        from ..runtime_setting_resolution import resolve_global_runtime_setting
        reserve_setting = int(
            resolve_global_runtime_setting(self, "request_output_reserve")
        )
        effective_reserve = min(reserve_setting, max_output_capability)
        if context_window <= 0:
            return (context_window, effective_reserve, 0)
        return (
            context_window,
            effective_reserve,
            max(0, context_window - effective_reserve),
        )

    def resolve_summarization_job(self) -> dict:
        """THE single Tier-1 summarization-job authority (TASK-602/610).

        Returns the global ``summarization_job`` dict with every key guaranteed
        present: ``session_gap_seconds``, ``min_messages_per_session``,
        ``protected_recent_turns``, ``trigger_tokens``, ``model``.

        Global-only (Tier 1 is not bot-aware): resolved via
        ``resolve_global_runtime_setting`` (bot rows ignored). The canonical CODE
        defaults are merged UNDER any stored dict, so a partial/absent row still
        yields a complete dict and no legacy env field is read. ``model`` is
        ``None`` by default => callers fall through to ``maintenance_model``.
        """
        from ..runtime_setting_resolution import resolve_global_runtime_setting
        from ..setting_definitions import SUMMARIZATION_JOB_DEFAULTS

        merged = dict(SUMMARIZATION_JOB_DEFAULTS)
        stored = resolve_global_runtime_setting(self, "summarization_job")
        if isinstance(stored, dict):
            merged.update({k: v for k, v in stored.items() if k in merged})
        return merged

    def get_model_n_gpu_layers(self, model_alias: str | None = None) -> int:
        """Get the effective n_gpu_layers for a model.

        Resolution order:
        1. Per-model `n_gpu_layers` in model definition
        2. Global LLAMA_CPP_N_GPU_LAYERS config
        3. Hardcoded default: -1 (all layers)
        """
        model_def = self.resolve_model(model_alias or "", default={})
        def_val = model_def.get("n_gpu_layers")
        if def_val is not None:
            return int(def_val)
        return self.LLAMA_CPP_N_GPU_LAYERS if self.LLAMA_CPP_N_GPU_LAYERS is not None else -1

    def get_tool_format(self, model_alias: str | None = None, model_def: Dict[str, Any] | None = None) -> str:
        """Return the tool format for a given model alias or definition.
        
        Resolution order:
        1. Explicit tool_support field (native|react|none)
        2. Legacy tool_format override (deprecated; XML maps to ReAct)
        3. Auto-detect by provider type
        """
        if model_def is None and model_alias:
            model_def = self.resolve_model(model_alias, default={})
        model_def = model_def or {}

        # 1. Explicit tool_support (new). XML was retired with the legacy
        # prompt protocol; old catalog rows degrade safely to ReAct text tools.
        tool_support = model_def.get("tool_support")
        if tool_support in ("native", "react", "none"):
            return tool_support
        if tool_support == "xml":
            return "react"

        # 2. Legacy tool_format override (existing)
        tool_format = str(model_def.get("tool_format") or "").strip().lower()
        if tool_format:
            return "react" if tool_format == "xml" else tool_format

        # 3. Auto-detect by provider type (existing + vLLM)
        model_type = model_def.get("type")
        if model_type == PROVIDER_OPENAI:
            return "native"
        if model_type == PROVIDER_GROK:
            return "native"  # Grok supports native tool calling
        if model_type == PROVIDER_VLLM:
            return "native"  # tools passed via chat template, parsed from output
        if model_type == PROVIDER_AGENT_BACKEND:
            return "none"
        if model_type in (PROVIDER_GGUF, PROVIDER_OLLAMA, PROVIDER_HF):
            return "react"
        return "react"


def set_config_value(key: str, value: str, config: Config) -> bool:
    """Sets a configuration value in the .env file.

    Args:
        key: The configuration key (e.g., 'DEFAULT_MODEL_ALIAS').
        value: The value to set.
        config: The loaded Config object to get the .env path and validate keys.

    Returns:
        True if successful, False otherwise.
    """
    dotenv_path = Path(config.model_config['env_file'])
    key_upper = key.upper()
    env_var_name = f"{config.model_config['env_prefix']}{key_upper}"
    valid_keys = {k.upper() for k in Config.model_fields.keys()}
    if key_upper not in valid_keys:
        console.print(f"[bold red]Error:[/bold red] Invalid configuration key '{key}'. Valid keys are: {', '.join(sorted(Config.model_fields.keys()))}")
        return False

    lines = []
    found = False
    try:
        if dotenv_path.is_file():
            with open(dotenv_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
    except IOError as e:
        console.print(f"[bold red]Error reading {dotenv_path}:[/bold red] {e}")
        return False
    new_line = f"{env_var_name}={value}\n"
    updated_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith(f"{env_var_name}="):
            updated_lines.append(new_line)
            found = True
        else:
            updated_lines.append(line) # Keep existing line

    if not found:
        updated_lines.append(new_line)
    try:
        dotenv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dotenv_path, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)
        return True
    except IOError as e:
        console.print(f"[bold red]Error writing to {dotenv_path}:[/bold red] {e}")
        return False


def has_database_credentials(cfg: Config | None = None) -> bool:
    """Check if database credentials are configured.
    
    Returns True if POSTGRES_PASSWORD is set (non-empty string).
    Other credentials have defaults so password is the key indicator.
    
    Args:
        cfg: Config instance to check. If None, uses the global config.
    """
    if cfg is None:
        cfg = config
    return bool(cfg.POSTGRES_PASSWORD and cfg.POSTGRES_PASSWORD.strip())


config = Config()
