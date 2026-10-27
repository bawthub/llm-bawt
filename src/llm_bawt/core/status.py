"""Shared system status collection used by both CLI and API."""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm_bawt.utils.config import Config

logger = logging.getLogger(__name__)


@dataclass
class ServiceInfo:
    """LLM service status."""

    available: bool = False
    healthy: bool = False
    uptime_seconds: float | None = None
    current_model: str | None = None
    default_model: str | None = None
    default_bot: str | None = None
    tasks_processed: int = 0
    tasks_pending: int = 0


@dataclass
class MemoryInfo:
    """Database and memory subsystem status."""

    postgres_connected: bool = False
    postgres_host: str | None = None
    postgres_error: str | None = None
    messages_count: int = 0
    memories_count: int = 0
    pgvector_available: bool = False
    embeddings_available: bool = False


@dataclass
class ModelStatusInfo:
    """Current model configuration details."""

    alias: str
    type: str = "unknown"
    max_tokens: int = 0
    max_tokens_source: str = "global"
    context_window: int | None = None
    context_source: str | None = None
    # GGUF-specific
    gpu_name: str | None = None
    vram_total_gb: float | None = None
    vram_free_gb: float | None = None
    vram_detection_method: str | None = None
    n_gpu_layers: str | None = None
    gpu_layers_source: str | None = None
    native_context_limit: int | None = None


@dataclass
class DependencyInfo:
    """Optional dependency availability."""

    cuda_version: str | None = None
    llama_cpp_available: bool = False
    llama_cpp_gpu: bool | None = None
    hf_hub_available: bool = False
    torch_available: bool = False
    openai_key_set: bool = False
    newsapi_key_set: bool = False
    search_provider: str | None = None
    embeddings_available: bool = False


@dataclass
class McpInfo:
    """llm-bawt MCP server status."""

    mode: str = "embedded"  # "server" | "embedded"
    status: str = "up"  # "up" | "down" | "error"
    url: str | None = None
    http_status: int | None = None


@dataclass
class BotSummary:
    """Minimal bot info for the status display."""

    slug: str
    name: str
    is_default: bool = False


@dataclass
class ConfigInfo:
    """System configuration summary."""

    version: str = "0.1.0"
    mode: str = "direct"  # "service" | "direct"
    service_url: str | None = None
    environment: str = "local"  # "docker" | "local"
    bot_name: str = ""
    bot_slug: str = ""
    model_alias: str | None = None
    model_source: str | None = None
    user_id: str | None = None
    all_bots: list[BotSummary] = field(default_factory=list)
    models_defined: int = 0
    models_service: int | None = None
    scheduler_enabled: bool = False
    scheduler_interval: int = 0
    ha_mcp_enabled: bool = False
    ha_mcp_url: str | None = None
    ha_native_mcp_url: str | None = None
    ha_native_mcp_tools: int = 0
    bind_host: str = "0.0.0.0"


@dataclass
class SystemStatus:
    """Full system status."""

    config: ConfigInfo
    service: ServiceInfo
    mcp: McpInfo
    model: ModelStatusInfo | None
    memory: MemoryInfo
    dependencies: DependencyInfo


def _collect_service_info(config: Config, local_only: bool = False) -> tuple[ServiceInfo, Any]:
    """Collect LLM service status. Returns (ServiceInfo, raw_service_status)."""
    from llm_bawt.model_manager import is_service_mode_enabled

    info = ServiceInfo()
    raw_status = None
    use_service = is_service_mode_enabled(config) and not local_only

    if not use_service:
        return info, raw_status

    try:
        from llm_bawt.service import ServiceClient

        host = "127.0.0.1" if config.SERVICE_HOST in {"0.0.0.0", "::"} else config.SERVICE_HOST
        service_url = f"http://{host}:{config.SERVICE_PORT}"
        client = ServiceClient(http_url=service_url)
        if client.is_available(force_check=True):
            info.available = True
            try:
                raw_status = client.get_status(silent=True)
                if raw_status and getattr(raw_status, "available", False):
                    info.healthy = True
                    info.uptime_seconds = raw_status.uptime_seconds
                    info.current_model = raw_status.current_model
                    info.tasks_processed = raw_status.tasks_processed
                    info.tasks_pending = raw_status.tasks_pending
            except Exception:
                pass
    except Exception:
        pass

    return info, raw_status


def _collect_mcp_info(config: Config, default_bot_slug: str) -> McpInfo:
    """Collect llm-bawt MCP server status."""
    mcp_server_url = (config.MCP_SERVER_URL or "").strip()
    if not mcp_server_url:
        return McpInfo(mode="embedded", status="up", url=None)

    mcp_url = mcp_server_url.rstrip("/")
    info = McpInfo(mode="server", url=mcp_url)

    try:
        import httpx

        probe_payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "get_messages",
                "arguments": {
                    "bot_id": default_bot_slug,
                    "since_seconds": 60,
                    "limit": 1,
                },
            },
            "id": "status-probe",
        }
        response = httpx.post(
            f"{mcp_url}/mcp",
            json=probe_payload,
            headers={"Accept": "application/json, text/event-stream"},
            timeout=2.0,
        )
        if response.status_code == 200:
            info.status = "up"
        else:
            info.status = "error"
            info.http_status = response.status_code
    except Exception:
        info.status = "down"

    return info


def _collect_memory_info(config: Config, default_bot_slug: str) -> MemoryInfo:
    """Collect database and memory status (service-side only).

    Only called from the service process (BackgroundService.get_status() and
    collect_system_status()). CLI gets memory info via _status_from_service().
    """
    from llm_bawt.utils.config import has_database_credentials

    info = MemoryInfo()
    info.embeddings_available = importlib.util.find_spec("sentence_transformers") is not None

    if not has_database_credentials(config):
        return info

    try:
        from llm_bawt.memory.postgresql import PostgreSQLMemoryBackend

        backend = PostgreSQLMemoryBackend(
            config,
            bot_id=default_bot_slug,
            provision_schema=False,
        )
        db_stats = backend.stats()
        info.postgres_connected = True
        info.postgres_host = config.POSTGRES_HOST
        info.memories_count = db_stats.get("memories", {}).get("total_count", 0)
        info.messages_count = db_stats.get("messages", {}).get("total_count", 0)
    except Exception as e:
        info.postgres_error = str(e)

    try:
        from pgvector.psycopg2 import register_vector  # noqa: F401

        info.pgvector_available = True
    except ImportError:
        pass

    return info


def _collect_model_info(config: Config, model_alias: str) -> ModelStatusInfo | None:
    """Collect current model details."""
    defined_models = config.defined_models.get("models", {})
    if model_alias not in defined_models:
        return None

    model_def = config.resolve_model(model_alias)
    model_type = model_def.get("type", "unknown")

    effective_max_tokens = model_def.get("max_tokens", config.get_model_max_tokens(model_alias))
    max_tokens_source = "model" if "max_tokens" in model_def else "global"

    info = ModelStatusInfo(
        alias=model_alias,
        type=model_type,
        max_tokens=effective_max_tokens,
        max_tokens_source=max_tokens_source,
    )

    if model_type == "gguf":
        from llm_bawt.utils.vram import auto_size_context_window, detect_vram

        vram_info = detect_vram()
        if vram_info:
            info.gpu_name = vram_info.gpu_name
            info.vram_total_gb = vram_info.total_gb
            info.vram_free_gb = vram_info.free_gb
            info.vram_detection_method = vram_info.detection_method

        sizing = auto_size_context_window(
            model_definition=model_def,
            global_n_ctx=config.LLAMA_CPP_N_CTX,
            global_max_tokens=effective_max_tokens,
        )
        info.context_window = sizing.context_window
        info.context_source = sizing.source

        n_gpu_layers = model_def.get("n_gpu_layers", config.LLAMA_CPP_N_GPU_LAYERS)
        info.gpu_layers_source = "model" if "n_gpu_layers" in model_def else "global"
        info.n_gpu_layers = "all" if n_gpu_layers == -1 else str(n_gpu_layers)

        native_limit = model_def.get("native_context_limit")
        if native_limit:
            info.native_context_limit = int(native_limit)
    else:
        ctx_window = model_def.get("context_window")
        if ctx_window:
            info.context_window = int(ctx_window)
            info.context_source = "model"
        else:
            # TASK-609: catalog per-model -> global default (no hardcoded 128000,
            # no type allow-list) — matches config.get_model_context_window.
            info.context_window = config.get_model_context_window(model_alias)
            info.context_source = "default"

    return info


def _collect_dependencies(config: Config) -> DependencyInfo:
    """Collect optional dependency availability."""
    from llm_bawt.utils.config import is_huggingface_available, is_llama_cpp_available

    info = DependencyInfo()

    # CUDA — try multiple detection strategies so we work both with and
    # without the CUDA toolkit (nvcc) in the PATH (common in runtime-only
    # Docker images).
    def _detect_cuda() -> str | None:
        # 1. PyTorch reports CUDA version when compiled with GPU support
        try:
            import torch  # type: ignore[import]
            cuda_ver = torch.version.cuda  # type: ignore[attr-defined]
            if cuda_ver:
                return cuda_ver
        except Exception:
            pass

        # 2. CUDA_VERSION env var (set by NVIDIA base images)
        cuda_env = os.environ.get("CUDA_VERSION") or os.environ.get("CUDA_VERSION_TAG")
        if cuda_env:
            return cuda_env.split("-")[0]  # strip suffixes like "12.4.0-rc"

        # 3. nvcc (full toolkit installed)
        nvcc_path = shutil.which("nvcc")
        if nvcc_path:
            try:
                result = subprocess.run(
                    ["nvcc", "--version"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    match = re.search(r"release ([\d.]+)", result.stdout)
                    if match:
                        return match.group(1)
            except Exception:
                pass

        # 4. nvidia-smi (driver installed, no toolkit)
        smi_path = shutil.which("nvidia-smi")
        if smi_path:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return f"driver {result.stdout.strip().splitlines()[0]}"
            except Exception:
                pass

        # 5. /usr/local/cuda/version.txt or version.json
        for path in ("/usr/local/cuda/version.txt", "/usr/local/cuda-12/version.txt"):
            try:
                with open(path) as fh:
                    line = fh.readline().strip()
                    match = re.search(r"([\d]+\.[\d.]+)", line)
                    if match:
                        return match.group(1)
            except OSError:
                pass
        try:
            import json as _json
            with open("/usr/local/cuda/version.json") as fh:
                data = _json.load(fh)
                ver = data.get("cuda", {}).get("version")
                if ver:
                    return ver
        except Exception:
            pass

        return None

    info.cuda_version = _detect_cuda()

    # llama-cpp-python
    info.llama_cpp_available = is_llama_cpp_available()
    if info.llama_cpp_available:
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            old_stdout_fd = os.dup(1)
            old_stderr_fd = os.dup(2)
            try:
                os.dup2(devnull, 1)
                os.dup2(devnull, 2)
                from llama_cpp import llama_supports_gpu_offload

                info.llama_cpp_gpu = llama_supports_gpu_offload()
            finally:
                os.dup2(old_stdout_fd, 1)
                os.dup2(old_stderr_fd, 2)
                os.close(devnull)
                os.close(old_stdout_fd)
                os.close(old_stderr_fd)
        except (ImportError, AttributeError, OSError):
            pass

    # huggingface-hub
    info.hf_hub_available = importlib.util.find_spec("huggingface_hub") is not None

    # torch
    info.torch_available = is_huggingface_available()

    # OpenAI
    info.openai_key_set = bool(os.getenv("OPENAI_API_KEY"))

    # NewsAPI
    from ..search.credentials import resolve_key  # noqa: PLC0415

    def _key_set(provider_id: str) -> bool:
        try:
            return bool(resolve_key(config, provider_id))
        except Exception as e:  # noqa: BLE001 — status must render without the store
            logger.debug("search key check failed for %s: %s", provider_id, e)
            return False

    info.newsapi_key_set = _key_set("newsapi")

    # Search provider
    search_provider = getattr(config, "SEARCH_PROVIDER", None)
    tavily_key = _key_set("tavily")
    brave_key = _key_set("brave")
    reddit_client_id = os.getenv("REDDIT_CLIENT_ID") or getattr(config, "REDDIT_CLIENT_ID", "")
    reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET") or getattr(config, "REDDIT_CLIENT_SECRET", "")
    if search_provider:
        info.search_provider = search_provider
    elif tavily_key:
        info.search_provider = "tavily"
    elif brave_key:
        info.search_provider = "brave"
    elif reddit_client_id and reddit_client_secret:
        info.search_provider = "reddit"
    else:
        if importlib.util.find_spec("ddgs"):
            info.search_provider = "duckduckgo"

    # Embeddings
    info.embeddings_available = importlib.util.find_spec("sentence_transformers") is not None

    return info


def _status_from_service(config: Config) -> SystemStatus | None:
    """Try to fetch full system status from the running service.

    Returns a ``SystemStatus`` if the service is reachable, otherwise ``None``.
    """
    from llm_bawt.service.client import get_service_client

    client = get_service_client(config)
    data = client.get_full_status()
    if data is None:
        return None

    try:
        cfg = data.get("config", {})
        svc = data.get("service", {})
        mcp = data.get("mcp", {})
        mdl = data.get("model")
        mem = data.get("memory", {})
        dep = data.get("dependencies", {})

        model_source = cfg.get("model_source")
        if not model_source:
            try:
                from llm_bawt.bots import BotManager

                cfg_bot_slug = cfg.get("bot_slug")
                cfg_model_alias = cfg.get("model_alias")
                if cfg_bot_slug and cfg_model_alias:
                    inferred = BotManager(config).select_model(None, bot_slug=cfg_bot_slug)
                    if inferred.alias == cfg_model_alias:
                        model_source = inferred.source
                    else:
                        model_source = "service_runtime"
            except Exception:
                model_source = None

        config_info = ConfigInfo(
            version=cfg.get("version", "0.1.0"),
            mode=cfg.get("mode", "service"),
            service_url=cfg.get("service_url"),
            environment=cfg.get("environment", "local"),
            bot_name=cfg.get("bot_name", ""),
            bot_slug=cfg.get("bot_slug", ""),
            model_alias=cfg.get("model_alias"),
            model_source=model_source,
            user_id=cfg.get("user_id"),
            all_bots=[
                BotSummary(slug=b["slug"], name=b["name"], is_default=b.get("is_default", False))
                for b in cfg.get("all_bots", [])
            ],
            models_defined=cfg.get("models_defined", 0),
            models_service=cfg.get("models_service"),
            scheduler_enabled=cfg.get("scheduler_enabled", False),
            scheduler_interval=cfg.get("scheduler_interval", 0),
            ha_mcp_enabled=cfg.get("ha_mcp_enabled", False),
            ha_mcp_url=cfg.get("ha_mcp_url"),
            ha_native_mcp_url=cfg.get("ha_native_mcp_url"),
            ha_native_mcp_tools=cfg.get("ha_native_mcp_tools", 0),
            bind_host=cfg.get("bind_host", "0.0.0.0"),
        )

        service_default_model = svc.get("default_model") or cfg.get("model_alias")
        service_default_bot = svc.get("default_bot") or cfg.get("bot_slug")

        service_info = ServiceInfo(
            available=svc.get("available", False),
            healthy=svc.get("healthy", False),
            uptime_seconds=svc.get("uptime_seconds"),
            current_model=svc.get("current_model"),
            default_model=service_default_model,
            default_bot=service_default_bot,
            tasks_processed=svc.get("tasks_processed", 0),
            tasks_pending=svc.get("tasks_pending", 0),
        )

        mcp_info = McpInfo(
            mode=mcp.get("mode", "embedded"),
            status=mcp.get("status", "up"),
            url=mcp.get("url"),
            http_status=mcp.get("http_status"),
        )

        model_info = None
        if mdl:
            model_info = ModelStatusInfo(
                alias=mdl["alias"],
                type=mdl.get("type", "unknown"),
                max_tokens=mdl.get("max_tokens", 0),
                max_tokens_source=mdl.get("max_tokens_source", "global"),
                context_window=mdl.get("context_window"),
                context_source=mdl.get("context_source"),
                gpu_name=mdl.get("gpu_name"),
                vram_total_gb=mdl.get("vram_total_gb"),
                vram_free_gb=mdl.get("vram_free_gb"),
                vram_detection_method=mdl.get("vram_detection_method"),
                n_gpu_layers=mdl.get("n_gpu_layers"),
                gpu_layers_source=mdl.get("gpu_layers_source"),
                native_context_limit=mdl.get("native_context_limit"),
            )

        memory_info = MemoryInfo(
            postgres_connected=mem.get("postgres_connected", False),
            postgres_host=mem.get("postgres_host"),
            postgres_error=mem.get("postgres_error"),
            messages_count=mem.get("messages_count", 0),
            memories_count=mem.get("memories_count", 0),
            pgvector_available=mem.get("pgvector_available", False),
            embeddings_available=mem.get("embeddings_available", False),
        )

        dep_info = DependencyInfo(
            cuda_version=dep.get("cuda_version"),
            llama_cpp_available=dep.get("llama_cpp_available", False),
            llama_cpp_gpu=dep.get("llama_cpp_gpu"),
            hf_hub_available=dep.get("hf_hub_available", False),
            torch_available=dep.get("torch_available", False),
            openai_key_set=dep.get("openai_key_set", False),
            newsapi_key_set=dep.get("newsapi_key_set", False),
            search_provider=dep.get("search_provider"),
            embeddings_available=dep.get("embeddings_available", False),
        )

        return SystemStatus(
            config=config_info,
            service=service_info,
            mcp=mcp_info,
            model=model_info,
            memory=memory_info,
            dependencies=dep_info,
        )
    except Exception as e:
        logger.warning("Failed to parse service status response: %s", e)
        return None


def collect_system_status(
    config: Config,
    bot_slug: str | None = None,
    model_alias: str | None = None,
    user_id: str | None = None,
    *,
    local_only: bool = False,
) -> SystemStatus:
    """Collect full system status.

    This is the shared entry point used by both the CLI ``show_status()`` and
    the ``/v1/status`` API endpoint.

    When the CLI is running in service mode, the status is fetched from the
    service's ``/v1/status`` endpoint so the CLI never needs direct DB or
    dependency access.

    Args:
        config: Application configuration.
        bot_slug: Explicit bot slug override (CLI ``-b`` flag).
        model_alias: Explicit model alias override (CLI ``-m`` flag).
        user_id: Explicit user override (CLI ``--user`` flag).
    """
    from llm_bawt.bots import BotManager
    from llm_bawt.model_manager import is_service_mode_enabled

    use_service = is_service_mode_enabled(config) and not local_only

    # In service mode, delegate entirely to the service.
    if use_service:
        remote = _status_from_service(config)
        if remote is not None:
            return remote
        # Service unreachable — fall through to local collection so we can
        # at least show something (with the service marked as unavailable).

    bot_manager = BotManager(config, local_only=local_only)
    default_bot = bot_manager.get_default_bot()

    # Resolve effective bot
    if bot_slug:
        target_bot = bot_manager.get_bot(bot_slug) or default_bot
    else:
        target_bot = default_bot

    # Resolve effective model
    if model_alias is None:
        selection = bot_manager.select_model(None, bot_slug=target_bot.slug)
    else:
        selection = bot_manager.select_model(model_alias, bot_slug=target_bot.slug)

    model_alias = selection.alias

    # Resolve user
    effective_user = user_id or config.DEFAULT_USER

    # Service
    effective_host = "127.0.0.1" if config.SERVICE_HOST in {"0.0.0.0", "::"} else config.SERVICE_HOST
    service_url = f"http://{effective_host}:{config.SERVICE_PORT}" if use_service else None

    service_info, raw_status = _collect_service_info(config, local_only=local_only)

    # Version
    version = "0.1.0"
    if raw_status and getattr(raw_status, "version", None):
        version = raw_status.version

    # Environment
    is_docker = os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")

    # Bot summaries
    all_bots = [
        BotSummary(slug=b.slug, name=b.name, is_default=(b.slug == default_bot.slug))
        for b in bot_manager.list_bots()
    ]

    # Model catalog counts
    defined_models = config.defined_models.get("models", {})
    models_service = None
    if raw_status and getattr(raw_status, "available_models", None):
        models_service = len(raw_status.available_models)

    config_info = ConfigInfo(
        version=version,
        mode="service" if use_service else "direct",
        service_url=service_url,
        environment="docker" if is_docker else "local",
        bot_name=target_bot.name,
        bot_slug=target_bot.slug,
        model_alias=model_alias,
        model_source=selection.source,
        user_id=effective_user,
        all_bots=all_bots,
        models_defined=len(defined_models),
        models_service=models_service,
        scheduler_enabled=config.SCHEDULER_ENABLED,
        scheduler_interval=config.SCHEDULER_CHECK_INTERVAL_SECONDS,
        ha_mcp_enabled=getattr(config, "HA_MCP_ENABLED", False),
        ha_mcp_url=getattr(config, "HA_MCP_URL", None),
        ha_native_mcp_url=getattr(config, "HA_NATIVE_MCP_URL", None) or None,
        bind_host=config.SERVICE_HOST,
    )

    mcp_info = _collect_mcp_info(config, default_bot.slug)

    model_info = None
    if model_alias:
        model_info = _collect_model_info(config, model_alias)

    memory_info = _collect_memory_info(config, default_bot.slug)
    dep_info = _collect_dependencies(config)

    return SystemStatus(
        config=config_info,
        service=service_info,
        mcp=mcp_info,
        model=model_info,
        memory=memory_info,
        dependencies=dep_info,
    )
