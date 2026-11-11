"""Service status + health response schemas.

Split out of ``service/schemas.py`` (TASK-557). ``schemas.py`` re-imports every
name here so ``from ..schemas import X`` across the service is unchanged.
"""

from __future__ import annotations


from pydantic import BaseModel

# Mirrors the per-module SERVICE_VERSION literal (api.py / background_service.py
# each keep their own copy); used as a default on the status schemas below.
SERVICE_VERSION = "0.1.0"


class ServiceInfoSchema(BaseModel):
    """LLM service status."""
    available: bool = False
    healthy: bool = False
    uptime_seconds: float | None = None
    current_model: str | None = None
    default_model: str | None = None
    default_bot: str | None = None
    tasks_processed: int = 0
    tasks_pending: int = 0

class MemoryInfoSchema(BaseModel):
    """Database and memory subsystem status."""
    postgres_connected: bool = False
    postgres_host: str | None = None
    postgres_error: str | None = None
    messages_count: int = 0
    memories_count: int = 0
    pgvector_available: bool = False
    embeddings_available: bool = False

class ModelStatusInfoSchema(BaseModel):
    """Current model configuration details."""
    alias: str
    type: str = "unknown"
    max_tokens: int = 0
    max_tokens_source: str = "global"
    context_window: int | None = None
    context_source: str | None = None
    gpu_name: str | None = None
    vram_total_gb: float | None = None
    vram_free_gb: float | None = None
    vram_detection_method: str | None = None
    n_gpu_layers: str | None = None
    gpu_layers_source: str | None = None
    native_context_limit: int | None = None

class DependencyInfoSchema(BaseModel):
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

class McpInfoSchema(BaseModel):
    """llm-bawt MCP server status."""
    mode: str = "embedded"
    status: str = "up"
    url: str | None = None
    http_status: int | None = None

class BotSummarySchema(BaseModel):
    """Minimal bot info for the status display."""
    slug: str
    name: str
    is_default: bool = False

class ConfigInfoSchema(BaseModel):
    """System configuration summary."""
    version: str = SERVICE_VERSION
    mode: str = "direct"
    service_url: str | None = None
    environment: str = "local"
    bot_name: str = ""
    bot_slug: str = ""
    model_alias: str | None = None
    model_source: str | None = None
    user_id: str | None = None
    all_bots: list[BotSummarySchema] = []
    models_defined: int = 0
    models_service: int | None = None
    scheduler_enabled: bool = False
    scheduler_interval: int = 0
    ha_mcp_enabled: bool = False
    ha_mcp_url: str | None = None
    ha_native_mcp_url: str | None = None
    ha_native_mcp_tools: int = 0
    bind_host: str = "0.0.0.0"

class SystemStatusResponse(BaseModel):
    """Full system status — mirrors ``core.status.SystemStatus``."""
    config: ConfigInfoSchema
    service: ServiceInfoSchema
    mcp: McpInfoSchema
    model: ModelStatusInfoSchema | None = None
    memory: MemoryInfoSchema
    dependencies: DependencyInfoSchema

class HealthResponse(BaseModel):
    """Simple health check response."""
    status: str = "ok"
    version: str = SERVICE_VERSION
