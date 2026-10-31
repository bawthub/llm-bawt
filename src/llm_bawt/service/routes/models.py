"""Model and bot listing routes."""

from fastapi import APIRouter, HTTPException, Query

from ...bots import BotManager, bot_registry_error
from ...steer_capabilities import bot_steer_caps
from ...voice_resolution import global_default_voice, resolve_bot_voice
from ..dependencies import get_service
from ..providers.registry import access_path_connections, route_eligibility
from ..schemas import (
    BotInfo,
    BotsResponse,
    ModelDetail,
    ModelInfo,
    ModelPricing,
    ModelsResponse,
    ModelSwitchRequest,
    ModelSwitchResponse,
)

router = APIRouter()


def _upstream_lookup(provider: str, config=None) -> list[dict]:
    """Fetch one provider's current normalized catalog without stale fallback."""
    from ..model_discovery import ModelDiscoveryError, discover_models

    try:
        return (
            discover_models(provider, config)
            if config is not None
            else discover_models(provider)
        )
    except ModelDiscoveryError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


def _coerce_pricing(raw: object) -> ModelPricing | None:
    """Best-effort coerce a stored ``extra['pricing']`` blob into ModelPricing.

    Pricing is user-entered JSONB, so a malformed value must never wedge the
    whole /v1/models listing — return None on anything unparseable, and keep
    only the known numeric rate keys.
    """
    if not isinstance(raw, dict):
        return None
    fields = ("input", "output", "cache_read", "cache_write")
    rates: dict[str, float] = {}
    for key in fields:
        val = raw.get(key)
        if val is None or isinstance(val, bool):
            continue
        try:
            rates[key] = float(val)
        except (TypeError, ValueError):
            continue
    return ModelPricing(**rates) if rates else None


def _public_model_type(info: dict | None) -> str | None:
    """Return the UI-facing provider type for a model definition."""
    if not info:
        return None
    model_type = info.get("type")
    if model_type == "agent_backend" and str(info.get("backend") or "").lower() == "codex":
        return "codex"
    return model_type


@router.get("/v1/models", response_model=ModelsResponse, tags=["OpenAI Compatible"])
def list_models():
    """List available models (OpenAI-compatible)."""
    service = get_service()
    defined = service.config.defined_models.get("models", {})
    infos = {alias: defined.get(alias, {}) for alias in service._available_models}
    # TASK-1302 A3: each alias names the route it rides and that route's
    # eligibility, so alias-valued pickers show API key vs subscription and
    # connection state from the same rule the bot save enforces.
    connections = access_path_connections(
        service.config,
        sorted({info["access_path"] for info in infos.values() if info.get("access_path")}),
    )
    models = []
    for alias, info in infos.items():
        path_key = info.get("access_path")
        models.append(ModelInfo(
            id=alias,
            type=_public_model_type(info),
            model_id=info.get("model_id"),
            description=info.get("description"),
            pricing=_coerce_pricing(info.get("pricing")),
            endpoint_id=info.get("endpoint_id"),
            access_path=path_key,
            eligibility=(
                route_eligibility(path_key, info.get("auth_mechanism"), connections[path_key])
                if path_key else None
            ),
        ))
    return ModelsResponse(data=models)


@router.get("/v1/models/upstream", tags=["Models"])
def list_upstream_models(
    provider: str = Query(
        ...,
        description="Provider catalog: openai | codex | grok | anthropic | kimi",
    ),
):
    """List available models from a provider's upstream catalog.

    Used by the Add Model dialog so users can pick from real model IDs instead
    of typing blind. Codex is bridge-backed; OpenAI, Grok, Anthropic, and Kimi
    use their current provider catalogs. Discovery failures surface directly.

    Returns ``{provider, models: [{id, description, context_length?}]}``.
    """
    provider_key = (provider or "").strip().lower()
    if not provider_key:
        raise HTTPException(status_code=400, detail="provider is required")
    return {
        "provider": provider_key,
        "models": _upstream_lookup(provider_key, get_service().config),
    }


@router.get("/v1/models/current", tags=["Models"])
def get_current_model():
    """Get the currently active model."""
    service = get_service()
    current = service.model_lifecycle.current_model
    if not current:
        return {"model": None, "message": "No model currently loaded"}
    info = service.model_lifecycle.get_model_info(current)
    detail = ModelDetail(
        id=current,
        type=_public_model_type(info),
        model_id=info.get("model_id", info.get("repo_id")) if info else None,
        description=info.get("description") if info else None,
        current=True,
    )
    return {"model": detail}

@router.post("/v1/models/switch", response_model=ModelSwitchResponse, tags=["Models"])
def switch_model(request: ModelSwitchRequest):
    """Switch to a different model. Takes effect on the next request."""
    service = get_service()
    previous = service.model_lifecycle.current_model
    success, message = service.model_lifecycle.switch_model(request.model)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return ModelSwitchResponse(
        success=True,
        message=message,
        previous_model=previous,
        new_model=request.model,
    )


@router.post("/v1/models/reload", tags=["Models"])
def reload_models_catalog():
    """Reload model catalog from DB and refresh service model availability."""
    service = get_service()
    config = service.config

    from ...memory.model_catalog_migration import migrate_model_catalog
    from ...model_catalog import ModelCatalogStore
    from ..dependencies import get_model_catalog_engine

    engine = get_model_catalog_engine(config)
    if engine is None:
        raise HTTPException(status_code=503, detail="Model catalog database unavailable")
    migrate_model_catalog(engine)
    normalized_catalog = ModelCatalogStore(engine).load()
    config.install_model_catalog(normalized_catalog)

    service._load_available_models()
    cleared = service.invalidate_all_instances()

    return {
        "ok": True,
        "models": list(service._available_models),
        "default_model": service._default_model,
        "catalog_endpoints_loaded": len(normalized_catalog),
        "cleared_instances": cleared,
    }

@router.get("/v1/bots", response_model=BotsResponse, tags=["System"])
def list_bots():
    """List available bots configured on the service."""
    service = get_service()
    bot_manager = BotManager(service.config)
    listed = bot_manager.list_bots()
    if bot_registry_error() and not listed:
        # An unreadable DB is not "no bots": a 200 [] sends clients to first-run
        # setup. The cause (DB error text) stays in the service log.
        raise HTTPException(status_code=503, detail="Bot registry unavailable; retrying")
    global_voice = global_default_voice(service.config)
    bots = []
    for bot in listed:
        voice = resolve_bot_voice(bot.default_voice, global_voice)
        bots.append(BotInfo(
            slug=bot.slug,
            name=bot.name,
            description=bot.description,
            system_prompt=bot.system_prompt,
            prompt_override_id=getattr(bot, "prompt_override_id", None),
            requires_memory=bot.requires_memory,
            voice_optimized=bot.voice_optimized,
            tts_mode=bot.tts_mode,
            include_summaries=bot.include_summaries,
            include_in_global_search=bot.include_in_global_search,
            tags=bot.tags,
            default_voice=bot.default_voice,
            effective_voice=voice.voice,
            effective_voice_source=voice.source,
            uses_tools=bot.uses_tools,
            uses_search=bot.uses_search,
            uses_home_assistant=bot.uses_home_assistant,
            default_model=bot.default_model,
            color=bot.color,
            avatar=bot.avatar,
            avatar_render=getattr(bot, "avatar_render", None),
            bot_type=bot.bot_type,
            harness=bot.harness,
            endpoint_id=bot.endpoint_id,
            agent_backend=bot.agent_backend,
            agent_backend_config=bot.agent_backend_config or {},
            settings=bot.settings,
            steer_capabilities=sorted(bot_steer_caps(bot)),
        ))
    return BotsResponse(data=bots)
