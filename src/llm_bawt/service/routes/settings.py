"""Runtime settings and bot profile routes.

OpenClaw SOUL.md sync lives in ``bot_soul.py`` (split out under the monolith
gate, TASK-1521).
"""

import json
import logging

from sqlalchemy import func
from sqlmodel import Session, select

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from ... import bot_profile_changes as profile_changes
from ...bot_profile_changes import ProfileFieldsChanged, ProfileRowChanged
from ...bot_tags import BehaviorTag
from ...bot_tags.errors import BotTagError, TagSetChanged
from ...bot_tags.store import BotTagStore
from ...bot_tags.vocabulary import API_ASSIGNER
from ...bot_types import normalize_bot_type
from ...runtime_settings import RuntimeSetting, purge_bot_data, cleanup_orphaned_bot_data
from ...orphan_partitions import PartitionExecutionHalted
from ..providers.base import CREDENTIAL_SETTING_PREFIXES, is_credential_setting_key
from ...working_memory.contracts import WorkingMemoryError
from ...working_memory.policy import WorkingMemoryPolicy
from ..profile_route_gate import ProfileRouteGate, ProfileSaveGuard, RouteIneligible
from ..profile_switch_gate import ProfileSwitchGate
from ..turn_admission import SystemGateRefused
from ..runtime_switch_rotation import (
    RUNTIME_SWITCH_SOURCE,
    rotate_active_threads,
    runtime_family,
    runtime_family_changed,
    thread_switched_event,
)
from ..dependencies import (
    get_bot_profile_store,
    get_prompt_template_store,
    get_runtime_settings_store,
    get_service,
)
from ..schemas import (
    BotCreateRequest,
    BotProfileListResponse,
    BotProfilePatchRequest,
    BotProfileResponse,
    BotProfileUpsertRequest,
    RuntimeSettingRecord,
    RuntimeSettingsListResponse,
    RuntimeSettingItem,
    RuntimeSettingBatchUpsertRequest,
    RuntimeSettingsResponse,
    RuntimeSettingUpsertRequest,
)
from .bot_soul import openclaw_agent_id, push_soul_background as _push_soul_background
from .bot_profile_validation import (
    _humanize_bot_constraint_error,
    _request_to_profile_payload,
    _validate_profile_payload,
)

logger = logging.getLogger(__name__)

router = APIRouter()
RETIRED_BALANCE_SETTINGS = frozenset({"anthropic_api_monthly_budget_usd", "anthropic_api_balance_anchor"})


def _normalize_scope(scope_type: str, scope_id: str | None, default_bot: str) -> tuple[str, str]:
    st = (scope_type or "").strip().lower()
    if st not in {"global", "bot"}:
        raise HTTPException(status_code=400, detail="scope_type must be 'global' or 'bot'")
    if st == "global":
        return "global", "*"
    sid = (scope_id or default_bot or "").strip().lower()
    if not sid:
        raise HTTPException(status_code=400, detail="scope_id is required for bot scope")
    return "bot", sid


def _refuse_credential_key(key: str) -> None:
    """Credential rows are written only through the provider/integration APIs (TASK-1520)."""
    if is_credential_setting_key(key):
        raise HTTPException(
            status_code=403,
            detail="Credential settings are managed through the provider and integration APIs",
        )


def _validate_setting_writes(engine, writes: list[tuple[str, str, str, object]]) -> None:
    """Refuse a setting write that a feature gate forbids, before anything is stored.

    ``working_memory_enabled`` (TASK-1281, docs/WORKING_MEMORY.md 7) is bot-scoped;
    true needs a non-protected claude-code bot. Credential keys are refused
    outright (TASK-1520).
    """
    for _scope_type, _scope_id, key, _value in writes:
        _refuse_credential_key(key)
    policy = WorkingMemoryPolicy(engine)
    for scope_type, scope_id, key, value in writes:
        if key in RETIRED_BALANCE_SETTINGS:
            # One writer for API credit: the per-provider counter (TASK-1461).
            raise HTTPException(status_code=400, detail="Set API balances with PUT /v1/usage/{provider}/balance")
        try:
            policy.validate_setting_write(scope_type, scope_id, key, value)
        except WorkingMemoryError as exc:
            raise HTTPException(status_code=400, detail=exc.message) from exc


def _resolve_effective_bot_prompt(config, profile) -> str:
    """Effective prompt for a bot: persona override body if set, else base (TASK-477).

    Falls back to the stored ``system_prompt`` if the override id is missing or
    no longer resolves to a persona template.
    """
    override_id = getattr(profile, "prompt_override_id", None)
    base = profile.system_prompt or ""
    if not override_id:
        return base
    try:
        from ...prompt_registry import PERSONA_CATEGORY

        row = get_prompt_template_store(config).get_by_id(int(override_id))
        if row is not None and row.category == PERSONA_CATEGORY and (row.body or "").strip():
            return row.body
    except Exception as e:
        logger.warning("Could not resolve persona override id=%s for %s: %s", override_id, profile.slug, e)
    return base


def _profile_tags(engine, slugs: list[str] | None = None) -> dict[str, list[str]] | None:
    """Assigned tags per bot (one read), or None when they cannot be read.

    A tag read failure must not fail a profile read or a committed write, so it
    reports the tags unavailable (``tags: null``) instead.
    """
    try:
        store = BotTagStore(engine)
        if slugs is not None and len(slugs) == 1:
            return {slugs[0]: store.tags_for(slugs[0])}
        return store.tags_by_bot()
    except Exception as exc:
        logger.warning("Bot tags unavailable for the profile response: %s", type(exc).__name__)
        return None


def _to_profile_response(profile, settings: dict[str, object] | None = None,
                         tags: dict[str, list[str]] | None = None) -> BotProfileResponse:
    return BotProfileResponse(
        slug=profile.slug,
        name=profile.name,
        description=profile.description,
        system_prompt=profile.system_prompt,
        prompt_override_id=getattr(profile, "prompt_override_id", None),
        requires_memory=profile.requires_memory,
        voice_optimized=profile.voice_optimized,
        tts_mode=profile.tts_mode,
        include_summaries=profile.include_summaries,
        include_in_global_search=profile.include_in_global_search,
        tags=None if tags is None else tags.get(profile.slug, []),
        uses_tools=profile.uses_tools,
        uses_search=profile.uses_search,
        uses_home_assistant=profile.uses_home_assistant,
        default_model=profile.default_model,
        color=profile.color,
        avatar=profile.avatar,
        avatar_render=getattr(profile, "avatar_render", None),
        default_voice=profile.default_voice,
        nextcloud_config=profile.nextcloud_config,
        bot_type=normalize_bot_type(getattr(profile, "bot_type", None), profile.agent_backend),
        harness=getattr(profile, "harness", None),
        endpoint_id=getattr(profile, "endpoint_id", None),
        agent_backend=profile.agent_backend,
        agent_backend_config=profile.agent_backend_config,
        settings=settings or {},
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


def _effective_bot_settings(service, slug: str) -> dict[str, object]:
    """Return effective settings with the same precedence as runtime resolution."""
    from ...bots import BotManager

    normalized = (slug or "").strip().lower()
    bot = BotManager(service.config).get_bot(normalized)
    effective: dict[str, object] = dict(getattr(bot, "settings", {}) or {})

    store = get_runtime_settings_store(service.config)
    if store.engine is None:
        return effective

    effective.update(store.get_scope_settings("global", "*"))
    effective.update(store.get_scope_settings("bot", normalized))
    return effective


async def _publish_bot_profile_changed(service, slug: str) -> None:
    """Notify open UI windows after the database write and cache reload succeed."""
    subscriber = getattr(service, "_redis_subscriber", None)
    if subscriber is None:
        logger.warning("Bot profile changed for %s but the event stream is unavailable", slug)
        return
    try:
        stream_id = await subscriber.publish_global_config_event({
            "_type": "bot_profile_changed", "bot_id": slug,
        })
        if stream_id is None:
            logger.warning("Bot profile changed for %s but the event was not published", slug)
    except Exception:
        # A transient Redis failure must not turn a committed profile write into
        # an HTTP error that callers might retry. SSE hello reconciles on reconnect.
        logger.exception("Bot profile changed for %s but event publication failed", slug)


def _reload_bot_registry() -> None:
    """Reload in-memory bot registry from YAML + DB overrides."""
    from ...bots import invalidate_bots_cache, _check_reload

    invalidate_bots_cache()
    _check_reload()


def _reload_service_model_catalog(service) -> None:
    """Rebuild available-model and agent-backend mappings after bot changes."""
    reload_models = getattr(service, "_load_available_models", None)
    if callable(reload_models):
        reload_models()


def _invalidate_bot_instance_cache(service, bot_id: str) -> int:
    """Drop cached ServiceLLMBawt instances for a bot so prompt changes apply."""
    invalidate = getattr(service, "invalidate_bot_instances", None)
    if callable(invalidate):
        return int(invalidate(bot_id))

    normalized = (bot_id or "").strip().lower()
    if not normalized:
        return 0
    keys_to_remove = [key for key in service._llm_bawt_cache if key[1] == normalized]
    for key in keys_to_remove:
        del service._llm_bawt_cache[key]
    return len(keys_to_remove)


def _clear_session_model_overrides(service, bot_id: str | None = None, user_id: str | None = None) -> int:
    """Clear sticky per-session model overrides so bot/default model changes take effect."""
    clear_overrides = getattr(service, "clear_session_model_overrides", None)
    if callable(clear_overrides):
        return int(clear_overrides(bot_id=bot_id, user_id=user_id))

    # Backward-compatible fallback for older service objects.
    if not hasattr(service, "_session_model_overrides"):
        return 0

    normalized_bot = (bot_id or "").strip().lower() if bot_id is not None else None
    normalized_user = (user_id or "").strip() if user_id is not None else None
    keys_to_remove = []
    for key in service._session_model_overrides:
        key_bot, key_user = key
        if normalized_bot is not None and key_bot != normalized_bot:
            continue
        if normalized_user is not None and key_user != normalized_user:
            continue
        keys_to_remove.append(key)

    for key in keys_to_remove:
        del service._session_model_overrides[key]
    return len(keys_to_remove)


def _invalidate_all_instance_cache(service) -> int:
    """Drop all cached ServiceLLMBawt instances."""
    invalidate = getattr(service, "invalidate_all_instances", None)
    if callable(invalidate):
        return int(invalidate())
    cleared = len(service._llm_bawt_cache)
    service._llm_bawt_cache.clear()
    return cleared


async def _persist_bot_profile(
    payload: dict[str, object],
    *,
    create_only: bool,
) -> BotProfileResponse:
    service = get_service()
    store = get_bot_profile_store(service.config)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Bot profiles DB unavailable")

    slug = str(payload.get("slug") or "").strip().lower()
    if not slug:
        raise HTTPException(status_code=400, detail="Bot slug is required")
    previous = store.get(slug)
    if create_only and previous is not None:
        raise HTTPException(status_code=409, detail=f"Bot '{slug}' already exists")

    _validate_profile_payload(payload)

    # Resolve the avatar to a self-hosted ``data:`` URL at write time so the
    # chat UI never fetches an avatar from a CDN at runtime. Only recompute
    # when the avatar actually changed (or was never rendered) — avoids a
    # redundant network fetch on every unrelated profile edit.
    if "avatar" in payload:
        import asyncio

        from ...media.avatar import resolve_avatar_render

        new_avatar = payload.get("avatar")
        needs_render = (
            previous is None
            or previous.avatar != new_avatar
            or not getattr(previous, "avatar_render", None)
        )
        if needs_render:
            payload["avatar_render"] = await asyncio.to_thread(
                resolve_avatar_render,
                new_avatar if isinstance(new_avatar, str) else None,
            )

    # Provider health is probed here, before the save's transaction, so no
    # network call runs under the bot's admission lock (TASK-1302 A1′).
    route_gate = ProfileRouteGate.for_payload(service.config, store.engine, payload)
    try:
        profile = store.upsert(payload, guard=ProfileSaveGuard(ProfileSwitchGate(), route_gate))
    except SystemGateRefused as refused:  # a switch onto a held runtime; nothing written
        raise HTTPException(status_code=refused.status_code, detail=str(refused))
    except RouteIneligible as refused:  # a new route with no usable credential; nothing written
        raise HTTPException(status_code=refused.status_code, detail=refused.reason)
    except ProfileRowChanged:  # a PATCH merged onto a stale row; the PATCH route re-merges
        raise
    except ProfileFieldsChanged as stale:  # expected_fields no longer match; nothing written
        raise HTTPException(status_code=stale.status_code, detail={
            "code": stale.code, "message": str(stale), "current": stale.current})
    except TagSetChanged as stale:  # expected_tags no longer match; nothing written
        raise HTTPException(status_code=stale.status_code, detail={
            "code": stale.code, "message": str(stale), "current_tags": stale.current})
    except BotTagError as refused:  # an unknown or malformed tag; nothing written
        raise HTTPException(status_code=refused.status_code, detail=str(refused))
    except Exception as exc:  # surface DB constraint violations as 400s
        msg = _humanize_bot_constraint_error(exc)
        if msg is not None:
            raise HTTPException(status_code=400, detail=msg)
        raise
    _reload_bot_registry()
    _reload_service_model_catalog(service)
    _invalidate_bot_instance_cache(service, profile.slug)
    _clear_session_model_overrides(service, bot_id=profile.slug)
    rotated_threads = _rotate_threads_on_runtime_switch(store.engine, previous, profile)

    effective_prompt = _resolve_effective_bot_prompt(service.config, profile)
    if profile.agent_backend == "openclaw" and effective_prompt.strip():
        await _push_soul_background(profile.slug, openclaw_agent_id(profile), effective_prompt)

    response = _to_profile_response(profile, settings=_effective_bot_settings(service, profile.slug),
                                    tags=_profile_tags(store.engine, [profile.slug]))
    await _publish_bot_profile_changed(service, profile.slug)
    await _publish_runtime_switch_threads(service, profile.slug, rotated_threads)
    return response


def _rotate_threads_on_runtime_switch(engine, previous, profile) -> list[tuple[str | None, str]]:
    """Start fresh threads when a save crosses runtime families (chat <-> agent).

    The profile write is already committed, so a rotation failure is logged,
    not raised: a 500 here would invite retries of a write that succeeded.
    """
    if not runtime_family_changed(previous, profile):
        return []
    try:
        return rotate_active_threads(engine, profile.slug)
    except Exception:
        logger.exception(
            "Bot %s changed runtime %s -> %s but thread rotation failed; "
            "the old thread stays active",
            profile.slug, runtime_family(previous), runtime_family(profile),
        )
        return []


async def _publish_runtime_switch_threads(
    service, slug: str, rotated: list[tuple[str | None, str]],
) -> None:
    """Tell each affected user's UI about its fresh thread (best-effort)."""
    subscriber = getattr(service, "_redis_subscriber", None)
    if subscriber is None or not rotated:
        return
    for user_id, new_session_id in rotated:
        if not user_id:
            continue
        try:
            await subscriber.publish_tool_event(
                slug, user_id,
                thread_switched_event(slug, user_id, new_session_id, RUNTIME_SWITCH_SOURCE),
            )
        except Exception:
            logger.exception("thread_switched publish failed for %s/%s", slug, user_id)


@router.get("/v1/settings", response_model=RuntimeSettingsResponse, tags=["System"])
def list_runtime_settings(
    scope_type: str = Query("bot", description="global or bot"),
    scope_id: str | None = Query(None, description="bot slug for bot scope"),
):
    """List runtime settings for a scope."""
    service = get_service()
    st, sid = _normalize_scope(scope_type, scope_id, service._default_bot)
    store = get_runtime_settings_store(service.config)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Runtime settings DB unavailable")
    data = store.get_scope_settings(st, sid)
    items = [
        RuntimeSettingItem(key=k, value=v)
        for k, v in sorted(data.items())
        if not is_credential_setting_key(k)
    ]
    return RuntimeSettingsResponse(scope_type=st, scope_id=sid, settings=items)


@router.get("/v1/settings/all", response_model=RuntimeSettingsListResponse, tags=["System"])
def list_all_runtime_settings(
    scope_type: str | None = Query(None, description="Filter by scope type: global or bot"),
    scope_id: str | None = Query(None, description="Filter by scope id ('*' for global)"),
    key: str | None = Query(None, description="Filter by exact key"),
    key_prefix: str | None = Query(None, description="Filter by key prefix"),
    limit: int = Query(200, ge=1, le=1000, description="Max rows to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """List runtime settings across all scopes."""
    service = get_service()
    store = get_runtime_settings_store(service.config)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Runtime settings DB unavailable")

    normalized_scope_type = scope_type.strip().lower() if scope_type else None
    if normalized_scope_type and normalized_scope_type not in {"global", "bot"}:
        raise HTTPException(status_code=400, detail="scope_type must be 'global' or 'bot'")

    normalized_scope_id = scope_id.strip().lower() if scope_id else None
    normalized_key = key.strip() if key else None
    normalized_key_prefix = key_prefix.strip() if key_prefix else None

    # Credential rows never appear here, whatever the filters (TASK-1520).
    conditions: list = [
        ~func.lower(func.trim(RuntimeSetting.key)).startswith(prefix)
        for prefix in CREDENTIAL_SETTING_PREFIXES
    ]
    if normalized_scope_type:
        conditions.append(RuntimeSetting.scope_type == normalized_scope_type)
    if normalized_scope_id:
        conditions.append(RuntimeSetting.scope_id == normalized_scope_id)
    if normalized_key:
        conditions.append(RuntimeSetting.key == normalized_key)
    if normalized_key_prefix:
        conditions.append(RuntimeSetting.key.startswith(normalized_key_prefix))

    statement = select(RuntimeSetting).where(*conditions)
    count_statement = select(func.count()).select_from(RuntimeSetting).where(*conditions)

    statement = statement.order_by(RuntimeSetting.scope_type, RuntimeSetting.scope_id, RuntimeSetting.key)

    with Session(store.engine) as session:
        total_count = int(session.exec(count_statement).one() or 0)
        rows = session.exec(statement.offset(offset).limit(limit)).all()

    items = []
    for row in rows:
        try:
            value = json.loads(row.value_json)
        except Exception:
            value = row.value_json
        items.append(
            RuntimeSettingRecord(
                scope_type=row.scope_type,
                scope_id=row.scope_id,
                key=row.key,
                value=value,
                updated_at=row.updated_at,
            )
        )

    return RuntimeSettingsListResponse(
        settings=items,
        total_count=total_count,
        filters={
            "scope_type": normalized_scope_type,
            "scope_id": normalized_scope_id,
            "key": normalized_key,
            "key_prefix": normalized_key_prefix,
            "limit": limit,
            "offset": offset,
        },
    )


@router.put("/v1/settings", tags=["System"])
def upsert_runtime_setting(request: RuntimeSettingUpsertRequest):
    """Upsert one runtime setting for global or bot scope."""
    service = get_service()
    st, sid = _normalize_scope(request.scope_type, request.scope_id, service._default_bot)
    store = get_runtime_settings_store(service.config)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Runtime settings DB unavailable")
    _validate_setting_writes(store.engine, [(st, sid, request.key, request.value)])
    store.set_value(st, sid, request.key, request.value)
    if st == "bot":
        _invalidate_bot_instance_cache(service, sid)
    else:
        _invalidate_all_instance_cache(service)
    return {
        "success": True,
        "scope_type": st,
        "scope_id": sid,
        "key": request.key,
        "value": request.value,
    }


@router.delete("/v1/settings", tags=["System"])
def delete_runtime_setting(
    scope_type: str = Query(..., description="global or bot"),
    key: str = Query(..., description="Setting key"),
    scope_id: str | None = Query(None, description="bot slug for bot scope"),
):
    """Delete one runtime setting."""
    _refuse_credential_key(key)
    service = get_service()
    st, sid = _normalize_scope(scope_type, scope_id, service._default_bot)
    store = get_runtime_settings_store(service.config)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Runtime settings DB unavailable")
    deleted = store.delete_value(st, sid, key)
    if st == "bot":
        _invalidate_bot_instance_cache(service, sid)
    else:
        _invalidate_all_instance_cache(service)
    return {
        "success": True,
        "scope_type": st,
        "scope_id": sid,
        "key": key,
        "deleted": bool(deleted),
    }


@router.post("/v1/settings/batch", tags=["System"])
def batch_upsert_runtime_settings(request: RuntimeSettingBatchUpsertRequest):
    """Batch upsert runtime settings."""
    service = get_service()
    store = get_runtime_settings_store(service.config)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Runtime settings DB unavailable")

    scoped = [
        (*_normalize_scope(item.scope_type, item.scope_id, service._default_bot), item)
        for item in request.items
    ]
    # Every item is validated before any is applied.
    _validate_setting_writes(store.engine, [(st, sid, item.key, item.value) for st, sid, item in scoped])
    applied: list[dict] = []
    touched_bots: set[str] = set()
    touched_global = False
    for st, sid, item in scoped:
        store.set_value(st, sid, item.key, item.value)
        if st == "bot":
            touched_bots.add(sid)
        else:
            touched_global = True
        applied.append(
            {
                "scope_type": st,
                "scope_id": sid,
                "key": item.key,
            }
        )
    if touched_global:
        _invalidate_all_instance_cache(service)
    else:
        for bot_id in touched_bots:
            _invalidate_bot_instance_cache(service, bot_id)
    return {"success": True, "applied_count": len(applied), "applied": applied}


@router.get("/v1/bots/profiles", response_model=BotProfileListResponse, tags=["System"])
def list_bot_profiles(
    q: str | None = Query(None, description="Text filter against slug, name, and description"),
    uses_tools: bool | None = Query(None, description="Filter by uses_tools"),
    uses_search: bool | None = Query(None, description="Filter by uses_search"),
    include_in_global_search: bool | None = Query(None, description="Filter by include_in_global_search"),
    requires_memory: bool | None = Query(None, description="Filter by requires_memory"),
    voice_optimized: bool | None = Query(None, description="Filter by voice_optimized"),
    uses_home_assistant: bool | None = Query(None, description="Filter by uses_home_assistant"),
    default_model: str | None = Query(None, description="Filter by default model alias"),
    limit: int = Query(200, ge=1, le=1000, description="Max rows to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """List DB-backed bot personality profiles with filters."""
    service = get_service()
    store = get_bot_profile_store(service.config)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Bot profiles DB unavailable")

    query_text = q.strip().lower() if q else None
    model_filter = default_model.strip() if default_model else None
    rows = store.list_all()

    filtered = []
    for row in rows:
        if query_text:
            haystack = " ".join(
                [
                    (row.slug or ""),
                    (row.name or ""),
                    (row.description or ""),
                ]
            ).lower()
            if query_text not in haystack:
                continue
        if uses_tools is not None and row.uses_tools != uses_tools:
            continue
        if uses_search is not None and row.uses_search != uses_search:
            continue
        if include_in_global_search is not None and row.include_in_global_search != include_in_global_search:
            continue
        if requires_memory is not None and row.requires_memory != requires_memory:
            continue
        if voice_optimized is not None and row.voice_optimized != voice_optimized:
            continue
        if uses_home_assistant is not None and row.uses_home_assistant != uses_home_assistant:
            continue
        if model_filter is not None and row.default_model != model_filter:
            continue
        filtered.append(row)

    total_count = len(filtered)
    page = filtered[offset : offset + limit]
    tags = _profile_tags(store.engine)

    return BotProfileListResponse(
        profiles=[
            _to_profile_response(profile, settings=_effective_bot_settings(service, profile.slug),
                                 tags=tags)
            for profile in page
        ],
        total_count=total_count,
        filters={
            "q": query_text,
            "uses_tools": uses_tools,
            "uses_search": uses_search,
            "include_in_global_search": include_in_global_search,
            "requires_memory": requires_memory,
            "voice_optimized": voice_optimized,
            "uses_home_assistant": uses_home_assistant,
            "default_model": model_filter,
            "limit": limit,
            "offset": offset,
        },
    )


@router.post("/v1/bots", response_model=BotProfileResponse, tags=["System"])
async def create_bot(request: BotCreateRequest):
    """Create a new bot profile. Returns 409 when slug already exists."""
    payload = _request_to_profile_payload(request.slug, request)
    return await _persist_bot_profile(payload, create_only=True)


@router.get("/v1/bots/{slug}/profile", response_model=BotProfileResponse, tags=["System"])
def get_bot_profile(slug: str):
    """Get bot personality profile by slug."""
    service = get_service()
    store = get_bot_profile_store(service.config)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Bot profiles DB unavailable")

    profile = store.get(slug)
    if profile is None:
        raise HTTPException(status_code=404, detail="Bot profile not found")
    return _to_profile_response(profile, settings=_effective_bot_settings(service, profile.slug),
                                tags=_profile_tags(store.engine, [profile.slug]))


@router.put("/v1/bots/{slug}/profile", response_model=BotProfileResponse, tags=["System"])
async def upsert_bot_profile(slug: str, request: BotProfileUpsertRequest):
    """Create or update a bot personality profile."""
    payload = _request_to_profile_payload(slug, request)
    return await _persist_bot_profile(payload, create_only=False)


@router.patch("/v1/bots/{slug}/profile", response_model=BotProfileResponse, tags=["System"])
async def patch_bot_profile(slug: str, request: BotProfilePatchRequest):
    """Partially update a bot profile. Only provided fields are changed.

    The store gets the whole merged row, so the merge is pinned to the row it
    was read from: if another write lands in between, the store refuses and the
    merge is redone on the fresh row (TASK-1521), up to ``_PATCH_ATTEMPTS``.
    """
    service = get_service()
    store = get_bot_profile_store(service.config)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Bot profiles DB unavailable")

    for attempt in range(1, _PATCH_ATTEMPTS + 1):
        existing = store.get(slug.strip().lower())
        if not existing:
            raise HTTPException(status_code=404, detail=f"Bot '{slug}' not found")
        payload = _patch_payload(existing, request)
        try:
            return await _persist_bot_profile(payload, create_only=False)
        except ProfileRowChanged as stale:
            if attempt == _PATCH_ATTEMPTS:
                raise HTTPException(status_code=stale.status_code, detail={
                    "code": stale.code, "message": str(stale)})
            logger.info("Bot profile %s changed under a PATCH; re-merging (attempt %d)",
                        existing.slug, attempt)
    raise AssertionError("unreachable")


#: How many times a PATCH re-merges onto a row another write changed underneath it.
_PATCH_ATTEMPTS = 3


def _patch_payload(existing, request: BotProfilePatchRequest) -> dict[str, object]:
    """The stored row with the request's explicitly-set fields merged over it."""
    # Build payload from existing + only the fields that were explicitly set
    payload: dict[str, object] = {
        "slug": existing.slug,
        "name": existing.name,
        "description": existing.description,
        "system_prompt": existing.system_prompt,
        "prompt_override_id": existing.prompt_override_id,
        "requires_memory": existing.requires_memory,
        "voice_optimized": existing.voice_optimized,
        "tts_mode": existing.tts_mode,
        "include_summaries": existing.include_summaries,
        "include_in_global_search": existing.include_in_global_search,
        "uses_tools": existing.uses_tools,
        "uses_search": existing.uses_search,
        "uses_home_assistant": existing.uses_home_assistant,
        "default_model": existing.default_model,
        "harness": getattr(existing, "harness", None),
        "endpoint_id": getattr(existing, "endpoint_id", None),
        "color": existing.color,
        "avatar": existing.avatar,
        "default_voice": existing.default_voice,
        "nextcloud_config": existing.nextcloud_config,
        # bot_type intentionally omitted here — it is derived from the final
        # merged agent_backend below, never copied from the (possibly stale)
        # stored value. See the derivation note after the field loop.
        "agent_backend": existing.agent_backend,
        "agent_backend_config": existing.agent_backend_config,
    }
    for field_name in request.model_fields_set - {"updated_by"}:
        payload[field_name] = getattr(request, field_name)
    if "tags" in request.model_fields_set:
        # A full-set replace (BOT_TAGS.md §8). The legacy flag mirrors
        # ``not private``, so naming both is only allowed when they agree.
        mirror = BehaviorTag.PRIVATE not in request.tags
        if ("include_in_global_search" in request.model_fields_set
                and request.include_in_global_search is not mirror):
            raise HTTPException(status_code=422, detail=(
                "include_in_global_search must equal 'private' not in tags when both are sent; "
                "send tags alone"))
        payload["tags_assigned_by"] = request.updated_by or API_ASSIGNER
    payload[profile_changes.CHANGED_BY] = request.updated_by or API_ASSIGNER
    payload[profile_changes.ROW_VERSION] = existing.updated_at

    # bot_type is DERIVED from agent_backend (the single source of truth), not
    # carried over from the stored row (TASK-516). A PATCH that touches an
    # unrelated field (persona, model) must not resurrect a stale bot_type: if
    # the existing row was already inconsistent (agent-typed with no backend),
    # copying its bot_type back in would make _validate_profile_payload reject
    # the edit before the store could heal it — the exact loop that blocked
    # nova/mira. Derive from the FINAL merged agent_backend so clearing the
    # backend in this same PATCH also demotes agent→chat. Only an explicit
    # bot_type in the request overrides (the store guard still enforces that an
    # explicit 'agent' has a backend).
    if "bot_type" not in request.model_fields_set:
        payload["bot_type"] = normalize_bot_type(None, payload.get("agent_backend"))
    return payload


@router.post("/v1/admin/reload-bots", tags=["Admin"])
def reload_bots():
    """Force reload bot registry from DB + YAML."""
    service = get_service()
    _reload_bot_registry()
    _reload_service_model_catalog(service)
    cleared_instances = _invalidate_all_instance_cache(service)
    cleared_session_overrides = _clear_session_model_overrides(service)
    return {
        "status": "reloaded",
        "cleared_instances": cleared_instances,
        "cleared_session_model_overrides": cleared_session_overrides,
    }


@router.delete("/v1/bots/{slug}/profile", tags=["System"])
async def delete_bot_profile(
    slug: str,
    purge: bool = Query(False, description="Also purge all bot data (messages, memories, settings, etc.)"),
):
    """Delete a bot personality profile. Pass ?purge=true to also wipe all associated data."""
    service = get_service()
    store = get_bot_profile_store(service.config)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Bot profiles DB unavailable")

    deleted = store.delete(slug)
    if not deleted:
        raise HTTPException(status_code=404, detail="Bot profile not found")

    normalized_slug = slug.strip().lower()
    _reload_bot_registry()
    _reload_service_model_catalog(service)
    _invalidate_bot_instance_cache(service, normalized_slug)
    _clear_session_model_overrides(service, bot_id=normalized_slug)

    response: dict = {"success": True, "slug": normalized_slug}
    if purge:
        response["purge"] = purge_bot_data(service.config, normalized_slug)
    await _publish_bot_profile_changed(service, normalized_slug)
    return response


@router.post("/v1/bots/{slug}/purge-data", tags=["System"])
def purge_bot_data_endpoint(slug: str):
    """Purge all data for a bot (messages, memories, settings, etc.) without deleting the profile."""
    service = get_service()
    normalized_slug = slug.strip().lower()
    if not normalized_slug:
        raise HTTPException(status_code=400, detail="Invalid slug")
    result = purge_bot_data(service.config, normalized_slug)
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return result


@router.post("/v1/bots/cleanup-orphans", tags=["System"])
def cleanup_orphans(
    dry_run: bool = Query(True, description="Preview a bounds-based partition manifest"),
    manifest_digest: str | None = Query(None, description="Digest from an exact preview; required to execute"),
):
    """Preview orphan partitions, or confirm that exact manifest to execute.

    This route never deletes shared rows. A changed manifest fails before any
    mutation; each child is revalidated under short, bounded locks.
    """
    service = get_service()
    try:
        result = cleanup_orphaned_bot_data(service.config, dry_run=dry_run,
                                           manifest_digest=manifest_digest)
    except PartitionExecutionHalted as exc:
        receipt = exc.receipt()
        # Even completed-with-warning uses 207: do not let a successful DB
        # result hide a journal-finalization failure. The body carries both.
        return JSONResponse(status_code=207, content=receipt)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return result
