"""Install-time bot presets (TASK-2516).

- ``GET  /v1/setup/bot-presets``: every pack, the stored selection (or the
  recommended one), and a dry-run plan against this install's usable endpoints.
- ``PUT  /v1/setup/bot-presets``: store a selection (the installer may record
  one before any provider is connected).
- ``POST /v1/setup/bot-presets``: create the selected bots that can run now,
  through the admin create-only writer. Existing slugs are left untouched; a
  bot the writer refuses for another reason is reported as ``failed`` with
  that reason, so a rerun can finish the lineup.

Catalog parsing and planning live in :mod:`llm_bawt.seeding.bot_presets`.
"""

from __future__ import annotations

import logging
from functools import partial

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy import text

from ...seeding import FIRST_BOT_AGENT_BACKEND
from ...seeding import bot_presets as presets
from ...seeding.setup_role_dependencies import SetupRoleConflict, install_missing_roles
from ... import bot_profile_changes as profile_changes
from ..agent_runtime import check_agent_runtime
from ..dependencies import get_bot_profile_store, get_runtime_settings_store, get_service
from ..providers.registry import access_path_connections, route_eligibility
from ..setup_orientation import prepare_first_chat, record_choice
from .model_catalog import harness_for_agent_backend
from .settings import _USER_ID_QUERY, _persist_bot_profile, profile_actor
from .setup import _http_error, _record_first_default_bot

router = APIRouter()
log = logging.getLogger(__name__)


class PresetSelection(BaseModel):
    selection: list[str] | None = Field(default=None, max_length=presets._MAX_PRESETS * 4)
    # Optional per-preset endpoint choice: ``{"starter/helper": 12}``.
    endpoints: dict[str, int] | None = None
    guide_me: bool | None = None


def _settings(config):
    store = get_runtime_settings_store(config)
    if store.engine is None:
        raise RuntimeError("Runtime settings DB unavailable")
    return store


def usable_endpoints(config) -> list[presets.EndpointChoice]:
    """Catalog endpoints an agent can run on, by the bot-save gate's own route rule."""
    catalog = config.ensure_model_catalog()
    if catalog is None:
        return []
    endpoints = [ep for ep in catalog.endpoints()
                 if harness_for_agent_backend(FIRST_BOT_AGENT_BACKEND, ep.access_path)]
    connections = access_path_connections(config, {ep.access_path.key for ep in endpoints})
    usable = []
    for ep in endpoints:
        verdict = route_eligibility(ep.access_path.key, ep.access_path.auth_mechanism,
                                    connections.get(ep.access_path.key))
        if not verdict["eligible"]:
            continue
        usable.append(presets.EndpointChoice(ep.id, ep.ref, ep.access_path.vendor.lower()))
    return usable


def _known_tags(store) -> set[str] | None:
    try:
        with store.engine.connect() as conn:
            return set(conn.execute(text("SELECT slug FROM bot_tag_catalog")).scalars())
    except Exception:  # noqa: BLE001 — unknown catalog: keep every tag, the writer validates
        return None


def load_state(config, selection: list[str] | None = None,
               endpoints: dict[str, int] | None = None, *, allow_retired: bool = False) -> dict:
    """Catalog, selection and dry-run plan. Shared by the routes and ``preset_cli``."""
    settings = _settings(config)
    packs = presets.all_packs(settings.get_value("global", "*", presets.PACKS_KEY))
    known = presets.presets_by_ref(packs)
    stored = settings.get_value("global", "*", presets.SELECTION_KEY)
    retired: list[str] = []
    if selection is not None:
        if allow_retired:
            selection, retired = presets.partition_selection(selection, known)
        else:
            selection = presets.normalize_selection(selection, known)
    elif isinstance(stored, list):
        selection = [ref for ref in stored if ref in known]
    else:
        selection = presets.recommended_refs(packs)
    profiles = get_bot_profile_store(config)
    existing = {p.slug for p in profiles.list_all()}
    tags = _known_tags(profiles)
    rows = presets.plan(selection, known, usable_endpoints(config), existing, tags, endpoints)
    for ref in retired:
        slug = ref.split("/", 1)[1]
        rows.append({"ref": ref, "slug": slug, "name": slug, "action": "blocked",
                     "reason": "This preset is no longer offered. Change your selection to continue."})
    selected = set(selection)
    visible = [p.public() for p in packs if any(not bot.legacy or bot.ref in selected for bot in p.presets)]
    return {"packs": visible, "selection": selection,
            "stored": isinstance(stored, list), "plan": rows,
            "_known": known, "_tags": tags}


def _state(service, body: PresetSelection | None = None, *, allow_retired: bool = False) -> dict:
    return load_state(service.config, body.selection if body is not None else None,
                      body.endpoints if body is not None else None, allow_retired=allow_retired)


def save_selection(config, selection: list[str]) -> dict:
    state = load_state(config, selection)
    _settings(config).set_value("global", "*", presets.SELECTION_KEY, state["selection"])
    state["stored"] = True
    return state


def _public(state: dict) -> dict:
    return {k: v for k, v in state.items() if not k.startswith("_")}


@router.get("/v1/setup/bot-presets", tags=["Setup"])
async def get_bot_presets():
    service = get_service()
    try:
        return _public(await run_in_threadpool(_state, service))
    except presets.PresetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise _http_error(exc, what="Bot presets") from exc


@router.put("/v1/setup/bot-presets", tags=["Setup"])
async def put_bot_presets(body: PresetSelection):
    service = get_service()
    if body.selection is None:
        raise HTTPException(status_code=400, detail="selection is required")

    try:
        return _public(await run_in_threadpool(save_selection, service.config, body.selection))
    except presets.PresetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise _http_error(exc, what="Bot presets") from exc


@router.post("/v1/setup/bot-presets", tags=["Setup"])
async def apply_bot_presets(http_request: Request, body: PresetSelection | None = None,
                            user_id: str | None = _USER_ID_QUERY):
    """Create the planned bots. Per-row results; a taken slug becomes ``skip``."""
    service = get_service()
    try:
        runtime = await check_agent_runtime(service.config)
        if not runtime.ready:
            raise HTTPException(status_code=503, detail=f"Agent runtime not ready: {runtime.detail}")
        state = await run_in_threadpool(_state, service, body, allow_retired=True)
        profiles = get_bot_profile_store(service.config)
        actor = profile_actor(http_request, user_id)
        known, tags = state["_known"], state["_tags"]
        guide_me = body.guide_me if body is not None else None
        audience = user_id.strip() if user_id and actor == user_id.strip() else None
        if guide_me and (not audience or actor.startswith("bot:")):
            raise HTTPException(status_code=403, detail="Choose an administrator before starting the introduction")
        dispatcher = getattr(service, "_inter_bot_dispatcher", None)
        prompt = await run_in_threadpool(prepare_first_chat, service.config) if guide_me else None
        delivery = None
        first_created = None

        def save_choice(conn, profile):
            nonlocal delivery
            delivery = record_choice(
                conn, profile, opted_in=bool(guide_me), user_id=audience,
                prompt=prompt, store=dispatcher.store if dispatcher is not None else None,
            )

        results = []
        for row in state["plan"]:
            result = dict(row)
            if row["action"] == "create":
                preset = known[row["ref"]]
                roles = preset.role_dependencies
                payload = {
                    "slug": preset.slug, "name": preset.name, "description": preset.summary,
                    "system_prompt": preset.system_prompt, "requires_memory": True,
                    "voice_optimized": False, "tts_mode": False, "include_summaries": True,
                    "uses_tools": True, "uses_search": True, "uses_home_assistant": False,
                    "bot_type": "agent", "agent_backend": FIRST_BOT_AGENT_BACKEND,
                    "harness": FIRST_BOT_AGENT_BACKEND, "endpoint_id": row["endpoint_id"],
                    "tags": [t for t in preset.tags if roles is not None or tags is None or t in tags],
                    "tags_assigned_by": actor,
                    profile_changes.CHANGED_BY: actor,
                    **{k: v for k, v in (("color", preset.color), ("avatar", preset.icon)) if v},
                }
                delivery = None
                try:
                    await _persist_bot_profile(
                        payload, create_only=True,
                        **({"before_create": partial(install_missing_roles, roles=roles)} if roles else {}),
                        **({"on_create": save_choice} if guide_me is not None and first_created is None else {}),
                    )
                except SetupRoleConflict as exc:
                    result.update(action="blocked", reason=str(exc))
                except HTTPException as exc:
                    if exc.status_code >= 500:
                        raise
                    # The writer answers 409 for a taken slug and for a refused
                    # route alike; only the row itself tells them apart.
                    taken = exc.status_code == 409 and await run_in_threadpool(profiles.get, preset.slug)
                    if taken:
                        result.update(action="skip",
                                      reason=f"A bot named '{preset.slug}' already exists; it is left unchanged.")
                    else:
                        result.update(action="failed", reason=_reason(exc.detail))
                else:
                    result["action"] = "created"
                    if first_created is None:
                        first_created = preset.slug
                        if delivery is not None:
                            try:
                                dispatcher.announce_admitted(delivery)
                            except Exception:
                                log.exception("First chat announcement failed; dispatcher will recover the durable row")
                        await run_in_threadpool(_record_first_default_bot, service, first_created)
            results.append(result)
    except presets.PresetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise _http_error(exc, what="Bot presets") from exc
    ok = all(r["action"] in {"created", "skip"} for r in results)
    return {"ok": ok, "selection": state["selection"], "results": results}


def _reason(detail) -> str:
    if isinstance(detail, dict):
        detail = detail.get("message") or detail.get("detail")
    text_ = str(detail or "").strip()
    return f"Not created: {text_}" if text_ else "Not created; try again."
