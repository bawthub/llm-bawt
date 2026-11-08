"""Effective-config inspector routes (TASK-487).

Read-only endpoint that returns the fully-resolved system prompt (per-section
provenance) and runtime settings (per-source) for a given (bot, user). No LLM
call, no writes — the safety net every Prompt & Config Unification refactor
verifies against.
"""

import logging

from fastapi import APIRouter, HTTPException, Query

from ..dependencies import get_service
from ..http_errors import internal_error

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/v1/config/effective", tags=["System"])
async def get_effective_config(
    bot: str = Query(..., description="Bot slug (e.g. nova, snark)"),
    user: str | None = Query(None, description="User id; defaults to service DEFAULT_USER"),
    prompt: str = Query("", description="Optional user prompt to reflect per-turn gates"),
):
    """Return the effective assembled prompt + resolved settings for a bot/user.

    Provenance is truthful because the assembled prompt is produced by the SAME
    ``_assemble_system_builder`` method a live turn uses (see TASK-487).
    """
    service = get_service()
    bot_id = (bot or "").strip().lower()
    if not bot_id:
        raise HTTPException(status_code=400, detail="bot is required")
    user_id = (user or getattr(service.config, "DEFAULT_USER", "") or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user is required (no DEFAULT_USER configured)")

    try:
        # Resolve the model the same way the chat path does so the inspected
        # instance matches production (agent bots resolve to their virtual model).
        model_alias, _warnings = service._resolve_request_model(None, bot_id, False)
        llm_bawt = service._get_llm_bawt(model_alias, bot_id, user_id, local_mode=False)
        return llm_bawt.describe_effective_config(prompt)
    except HTTPException:
        raise
    except Exception as e:
        raise internal_error(
            e, f"effective-config inspection failed for bot={bot_id} user={user_id}",
            public="Inspection failed",
        ) from e


@router.get("/v1/config/schema", tags=["System"])
async def get_config_schema(
    bot_type: str | None = Query(
        None, description="Optional bot_type filter (chat|agent); omit for all settings"
    ),
):
    """Expose the typed setting registry (TASK-491) so the frontend renders only
    the controls that apply to a bot's ``bot_type`` instead of hardcoding flag
    lists. Read-only, no auth-sensitive data — pure metadata.

    Returns ``{key: {type, applies_to, storage, label, help, default, legacy_keys, ui_widget}}``.
    When ``bot_type`` is given, only settings whose ``applies_to`` includes it
    are returned — the exact filter the admin UI uses to hide inert switches.
    """
    from ...setting_definitions import SETTING_DEFINITIONS

    bt = (bot_type or "").strip().lower() or None
    if bt and bt not in ("chat", "agent"):
        raise HTTPException(status_code=400, detail="bot_type must be 'chat' or 'agent'")

    schema: dict[str, dict] = {}
    for key, d in SETTING_DEFINITIONS.items():
        if bt and bt not in d.applies_to:
            continue
        schema[key] = {
            "type": d.type,
            "applies_to": list(d.applies_to),
            "storage": d.storage,
            "label": d.label,
            "help": d.help,
            "default": d.default,
            "legacy_keys": list(d.legacy_keys),
            "ui_widget": d.ui_widget,
        }
    return {"bot_type": bt, "settings": schema}
