"""Bot-profile payload building, normalization, and config-time validation.

Split out of ``routes/settings.py`` (TASK-554) as a cohesive unit: turning an
upsert/create request into a profile payload, normalizing the legacy
``agent_backend_config.model`` key, validating that an agent bot's normalized
``endpoint_id`` resolves to a harness-compatible catalog endpoint, and
humanizing the ``bot_profiles`` DB constraint errors.

``routes/settings.py`` imports the names it invokes directly; tests target the
helpers on this module.
"""

import logging

from fastapi import HTTPException

from ...bot_types import normalize_bot_type
from ..dependencies import get_service
from ..schemas import BotCreateRequest, BotProfileUpsertRequest

logger = logging.getLogger(__name__)


def _request_to_profile_payload(
    slug: str,
    request: BotProfileUpsertRequest | BotCreateRequest,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "slug": slug,
        "name": request.name,
        "description": request.description,
        "system_prompt": request.system_prompt,
        "requires_memory": request.requires_memory,
        "voice_optimized": request.voice_optimized,
        "tts_mode": request.tts_mode,
        "include_summaries": request.include_summaries,
        "include_in_global_search": request.include_in_global_search,
        "uses_tools": request.uses_tools,
        "uses_search": request.uses_search,
        "uses_home_assistant": request.uses_home_assistant,
    }
    for field_name in (
        "prompt_override_id",
        "default_model",
        "color",
        "avatar",
        "default_voice",
        "nextcloud_config",
        "bot_type",
        "harness",
        "endpoint_id",
        "agent_backend",
        "agent_backend_config",
    ):
        if field_name in request.model_fields_set:
            payload[field_name] = getattr(request, field_name)
    return payload


#: ``agent_backend_config.codex_transport`` values (TASK-1243): ``exec`` is the
#: one-shot ``codex exec`` stream, ``app_server`` the steerable JSON-RPC server.
CODEX_TRANSPORTS = ("exec", "app_server")


def _validate_profile_payload(payload: dict[str, object]) -> None:
    resolved_type = normalize_bot_type(
        str(payload.get("bot_type")) if payload.get("bot_type") is not None else None,
        str(payload.get("agent_backend")) if payload.get("agent_backend") is not None else None,
    )
    agent_backend = str(payload.get("agent_backend")).strip() if payload.get("agent_backend") is not None else ""
    if resolved_type == "agent" and not agent_backend:
        raise HTTPException(status_code=400, detail="Agent bots require agent_backend")
    config = payload.get("agent_backend_config")
    if isinstance(config, dict) and config.get("skill_bundle") is not None:
        from agent_bridge.skill_registry import _name
        try:
            _name(config["skill_bundle"])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="skill_bundle must be a lowercase kebab-case registry name") from exc
        if agent_backend not in {"claude-code", "codex"}:
            raise HTTPException(status_code=400, detail="skill_bundle requires Claude Code or Codex")
    if isinstance(config, dict) and config.get("codex_transport") is not None:
        # TASK-1243: which codex runtime serves the bot's turns.
        if agent_backend != "codex":
            raise HTTPException(status_code=400, detail="codex_transport requires the Codex backend")
        if config["codex_transport"] not in CODEX_TRANSPORTS:
            raise HTTPException(status_code=400, detail="codex_transport must be one of: " + ", ".join(CODEX_TRANSPORTS))
    _normalize_agent_backend_config_model(agent_backend, payload)
    _validate_model_endpoint(payload)


#: Agent backends whose model is canonically configured via ``default_model``
#: (a catalog alias).
_CATALOG_MODEL_BACKENDS = ("claude-code", "codex")


def _normalize_agent_backend_config_model(
    agent_backend: str,
    payload: dict[str, object],
) -> None:
    """Mirror a legacy ``agent_backend_config.model`` key on profile save.

    ``default_model`` is canonical. Historical UI/config payloads may still
    carry ``model``; preserve compatibility by mirroring a non-empty value to
    ``session_model`` rather than rejecting the whole profile. Empty/non-string
    values are dropped. Agent bridges no longer read or write this metadata.
    """
    if agent_backend not in _CATALOG_MODEL_BACKENDS:
        return
    config = payload.get("agent_backend_config")
    if not isinstance(config, dict) or "model" not in config:
        return
    config = dict(config)
    legacy = config.get("model")
    if isinstance(legacy, str) and legacy.strip():
        config["session_model"] = legacy.strip()
        logger.warning(
            "bot %s: mirrored legacy agent_backend_config.model=%r to "
            "session_model for profile compatibility — the bot's "
            "model is configured via default_model",
            payload.get("slug"), legacy,
        )
    else:
        config.pop("model")
    payload["agent_backend_config"] = config


#: The Claude Code harness drives BOTH native Anthropic endpoints
#: (``claude-code``) and third-party ones tunnelled through the bridge's
#: Anthropic-compat proxy (``claude-proxy``). Same SDK, same bridge, same
#: session store — ``send_stream._build_sdk_env`` picks native-vs-proxy per
#: turn from the resolved model string alone. So the pair is a routing
#: DETAIL of the chosen endpoint, not an independent user setting: when a
#: caller changes only the endpoint, heal the harness to its sibling instead
#: of rejecting the write (the DB trigger would reject it too).
_CLAUDE_HARNESS_SIBLING = {
    "claude-code": "claude-proxy",
    "claude-proxy": "claude-code",
}


def _validate_model_endpoint(payload: dict[str, object]) -> None:
    """Validate and canonicalize the bot's normalized model assignment."""
    from ...model_catalog import (
        IncompatibleModelError,
        ModelNotFoundError,
        ProtocolCompatibility,
    )

    harness = str(payload.get("harness") or "").strip().lower()
    if not harness:
        harness = "chat"
        payload["harness"] = harness

    endpoint_id = payload.get("endpoint_id")
    if not isinstance(endpoint_id, int):
        raise HTTPException(
            status_code=422,
            detail=f"harness={harness} requires a normalized endpoint_id",
        )

    service = get_service()
    catalog = service.config.ensure_model_catalog()
    if catalog is None:
        raise HTTPException(status_code=503, detail="Model catalog unavailable")
    # Resolve by id WITHOUT the harness filter: an int ref is unambiguous, so
    # the harness would only gate compatibility — which is checked (and, for
    # the Claude pair, healed) explicitly below.
    try:
        endpoint = catalog.resolve_endpoint(endpoint_id)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except IncompatibleModelError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not ProtocolCompatibility.is_compatible(harness, endpoint.access_path):
        sibling = _CLAUDE_HARNESS_SIBLING.get(harness)
        if sibling and ProtocolCompatibility.is_compatible(sibling, endpoint.access_path):
            logger.info(
                "bot %s: endpoint %s is %s — switching harness %s -> %s",
                payload.get("slug"), endpoint_id, endpoint.access_path.key,
                harness, sibling,
            )
            harness = sibling
            payload["harness"] = sibling
        else:
            raise HTTPException(
                status_code=422,
                detail=f"endpoint_id={endpoint_id} is incompatible with harness={harness}",
            )

    expected_backend = {
        "chat": None,
        "claude-code": "claude-code",
        "claude-proxy": "claude-code",
        "codex": "codex",
    }[harness]
    payload["default_model"] = endpoint.model.key
    payload["agent_backend"] = expected_backend


def _humanize_bot_constraint_error(exc: Exception) -> str | None:
    """Extract a clean message from a bot_profiles DB constraint failure.

    The catalog migration installs a BEFORE trigger
    (``bot_profiles_model_endpoint_check``) that enforces harness↔endpoint
    compatibility on ``bot_profiles`` (e.g. a claude-code bot must point at an
    Anthropic Messages endpoint). It surfaces through SQLAlchemy as a
    ``DBAPIError`` wrapping a psycopg ``RaiseException`` whose ``RAISE
    EXCEPTION`` text is already user-friendly (e.g. ``bot loopy: harness
    claude-code requires an Anthropic Messages endpoint``), so we just lift it
    out of the SQLAlchemy wrapper.

    Returns the cleaned message, or None if this isn't a bot-constraint
    error (in which case the caller re-raises).
    """
    text = str(getattr(exc, "orig", exc) or exc)
    # Every trigger RAISE starts with ``bot <slug>: ...`` — lift the first such
    # line verbatim; that alone identifies it as our constraint.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("bot ") and ":" in stripped:
            return stripped
    # Fall back to constraint/FK-name detection for non-RAISE violations.
    markers = (
        "bot_profiles_model_endpoint_check",
        "bot_profiles_default_model_fk",
        "endpoint_id",
        "harness",
    )
    if not any(m in text for m in markers):
        return None
    return text.split("CONTEXT:")[0].strip() or None
