"""``bots_profile_patch``: one bot edits another bot's profile (TASK-1521).

The write goes through the app's ``PATCH /v1/bots/{slug}/profile``. That route
owns the reload, cache invalidation and SOUL push, and its store write takes
the bot lock. What this tool adds:

- The caller comes only from the signed turn capability; ``updated_by`` is
  stamped from it, never from an argument. A caller can't change its own tags
  (``self_tag_edit_refused``): another bot or a person does that.
- Private bots and unknown bots get the same ``access_denied``. The private
  tag can't be added or removed here, nor can a role carrying a protected
  capability (the grant-manager and model-switch capabilities, both global
  runtime settings): a person assigns those.
- A model switch (TASK-1818) needs a caller holding a role with the
  ``bots_model_switch_capability`` capability; unset means no bot may. The
  caller names an endpoint; the tool looks it up in the same catalog list the
  UI model picker shows for the bot's runtime and writes what a pick writes
  (``default_model``, ``endpoint_id`` and the runtime's ``harness``), so the
  app route applies the same session and prompt handling as a UI switch.
- Every write is a compare-and-set. The tool sends ``expected_tags`` (the tag
  set it read) and ``expected_fields`` (the caller's ``expected`` or the values
  it read) for each changed field, so a concurrent write makes this one 409
  instead of being clobbered. If the bot turned private in between, that 409
  answers ``access_denied``.
- The result holds before/after for each changed field and a ready rollback
  call. The app writes the audit row (``bot_profile_changes``) with the reason.

Capability escalation through another bot's tags is out of scope: llm-bawt
is open by design and BawtHub is the auth layer.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from llm_bawt.bot_access import BotAccessError, BotPolicyResolver
from llm_bawt.bot_tags import BehaviorTag
from llm_bawt.task_turn_context import TaskTurnContextError

from .registry import mcp
from .task_association import current_task_turn_context

logger = logging.getLogger(__name__)

_APP_BASE_URL = os.getenv("LLM_BAWT_APP_BASE_URL", "http://localhost:8642").rstrip("/")
_TIMEOUT_S = 30.0
_REASON_MAX = 1000

TAGS = "tags"
MODEL = "default_model"
ENDPOINT = "endpoint_id"
HARNESS = "harness"
#: The model switch: an endpoint (and optionally the model it must serve).
MODEL_FIELDS = frozenset({MODEL, ENDPOINT})
#: Persona, presentation and capability-flag fields, plus the model switch.
#: harness is derived from the endpoint, never set; backend, bot_type and the
#: legacy privacy flag are not editable here.
ALLOWED_FIELDS = frozenset({
    TAGS, "name", "description", "color", "avatar", "default_voice", "system_prompt",
    "prompt_override_id", "requires_memory", "voice_optimized", "tts_mode", "uses_tools",
    "uses_search", "uses_home_assistant", *MODEL_FIELDS,
})
#: Global runtime setting naming the role capability whose holders switch models.
MODEL_SWITCH_CAPABILITY_SETTING = "bots_model_switch_capability"
#: Agent backends whose picker scope is the chat harness (endpointScopeFor in BawtHub).
_CHAT_BACKENDS = frozenset({"", "default", "chat"})
CHAT_HARNESS = "chat"


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _error(error: str, message: str, **extra: Any) -> str:
    return _dump({"error": error, "message": message, **extra})


def _denied() -> str:
    # Unknown and private are intentionally indistinguishable.
    return _error("access_denied", "no such editable bot")


def _same(field: str, left: Any, right: Any) -> bool:
    if field == TAGS:
        return sorted(left or []) == sorted(right or [])
    return left == right


def _current(profile: dict[str, Any], field: str) -> Any:
    value = profile.get(field)
    return sorted(value or []) if field == TAGS else value


async def _resolve_access():
    from llm_bawt.utils.config import Config
    from llm_bawt.utils.db import get_shared_engine
    return await BotPolicyResolver(get_shared_engine(Config())).resolve()


def _global_settings() -> dict[str, Any]:
    from llm_bawt.service.dependencies import get_runtime_settings_store
    from llm_bawt.utils.config import Config
    return get_runtime_settings_store(Config()).get_scope_settings("global", "*")


def _setting(settings: dict[str, Any], key: str) -> str | None:
    value = settings.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _protected_roles() -> frozenset[str]:
    """Role tags this tool can't add or remove: each carries a protected capability.

    The protected capabilities are the grant manager's (TASK-1609: a bot that
    could hand those out could ordain a peer that ordains it) and the model
    switch's (TASK-1818: a bot could make a peer a model switcher). Both are
    global runtime settings; an unset one protects nothing.
    """
    from llm_bawt.bot_tags import grants
    from llm_bawt.utils.config import Config
    from llm_bawt.utils.db import get_shared_engine

    settings = _global_settings()
    capabilities = {grants.manager_capability(settings),
                    _setting(settings, MODEL_SWITCH_CAPABILITY_SETTING)} - {None}
    if not capabilities:
        return frozenset()
    with get_shared_engine(Config()).connect() as conn:
        return frozenset().union(*(grants.manager_roles(conn, cap) for cap in capabilities))


def _may_switch_models(bot_slug: str) -> bool:
    """Whether ``bot_slug`` holds a role carrying the model-switch capability."""
    from llm_bawt.bot_tags import grants
    from llm_bawt.utils.config import Config
    from llm_bawt.utils.db import get_shared_engine

    capability = _setting(_global_settings(), MODEL_SWITCH_CAPABILITY_SETTING)
    if capability is None:
        return False
    with get_shared_engine(Config()).connect() as conn:
        return grants.holds_capability(conn, bot_slug, capability)


async def _picker_endpoints(agent_backend: str | None) -> list[dict[str, Any]]:
    """The endpoints the UI model picker lists for a bot on ``agent_backend``."""
    import httpx

    backend = (agent_backend or "").strip().lower()
    params = ({"harness": CHAT_HARNESS} if backend in _CHAT_BACKENDS
              else {"agent_backend": backend})
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{_APP_BASE_URL}/v1/models/catalog/endpoints",
                                    params=params, timeout=_TIMEOUT_S)
    response.raise_for_status()
    return list(response.json().get("endpoints") or [])


async def _model_assignment(changes: dict[str, Any], profile: dict[str, Any]
                            ) -> tuple[dict[str, Any] | None, str | None]:
    """The profile fields a picker pick of ``changes[endpoint_id]`` writes, or an error.

    ``endpoint_id: null`` clears the model like the picker's clear (harness kept).
    """
    endpoint_id = changes[ENDPOINT]
    if endpoint_id is None:
        if changes.get(MODEL) is not None:
            return None, _error("model_endpoint_mismatch",
                                "clearing the endpoint clears the model; pass default_model null")
        return {MODEL: None, ENDPOINT: None}, None
    backend = (profile.get("agent_backend") or "").strip().lower()
    try:
        endpoints = await _picker_endpoints(backend)
    except Exception:
        logger.exception("bots_profile_patch: model catalog read failed")
        return None, _error("unavailable", "could not read the model catalog")
    endpoint = next((row for row in endpoints if row.get("id") == endpoint_id), None)
    if endpoint is None:
        return None, _error("unknown_endpoint",
                            f"endpoint {endpoint_id} is not in this bot's model catalog")
    if MODEL in changes and changes[MODEL] != endpoint.get("model_key"):
        return None, _error("model_endpoint_mismatch",
                            f"endpoint {endpoint_id} serves {endpoint.get('model_key')!r}, "
                            f"not {changes[MODEL]!r}", endpoint_model=endpoint.get("model_key"))
    eligibility = endpoint.get("eligibility") or {}
    if eligibility.get("eligible") is False:
        return None, _error("endpoint_ineligible", str(eligibility.get("reason")
                                                       or "this endpoint's route isn't usable"))
    harness = endpoint.get(HARNESS) or (CHAT_HARNESS if backend in _CHAT_BACKENDS else backend)
    return {MODEL: endpoint.get("model_key"), ENDPOINT: endpoint_id, HARNESS: harness}, None


async def _app(method: str, slug: str, body: dict[str, Any] | None = None):
    import httpx

    async with httpx.AsyncClient() as client:
        return await client.request(method, f"{_APP_BASE_URL}/v1/bots/{slug}/profile",
                                    json=body, timeout=_TIMEOUT_S)


def _validate(changes: Any, reason: Any, expected: Any) -> str | None:
    if not isinstance(changes, dict) or not changes:
        return _error("invalid_request", "changes must be a non-empty object of field: value")
    refused = sorted(set(changes) - ALLOWED_FIELDS)
    if refused:
        return _error("field_not_allowed", f"not editable here: {refused}",
                      allowed=sorted(ALLOWED_FIELDS))
    if MODEL in changes and ENDPOINT not in changes:
        return _error("invalid_request", "a model switch names endpoint_id (the model it serves "
                      "is written with it); default_model alone is refused")
    if ENDPOINT in changes and not (changes[ENDPOINT] is None or (
            isinstance(changes[ENDPOINT], int) and not isinstance(changes[ENDPOINT], bool))):
        return _error("invalid_request", "endpoint_id must be an integer id or null")
    if MODEL in changes and changes[MODEL] is not None and not isinstance(changes[MODEL], str):
        return _error("invalid_request", "default_model must be a model key or null")
    if TAGS in changes and (not isinstance(changes[TAGS], list)
                            or not all(isinstance(tag, str) for tag in changes[TAGS])):
        return _error("invalid_request", "tags must be a list of tag slugs (a full-set replace)")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > _REASON_MAX:
        return _error("invalid_request", f"reason is required (1-{_REASON_MAX} characters)")
    if expected is not None:
        if not isinstance(expected, dict):
            return _error("invalid_request", "expected must be an object of field: value")
        stray = sorted(set(expected) - set(changes))
        if stray:
            return _error("invalid_request", f"expected names fields not in changes: {stray}")
    return None


@mcp.tool(name="bots_profile_patch")
async def bots_profile_patch(
    bot_id: str,
    changes: dict[str, Any],
    reason: str,
    expected: dict[str, Any] | None = None,
) -> str:
    """Edit another bot's profile fields or tags as a compare-and-set.

    Args:
        bot_id: The bot to edit. Private bots are refused.
        changes: Field -> new value. ``tags`` is a full-set replace; the
            private tag and protected roles can't be added or removed.
            ``endpoint_id`` (with optional ``default_model`` it must serve)
            switches the model; it needs the model-switch capability.
        reason: Why; stored on the audit row.
        expected: Optional field -> value you read. Fields you change but leave
            out here are pinned to the value read just before the write.
    """
    try:
        turn = current_task_turn_context()
    except TaskTurnContextError:
        return _error("unauthorized", "a trusted turn capability is required")
    invalid = _validate(changes, reason, expected)
    if invalid is not None:
        return invalid
    expected = dict(expected or {})
    slug = bot_id.strip().lower() if isinstance(bot_id, str) else ""
    if TAGS in changes and slug == turn.bot_id.strip().lower():
        return _error("self_tag_edit_refused", "a bot can't change its own tags")
    if MODEL_FIELDS & set(changes):
        try:
            allowed = await asyncio.to_thread(_may_switch_models, turn.bot_id.strip().lower())
        except Exception:
            logger.exception("bots_profile_patch: model switch capability unreadable")
            return _error("unavailable", "bot tag policy unavailable")
        if not allowed:
            return _error("capability_required",
                          "switching a model needs a role with the model-switch capability")

    try:
        access = await _resolve_access()
    except BotAccessError as error:
        return _error(error.code, "bot policy unavailable")
    if slug not in access.partitions or slug in access.private_bots:
        return _denied()

    try:
        read = await _app("GET", slug)
    except Exception:
        logger.exception("bots_profile_patch: profile read failed for %s", slug)
        return _error("unavailable", "could not read the profile")
    if read.status_code == 404:
        return _denied()
    if read.status_code >= 400:
        return _error("unavailable", f"profile read failed ({read.status_code})")
    profile = read.json()
    current_tags = sorted(profile.get(TAGS) or [])
    if BehaviorTag.PRIVATE in current_tags:
        return _denied()
    if TAGS in changes and BehaviorTag.PRIVATE in changes[TAGS]:
        return _error("private_tag_locked", "the private tag can't be changed with this tool")
    if TAGS in changes:
        try:
            locked = await asyncio.to_thread(_protected_roles)
        except Exception:
            logger.exception("bots_profile_patch: protected roles unreadable")
            return _error("unavailable", "bot tag policy unavailable")
        moved = sorted(locked & (set(changes[TAGS]) ^ set(current_tags)))
        if moved:
            return _error("tag_locked", "roles carrying a protected capability can't be changed "
                          f"with this tool: {moved}", tags=moved)
    if ENDPOINT in changes:
        assignment, refused = await _model_assignment(changes, profile)
        if refused is not None:
            return refused
        changes = {**{k: v for k, v in changes.items() if k not in MODEL_FIELDS}, **assignment}

    stale = {field: _current(profile, field) for field, value in expected.items()
             if not _same(field, value, _current(profile, field))}
    if stale:
        return _error("conflict", "fields changed since you read them; nothing was saved",
                      code="fields_changed", current=stale)
    changed = {field: value for field, value in changes.items()
               if not _same(field, value, _current(profile, field))}
    if not changed:
        return _error("no_change", "every field already has the requested value")

    body: dict[str, Any] = dict(changed)
    body.update({
        "updated_by": f"bot:{turn.bot_id}",
        "change_note": reason.strip(),
        "expected_tags": current_tags,
    })
    pinned = {field: expected.get(field, profile.get(field)) for field in changed if field != TAGS}
    if pinned:
        body["expected_fields"] = pinned

    try:
        written = await _app("PATCH", slug, body)
    except Exception:
        logger.exception("bots_profile_patch: profile write failed for %s", slug)
        return _error("unavailable", "the write's outcome is unknown; read the profile before retrying")
    detail = _detail(written)
    if written.status_code == 409:
        if (isinstance(detail, dict) and detail.get("code") == "tags_changed"
                and BehaviorTag.PRIVATE in (detail.get("current_tags") or [])):
            return _denied()
        if isinstance(detail, dict):
            return _error("conflict", str(detail.get("message") or "the profile changed; nothing was saved"),
                          code=detail.get("code"),
                          current=detail.get("current") or detail.get("current_tags"))
        return _error("conflict", str(detail))
    if written.status_code == 404:
        return _denied()
    if written.status_code >= 400:
        return _error("rejected", f"the app refused the write ({written.status_code})", detail=detail)

    after = written.json()
    diff = {field: {"before": _current(profile, field), "after": _current(after, field)}
            for field in changed}
    return _dump({
        "bot_id": slug,
        "updated_by": body["updated_by"],
        "reason": body["change_note"],
        "changed": diff,
        "rollback": _rollback(diff, profile, after),
    })


def _rollback(diff: dict[str, dict[str, Any]], before: dict[str, Any],
              after: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The call that undoes this write.

    harness follows the endpoint, so it is never named; a model switch is
    undone as a whole (endpoint and model) even if only one of them changed.
    """
    fields = [field for field in diff if field != HARNESS]
    if MODEL_FIELDS & set(fields):
        fields = [field for field in fields if field not in MODEL_FIELDS] + [MODEL, ENDPOINT]
    return {"changes": {field: _current(before, field) for field in fields},
            "expected": {field: _current(after, field) for field in fields}}


def _detail(response) -> Any:
    try:
        return response.json().get("detail")
    except Exception:
        return response.text
