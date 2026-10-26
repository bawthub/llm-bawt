"""``bots_profile_patch``: one bot edits another bot's profile (TASK-1521).

The write goes through the app's ``PATCH /v1/bots/{slug}/profile``. That route
owns the reload, cache invalidation and SOUL push, and its store write takes
the bot lock. What this tool adds:

- The caller comes only from the signed turn capability; ``updated_by`` is
  stamped from it, never from an argument.
- Private bots and unknown bots get the same ``access_denied``. The private
  tag can't be added or removed here.
- Every write is a compare-and-set. The tool sends ``expected_tags`` (the tag
  set it read) and ``expected_fields`` (the caller's ``expected`` or the values
  it read) for each changed field, so a concurrent write makes this one 409
  instead of being clobbered. If the bot turned private in between, that 409
  answers ``access_denied``.
- The result holds before/after for each changed field and a ready rollback
  call. The app writes the audit row (``bot_profile_changes``) with the reason.

Capability escalation through tags is out of scope by Nick's 2026-10-26
decision: llm-bawt is open by design and BawtHub is the auth layer.
"""
from __future__ import annotations

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
#: Persona, presentation and capability-flag fields. Runtime and route fields
#: (model, harness, endpoint, backend, bot_type) and the legacy privacy flag
#: are not editable here.
ALLOWED_FIELDS = frozenset({
    TAGS, "name", "description", "color", "avatar", "default_voice", "system_prompt",
    "prompt_override_id", "requires_memory", "voice_optimized", "tts_mode", "uses_tools",
    "uses_search", "uses_home_assistant",
})


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
            private tag can't be added or removed.
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
        "rollback": {
            "changes": {field: values["before"] for field, values in diff.items()},
            "expected": {field: values["after"] for field, values in diff.items()},
        },
    })


def _detail(response) -> Any:
    try:
        return response.json().get("detail")
    except Exception:
        return response.text
