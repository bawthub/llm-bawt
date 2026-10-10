"""MCP tool letting an agent bot view or switch its own model (TASK-1047).

``self_model`` is a thin client over two existing, server-validated paths:

* **Switch** = the chat model picker's write: ``PATCH /v1/bots/{slug}/profile``
  with ``{"endpoint_id": N}``. The server canonicalizes ``default_model``, heals
  the claude-code <-> claude-proxy harness pair, reloads the registry and
  invalidates cached instances. Nothing about compatibility is duplicated here.
* **Continue** = a durable ``when_idle`` inter-bot delivery to the switched bot.
  The current turn cannot change model mid-flight, so the follow-up must run as
  a separate turn after this one ends. ``when_idle`` is mandatory: the default
  steer mode would inject the prompt into the still-running old-model turn.

The tool addresses ``bot_id`` like the other ``self_*`` tools (LAN trust model).
When a trusted current-turn capability is present, the caller is recorded as
``changed_by`` and used as the continuation's sender, so a cross-bot switch is
attributed correctly. Mira is refused unless she is the trusted caller.

Continuation scope (Snark review, TASK-1047): a bot-authored delivery always
runs as DEFAULT_USER in the target's ACTIVE thread. ``continue_prompt`` is
therefore accepted only when that is where the follow-up belongs: a trusted
caller context, the default user, and, for a self-switch, the caller's own
active thread (not a task/archived thread). Anything else is refused BEFORE
the switch, so nothing lands in the wrong conversation and no switch is left
half-done. The idempotency key is scoped to the trusted calling turn, so a
retried call re-attaches instead of queueing a second follow-up.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Literal

import httpx

from ..task_turn_context import TaskTurnContext, TaskTurnContextError
from .server import mcp

logger = logging.getLogger(__name__)

_APP_BASE_URL = os.getenv("LLM_BAWT_APP_BASE_URL", "http://localhost:8642").rstrip("/")
_PROTECTED_BOTS = frozenset({"mira"})
# Harnesses whose bots can move between each other via the server-side heal in
# ``_validate_model_endpoint``; the picker offers the union for the same reason.
_HARNESS_OPTIONS: dict[str, tuple[str, ...]] = {
    "claude-code": ("claude-code", "claude-proxy"),
    "claude-proxy": ("claude-code", "claude-proxy"),
    "codex": ("codex",),
    "chat": ("chat",),
}
_CONTINUE_KIND = "MODEL_SWITCH_CONTINUE"


def _trusted_caller() -> TaskTurnContext | None:
    """Return the signed current-turn context, or None when absent/invalid."""
    from .task_association import current_task_turn_context

    try:
        return current_task_turn_context()
    except TaskTurnContextError:
        return None


async def _get_active_session(bot_id: str) -> dict:
    """The DEFAULT_USER's active thread for ``bot_id`` (where deliveries run)."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_APP_BASE_URL}/v1/sessions/active", params={"bot_id": bot_id}, timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()


async def _continuation_scope_error(bot_id: str, context: TaskTurnContext | None) -> str | None:
    """Why a continuation would run in the wrong thread, or None if it is safe."""
    if context is None:
        return ("continue_prompt needs a trusted current-turn context. Switch "
                "without it and ask the user to continue.")
    caller = context.bot_id.strip().lower()
    try:
        active = await _get_active_session(caller)
    except Exception as exc:
        return _http_error("Could not resolve the default user's active thread", exc)
    if context.user_id != active.get("user_id"):
        return ("continue_prompt is only supported for the default user's "
                "conversation; a queued follow-up would run in their thread, not this one.")
    if caller == bot_id and context.session_id != active.get("id"):
        return ("This turn is not in your active thread (e.g. a task thread), so the "
                "follow-up would land in the main chat. Switch without continue_prompt.")
    return None


async def _get_profile(bot_id: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{_APP_BASE_URL}/v1/bots/{bot_id}/profile", timeout=15.0)
        resp.raise_for_status()
        return resp.json()


async def _patch_endpoint(bot_id: str, endpoint_id: int) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.patch(
            f"{_APP_BASE_URL}/v1/bots/{bot_id}/profile",
            json={"endpoint_id": endpoint_id},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()


async def _list_endpoints(harness: str) -> list[dict]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_APP_BASE_URL}/v1/models/catalog/endpoints",
            params={"harness": harness},
            timeout=15.0,
        )
        resp.raise_for_status()
        return resp.json().get("endpoints") or []


async def _enqueue_continue(**kwargs: Any) -> dict:
    """Queue the follow-up turn through the shared durable-delivery client."""
    from .inter_bot_tools import _enqueue_durable

    return await _enqueue_durable(prefer_steer=False, **kwargs)


def _model_summary(profile: dict) -> dict:
    return {
        "endpoint_id": profile.get("endpoint_id"),
        "model": profile.get("default_model"),
        "harness": profile.get("harness"),
        "agent_backend": profile.get("agent_backend"),
    }


def _http_error(prefix: str, exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            detail = exc.response.json().get("detail")
        except Exception:
            detail = exc.response.text
        return f"{prefix} ({exc.response.status_code}): {detail}"
    return f"{prefix}: {exc}"


async def _options_for(harness: str | None) -> list[dict]:
    seen: dict[int, dict] = {}
    for option_harness in _HARNESS_OPTIONS.get(str(harness or "").lower(), ()):
        for row in await _list_endpoints(option_harness):
            seen.setdefault(row["id"], {
                "endpoint_id": row["id"],
                "model": row.get("model_key"),
                "access_path": row.get("access_path_key"),
                "harness": option_harness,
                "context_window": row.get("context_window_override")
                or row.get("default_context_window"),
            })
    return sorted(seen.values(), key=lambda o: (o["harness"], o["model"] or "", o["access_path"] or ""))


def _continue_message(old: dict, new: dict, changed_by: str, prompt: str) -> str:
    return (
        f"[self_model] {changed_by} switched this bot from {old.get('model')} "
        f"(endpoint {old.get('endpoint_id')}) to {new.get('model')} "
        f"(endpoint {new.get('endpoint_id')}). This turn runs on the new model.\n\n"
        f"{prompt}"
    )


def _continue_key(bot_id: str, endpoint_id: int, turn_id: str, prompt: str) -> str:
    # Stable within one calling turn (a retried call re-attaches to the same
    # delivery); unique across turns so a later identical switch still delivers.
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
    return f"self_model:{bot_id}:{endpoint_id}:{turn_id}:{digest}"


@mcp.tool(name="self_model")
async def self_model(
    bot_id: str,
    action: Literal["view", "switch"] = "view",
    endpoint_id: int | None = None,
    continue_prompt: str | None = None,
) -> dict:
    """View or switch a bot's model; optionally queue one follow-up turn on it.

    view: current endpoint/model/harness plus compatible endpoints to choose from.
    switch: persist ``endpoint_id`` through the profile PATCH (server validates,
    heals claude-code <-> claude-proxy). Applies from the bot's NEXT turn; the
    in-flight turn keeps its model. With ``continue_prompt``, also queue a
    durable ``when_idle`` delivery so the bot resumes on the new model once the
    current turn ends. ``continue_prompt`` is refused (before any switch) unless
    the follow-up would run in the right thread; see the module docstring. The
    switch is not rolled back if queueing itself fails.
    """
    bot_id = (bot_id or "").strip().lower()
    context = _trusted_caller()
    caller = context.bot_id.strip().lower() if context else None
    base = {"bot_id": bot_id, "action": action}
    if not bot_id or bot_id == "default":
        return {**base, "error": "Pass an explicit bot slug (your own); 'default' is refused."}
    if bot_id in _PROTECTED_BOTS and caller != bot_id:
        return {**base, "error": f"'{bot_id}' is a protected bot; only it may change its own model."}

    try:
        before = await _get_profile(bot_id)
    except Exception as exc:
        return {**base, "error": _http_error("Could not read profile", exc)}
    current = _model_summary(before)
    harness = str(before.get("harness") or "").lower()

    if action == "view":
        if harness == "openclaw":
            return {**base, "current": current, "options": [],
                    "note": "OpenClaw bots have no catalog endpoint to switch."}
        try:
            options = await _options_for(harness)
        except Exception as exc:
            return {**base, "current": current, "error": _http_error("Could not list endpoints", exc)}
        return {**base, "current": current, "options": options}

    if action != "switch":
        return {**base, "error": f"Unknown action '{action}'. Valid: view, switch."}
    if harness == "openclaw":
        return {**base, "current": current, "error": "OpenClaw bots have no catalog endpoint to switch."}
    if not isinstance(endpoint_id, int) or isinstance(endpoint_id, bool):
        return {**base, "current": current, "error": "action='switch' requires an integer endpoint_id (see view)."}
    prompt = (continue_prompt or "").strip()
    if prompt:
        scope_error = await _continuation_scope_error(bot_id, context)
        if scope_error:
            return {**base, "current": current, "switched": False, "error": scope_error}

    try:
        after = await _patch_endpoint(bot_id, endpoint_id)
    except Exception as exc:
        return {**base, "previous": current, "switched": False,
                "error": _http_error("Switch rejected", exc)}
    new = _model_summary(after)
    changed_by = caller or bot_id
    result: dict[str, Any] = {
        **base,
        "switched": True,
        "changed": new["endpoint_id"] != current["endpoint_id"],
        "previous": current,
        "current": new,
        "changed_by": changed_by,
        "note": "Applies from the next turn; this in-flight turn keeps its model.",
    }
    if result["changed"] and new.get("agent_backend") == "codex":
        # resolve_agent_session_key only skips its model guard for claude_code.
        result["note"] += " Codex sessions are model-bound: the next turn starts a fresh provider session."
    logger.info(
        "self_model: %s switched %s endpoint %s -> %s (%s)",
        changed_by, bot_id, current["endpoint_id"], new["endpoint_id"], new["model"],
    )
    if not prompt:
        return result

    receipt = await _enqueue_continue(
        target_bot_id=bot_id,
        message=_continue_message(current, new, changed_by, prompt),
        sender_bot_id=changed_by,
        max_tokens=None,
        temperature=0.7,
        timeout_seconds=1800.0,
        idempotency_key=_continue_key(bot_id, endpoint_id, context.turn_id, prompt),
        project_id=None,
        task_id=None,
        message_kind=_CONTINUE_KIND,
        metadata={"self_model": {"from": current, "to": new, "changed_by": changed_by}},
        session_policy="continue",
        reset_session_before_delivery=None,
        retain_history=None,
        reset_reason=None,
    )
    if receipt.get("delivery_id"):
        result["continue"] = {
            "queued": True,
            "delivery_id": receipt["delivery_id"],
            "status": receipt.get("status"),
            "delivery": "when_idle",
        }
    else:
        result["continue"] = {"queued": False, "error": receipt.get("error") or "delivery not queued"}
    return result
