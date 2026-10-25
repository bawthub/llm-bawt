"""MCP tools for immediate and durable inter-bot communication."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

import httpx
from pydantic import Field

from llm_bawt.bot_access import BotAccessError, BotPolicyResolver
from llm_bawt.bot_access.context import current_bot_access
from llm_bawt.bot_tags import BehaviorTag
from llm_bawt.bot_types import normalize_bot_type
from .bot_access_deliveries import ScopedDeliveryReader
from llm_bawt.inter_bot_send_targets import SelectorError
from .inter_bot_fanout import selector_error, send_batch, single_explicit_target
from .inter_bot_send_mode import (
    STEER_ONLY, STEER_OR_IDLE, WHEN_IDLE, SendMode, SendModeError,
    require_one_recipient, resolve_send_mode,
)
from .inter_bot_sender import SenderIdentityError, resolve_sender_bot_id
from .server import _get_storage, mcp
from ..task_turn_context import TaskTurnContextError, is_delivery_turn_id

logger = logging.getLogger(__name__)
_APP_BASE_URL = "http://localhost:8642"
# bots_deliveries_list is a bounded diagnostic list (TASK-1239).
_DELIVERY_LIST_DEFAULT_LIMIT = 10
_DELIVERY_LIST_MAX_LIMIT = 50


def _compat_hook(name: str, default):
    """Honor legacy tests/importers that monkeypatch helpers on server.py."""
    from . import server

    candidate = getattr(server, name, default)
    return candidate if candidate is not default else default
_BOT_SEND_WAIT_SETTING = "bot_send_wait_seconds"
_bot_send_settings_resolver = None


def _bot_send_wait_ceiling_seconds() -> float:
    global _bot_send_settings_resolver
    try:
        if _bot_send_settings_resolver is None:
            from llm_bawt.runtime_settings import RuntimeSettingsResolver
            _bot_send_settings_resolver = RuntimeSettingsResolver(
                config=_get_storage().config, bot=None,
            )
        value = float(_bot_send_settings_resolver.resolve(
            _BOT_SEND_WAIT_SETTING, fallback=300,
        ))
        return value if value > 0 else 300.0
    except Exception:
        logger.warning("Could not resolve %s; using 300s", _BOT_SEND_WAIT_SETTING)
        return 300.0


async def _find_delivery_by_key(
    sender_bot_id: str, target_bot_id: str, idempotency_key: str,
) -> dict | None:
    """Return the durable row already owning this key, if any (read-only)."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{_APP_BASE_URL}/v1/inter-bot-deliveries",
            params={
                "sender_bot_id": sender_bot_id,
                "target_bot_id": target_bot_id,
                "idempotency_key": idempotency_key,
                "limit": 5,
            },
            timeout=10.0,
        )
        response.raise_for_status()
        rows = response.json().get("deliveries") or []
    for row in rows:
        author = row.get("author") or {}
        if author.get("entity_type", "bot") == "bot":
            return row
    return None


async def _get_json(path: str) -> dict | None:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{_APP_BASE_URL}{path}", timeout=10.0)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()


_TERMINAL_DELIVERY = {"DELIVERED", "FAILED", "CANCELLED"}
_now = time.monotonic  # patch point; never patch time.monotonic (event loop)


async def _waited_reply_origin(sender_bot_id: str, target_bot_id: str) -> str | None:
    """The awaited inbound delivery whose sender is ``target_bot_id``, if any."""
    origin = await _waited_inbound(sender_bot_id)
    if origin and origin[1] == target_bot_id.strip().lower():
        return origin[0]
    return None


async def _waited_inbound(sender_bot_id: str) -> tuple[str, str] | None:
    """``(delivery_id, waiting sender)`` of this awaited inbound turn, if any.

    Read from signed, request-local context only.
    """
    from .task_association import current_task_turn_context

    try:
        context = current_task_turn_context()
    except TaskTurnContextError:
        return None  # Other backends and external MCP callers have no signed turn.
    if context.bot_id != sender_bot_id.strip().lower() or not is_delivery_turn_id(context.turn_id):
        return None
    delivery_id = context.turn_id.replace("turn-delivery-", "delivery-", 1)
    try:
        record = await _compat_hook("_get_json", _get_json)(
            f"/v1/inter-bot-deliveries/{delivery_id}"
        )
    except Exception as exc:
        logger.warning("Cannot check waited callback origin %s: %s", delivery_id, exc)
        return None
    if (record and record.get("turn_id") == context.turn_id
            and record.get("user_message_id") == context.trigger_message_id
            and record.get("target_bot_id") == context.bot_id
            and record.get("sender_bot_id")
            and record.get("reply_mode") == "waited"):
        return delivery_id, str(record["sender_bot_id"])
    return None


async def _await_delivery(receipt: dict, wait_seconds: float) -> dict:
    """Bounded wait on ONE durable delivery; never re-sends anything.

    The receipt is already persisted before this runs, so a caller interrupted
    mid-wait (steer, Stop, timeout) can retry with the same idempotency key and
    re-attach to the same delivery/turn instead of creating another.
    """
    get_json = _compat_hook("_get_json", _get_json)
    delivery_id = receipt.get("delivery_id")
    deadline = _now() + max(1.0, wait_seconds)
    current = receipt
    delay = 0.5
    while str(current.get("status") or "").upper() not in _TERMINAL_DELIVERY:
        remaining = deadline - _now()
        if remaining <= 0:
            return {
                **current,
                "success": False,
                "dispatched": True,
                "error": "timeout",
                "in_flight": True,
                "warning": (
                    f"Target bot did not finish within {wait_seconds:.0f}s. The delivery "
                    "is durable and still progressing. DO NOT send a new message; inspect "
                    f"bots_delivery_get('{delivery_id}') or retry with the SAME "
                    "idempotency_key to re-attach."
                ),
                "content": "",
            }
        await asyncio.sleep(min(delay, remaining))
        delay = min(delay * 2, 3.0)
        fetched = await get_json(f"/v1/inter-bot-deliveries/{delivery_id}")
        if fetched is None:
            return {**current, "success": False, "dispatched": True,
                    "error": "delivery disappeared", "content": ""}
        current = {**current, **fetched}

    status = str(current.get("status")).upper()
    if status != "DELIVERED":
        return {**current, "success": False, "dispatched": True,
                "error": current.get("last_error") or f"delivery {status.lower()}",
                "content": ""}
    turn = await get_json(f"/v1/turn-logs/{current.get('turn_id')}") or {}
    return {
        **current,
        "success": True,
        "dispatched": True,
        "content": turn.get("response") or "",
        "response_model": current.get("response_model") or turn.get("model"),
        "turn_status": turn.get("status"),
    }


async def _check_bot_in_turn(target_bot_id: str) -> dict | None:
    """Return active-turn data, failing open for legacy immediate sends."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{_APP_BASE_URL}/v1/bots/{target_bot_id}/in-turn", timeout=5.0
            )
            response.raise_for_status()
            data = response.json()
        return data if data.get("in_turn") else None
    except Exception as exc:
        logger.warning("in-turn check for %s failed (allowing send): %s", target_bot_id, exc)
        return None


def _policy_resolver() -> BotPolicyResolver:
    from llm_bawt.utils.config import Config
    from llm_bawt.utils.db import get_shared_engine

    return BotPolicyResolver(get_shared_engine(Config()))


async def _target_refusal(target_bot_id: str) -> dict | None:
    """A new send to an ``inactive`` bot is refused and never queued (TASK-1329)."""
    try:
        access = await _policy_resolver().resolve()
    except BotAccessError as exc:
        return {"error": exc.code}
    target = (target_bot_id or "").strip().lower()
    if access.has_tag(target, BehaviorTag.INACTIVE):
        return {"error": "target_inactive",
                "detail": f"'{target}' is inactive; it must be reactivated before it can receive messages."}
    return None


async def _enqueue_durable(
    *,
    prefer_steer: bool,
    target_bot_id: str,
    message: str,
    sender_bot_id: str,
    max_tokens: int | None,
    temperature: float,
    timeout_seconds: float,
    idempotency_key: str | None,
    project_id: str | None,
    task_id: str | None,
    message_kind: str | None,
    metadata: dict[str, Any] | None,
    session_policy: str | None,
    reset_session_before_delivery: bool | None,
    retain_history: bool | None,
    reset_reason: str | None,
    reply_mode: str = "async",
    headers: dict[str, str] | None = None,
    steer_only: bool = False,
) -> dict:
    """``headers`` is for first-party tool code only (the self-model continue
    forwards its signed turn capability); bots_send_message never sets it.

    ``steer_only`` is sent only when set, so every other body is unchanged.
    """
    delivery_name = STEER_ONLY if steer_only else STEER_OR_IDLE if prefer_steer else WHEN_IDLE
    body = {
        "reply_mode": reply_mode,
        "target_bot_id": target_bot_id,
        "message": message,
        "sender_bot_id": sender_bot_id,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "timeout_seconds": timeout_seconds,
        "idempotency_key": idempotency_key,
        "project_id": project_id,
        "task_id": task_id,
        "message_kind": message_kind,
        "metadata": metadata or {},
        "prefer_steer": prefer_steer,
        "session_policy": session_policy,
        "reset_session_before_delivery": reset_session_before_delivery,
        "retain_history": retain_history,
        "reset_reason": reset_reason,
    }
    if steer_only:
        body["steer_only"] = True
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{_APP_BASE_URL}/v1/inter-bot-deliveries",
                json=body,
                headers=headers,
                timeout=15.0,
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        try:
            body = exc.response.json()
        except Exception:
            body = None
        if isinstance(body, dict) and body.get("delivery_id"):
            return {**body, "status_code": exc.response.status_code}
        detail: Any = body.get("detail") if isinstance(body, dict) else exc.response.text
        return {
            "success": False,
            "queued": False,
            "delivery": delivery_name,
            "bot_id": target_bot_id,
            "sender": sender_bot_id,
            "error": str(detail),
            "status_code": exc.response.status_code,
        }
    except Exception as exc:
        return {
            "success": False,
            "queued": False,
            "delivery": delivery_name,
            "bot_id": target_bot_id,
            "sender": sender_bot_id,
            "error": str(exc) or exc.__class__.__name__,
        }


async def _send_fanout(
    targets: Any, mode: SendMode, common: dict[str, Any], *, separate: bool,
) -> dict:
    """One batch to every bot ``targets`` selects (TASK-1431 design r3 §7).

    In an awaited inbound turn the waiting sender already gets this turn's
    final answer, so it is left out of the batch and reported as skipped.
    """
    exclude: list[str] = []
    skipped: list[dict[str, str]] = []
    if not separate:
        origin = await _waited_inbound(common["sender_bot_id"])
        if origin:
            exclude.append(origin[1])
            skipped.append({
                "target": origin[1], "in_reply_to_delivery_id": origin[0],
                "reason": "awaiting this turn's final answer",
            })
    fields = {key: value for key, value in common.items() if value is not None}
    result = await send_batch(
        _APP_BASE_URL, targets=targets, mode=mode, exclude=exclude, fields=fields,
    )
    return {**result, "skipped": skipped} if skipped else result


@mcp.tool(name="bots_send_message")
async def send_message_to_bot(
    target_bot_id: str = "",
    *,
    message: str,
    sender_bot_id: str = "",
    max_tokens: int | None = None,
    temperature: float = 0.7,
    fire_and_forget: bool | None = None,
    timeout_seconds: float = 300.0,
    force: bool = False,
    wait_for_reply: bool = False,
    queue_if_busy: bool = False,
    delivery: str | None = None,
    idempotency_key: str | None = None,
    project_id: str | None = None,
    task_id: str | None = None,
    message_kind: str | None = None,
    metadata: dict[str, Any] | None = None,
    session_policy: str | None = None,
    reset_session_before_delivery: bool | None = None,
    retain_history: bool | None = None,
    reset_reason: str | None = None,
    separate_message_reason: str | None = None,
    targets: dict[str, Any] | list[str] | None = None,
) -> dict:
    """Send a message to another bot without creating concurrent agent turns.

    Every send is durable and returns a stable ``delivery_id``; an
    ``idempotency_key`` deduplicates across ALL modes, so retrying the same key
    re-attaches to the original delivery and can never start a second turn.

    Mode flags (validated together before any side effect; contradictions are
    rejected with ``dispatched: false``):

    * default — steer an active Claude Code turn in place, else start exactly
      one safe idle turn. Returns immediately.
    * ``delivery="when_idle"`` (or ``queue_if_busy=True``) — never steer; wait
      for one separate idle turn. Returns immediately.
    * ``delivery="steer_only"`` — steer an executing Claude Code turn, else
      settle as not needed; never starts a turn. Cannot be waited.
    * ``wait_for_reply=True`` (or ``fire_and_forget=False``) — when_idle delivery
      plus a bounded wait for the reply. A busy target is rejected without side
      effects unless the key already owns a delivery. Cannot be combined with
      ``delivery="steer_or_idle"``; task handoffs (``task_id``) require a key.

    Name the recipient with exactly one of ``target_bot_id`` or ``targets``.
    ``targets`` is ``{"bots": [...]}`` (or a bare list), ``{"all_agents": true}``,
    ``{"backends": [...]}`` or ``{"tags": [...]}``; you are left out. One
    explicit bot returns the usual receipt. More returns ``{batch_id,
    deliveries, refused}``, needs an ``idempotency_key``, never waits or
    resets, and is limited to 3 new batches per 10 minutes.

    ``force=True`` is accepted for compatibility but never permits concurrency.
    In a signed inbound waited turn, a callback to that sender is redundant:
    answer in this turn's final response instead. Pass ``separate_message_reason``
    only for an intentional, distinct follow-up to the waiting sender.
    Results carry ``mode`` and ``dispatched`` so a caller can tell whether the
    target received anything.

    ``sender_bot_id`` is optional: the sender is taken from your signed turn.
    Passing a different bot id is rejected. Callers without a signed turn must
    pass their own bot id.
    """
    try:
        sender_bot_id = resolve_sender_bot_id(sender_bot_id)
    except SenderIdentityError as exc:
        return {
            "success": False,
            "dispatched": False,
            "error": str(exc),
            "bot_id": target_bot_id,
        }
    try:
        require_one_recipient(target_bot_id, targets)
        mode = resolve_send_mode(
            delivery=delivery,
            queue_if_busy=queue_if_busy,
            fire_and_forget=fire_and_forget,
            wait_for_reply=wait_for_reply,
            task_id=task_id,
            idempotency_key=idempotency_key,
        )
    except SendModeError as exc:
        return {
            "success": False,
            "dispatched": False,
            "error": str(exc),
            "bot_id": target_bot_id,
            "sender": sender_bot_id,
        }

    if separate_message_reason is not None and len(separate_message_reason.strip()) > 500:
        return {"success": False, "dispatched": False,
                "error": "separate_message_reason must be at most 500 characters"}
    common = dict(
        message=message,
        sender_bot_id=sender_bot_id,
        max_tokens=max_tokens,
        temperature=temperature,
        # The target turn's own bridge budget; never the caller's wait budget.
        timeout_seconds=max(timeout_seconds, 1800.0),
        idempotency_key=idempotency_key,
        project_id=project_id,
        task_id=task_id,
        message_kind=message_kind,
        metadata=metadata,
        session_policy=session_policy,
        reset_session_before_delivery=reset_session_before_delivery,
        retain_history=retain_history,
        reset_reason=reset_reason,
    )
    if targets is not None:
        try:
            single = single_explicit_target(targets, sender_bot_id)
        except SelectorError as exc:
            return selector_error(exc, sender_bot_id)
        if single is None:
            return await _send_fanout(
                targets, mode, common, separate=bool((separate_message_reason or "").strip()))
        target_bot_id = single
    if not (separate_message_reason or "").strip():
        origin_id = await _waited_reply_origin(sender_bot_id, target_bot_id)
        if origin_id:
            return {
                "success": False,
                "dispatched": False,
                "mode": mode.label,
                "in_reply_to_delivery_id": origin_id,
                "error": (
                    "The sender receives this awaited turn's final answer through its "
                    "original tool call or durable receipt. Put the answer in your final "
                    "response instead of sending a duplicate callback. For an intentional "
                    "distinct follow-up, set separate_message_reason."
                ),
            }

    key = (idempotency_key or "").strip()
    find_existing = _compat_hook("_find_delivery_by_key", _find_delivery_by_key)
    existing = None
    refusal = await _target_refusal(target_bot_id)
    if refusal is not None:
        # A same-key retry reattaches to the delivery accepted before the
        # target went inactive; only genuinely new sends are refused.
        if key:
            try:
                existing = await find_existing(
                    sender_bot_id, target_bot_id.strip().lower(), key)
            except Exception as exc:
                logger.warning("key lookup for refused send failed: %s", exc)
        if existing is None:
            return {"success": False, "dispatched": False, "mode": mode.label,
                    "bot_id": target_bot_id, "sender": sender_bot_id, **refusal}

    durable_kwargs = dict(common, target_bot_id=target_bot_id, reply_mode=mode.label)
    enqueue = _compat_hook("_enqueue_durable", _enqueue_durable)
    if not mode.wait:
        if existing is not None:
            receipt = {**existing, "duplicate": True}
        else:
            receipt = await enqueue(
                prefer_steer=mode.prefer_steer,
                **({"steer_only": True} if mode.steer_only else {}),
                **durable_kwargs,
            )
        return {**receipt, "mode": mode.label,
                "dispatched": bool(receipt.get("delivery_id"))}

    ceiling = _compat_hook(
        "_bot_send_wait_ceiling_seconds", _bot_send_wait_ceiling_seconds
    )()
    wait_seconds = min(timeout_seconds, ceiling)
    clamp = (
        {
            "timeout_clamped": True,
            "requested_timeout_seconds": timeout_seconds,
            "effective_timeout_seconds": wait_seconds,
        }
        if timeout_seconds > ceiling else {}
    )

    if existing is None and key:
        existing = await find_existing(sender_bot_id, target_bot_id.strip().lower(), key)
    if existing is None:
        active = await _compat_hook("_check_bot_in_turn", _check_bot_in_turn)(target_bot_id)
        if active is not None:
            return {
                "success": False,
                "dispatched": False,
                "mode": mode.label,
                "in_turn": True,
                "bot_id": target_bot_id,
                "sender": sender_bot_id,
                "content": "",
                "turn_id": active.get("turn_id"),
                "turn_status": active.get("status"),
                "note": (
                    f"Agent '{target_bot_id}' is in turn — waited send not started. "
                    "Use the default asynchronous mode to durably steer or safely queue it."
                ),
                **clamp,
            }
        receipt = await enqueue(prefer_steer=False, **durable_kwargs)
        if not receipt.get("delivery_id"):
            return {**receipt, "mode": mode.label, "dispatched": False, **clamp}
    else:
        receipt = {**existing, "duplicate": True}

    result = await _compat_hook("_await_delivery", _await_delivery)(receipt, wait_seconds)
    return {
        **result,
        "mode": mode.label,
        "bot_id": target_bot_id,
        "sender": sender_bot_id,
        **clamp,
    }


@mcp.tool(name="bots_delivery_get")
async def get_delivery(delivery_id: str) -> dict:
    """Inspect one durable inter-bot delivery by stable delivery ID."""
    if access := current_bot_access():
        return await ScopedDeliveryReader(access).get(delivery_id)
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{_APP_BASE_URL}/v1/inter-bot-deliveries/{delivery_id}", timeout=10.0
        )
        if response.status_code == 404:
            return {"delivery_id": delivery_id, "error": "not found"}
        response.raise_for_status()
        return response.json()


@mcp.tool(name="bots_deliveries_list")
async def list_deliveries(
    bot_id: str | None = None,
    sender_bot_id: str | None = None,
    target_bot_id: str | None = None,
    status: str | None = None,
    active_only: bool = False,
    task_id: str | None = None,
    since_minutes: int | None = None,
    include_chat: bool = False,
    limit: int = _DELIVERY_LIST_DEFAULT_LIMIT,
) -> dict:
    """List compact delivery rows; full receipts via bots_delivery_get."""
    if access := current_bot_access():
        return await ScopedDeliveryReader(access).list(
            bot_id=bot_id, sender_bot_id=sender_bot_id, target_bot_id=target_bot_id,
            status=status, active_only=active_only, task_id=task_id,
            since_minutes=since_minutes, include_chat=include_chat,
            limit=min(max(int(limit), 1), _DELIVERY_LIST_MAX_LIMIT),
        )
    params: dict[str, Any] = {
        key: value for key, value in {
            "bot_id": bot_id,
            "sender_bot_id": sender_bot_id,
            "target_bot_id": target_bot_id,
            "status": status,
            "task_id": task_id,
        }.items() if value
    }
    if active_only:
        params["active_only"] = "true"
    if since_minutes is not None and since_minutes > 0:
        since = datetime.now(timezone.utc) - timedelta(minutes=since_minutes)
        params["since"] = since.isoformat()
    params["include_chat"] = "true" if include_chat else "false"
    params["view"] = "compact"
    params["limit"] = min(max(int(limit), 1), _DELIVERY_LIST_MAX_LIMIT)
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{_APP_BASE_URL}/v1/inter-bot-deliveries", params=params, timeout=10.0,
        )
        if response.status_code == 422:
            try:
                detail = response.json().get("detail")
            except ValueError:
                detail = response.text
            return {"error": detail}
        response.raise_for_status()
        return response.json()


@mcp.tool(name="bots_delivery_cancel")
async def cancel_delivery(delivery_id: str) -> dict:
    """Cancel a durable delivery while it is still QUEUED."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{_APP_BASE_URL}/v1/inter-bot-deliveries/{delivery_id}/cancel",
            timeout=10.0,
        )
        if response.status_code in (404, 409):
            try:
                detail = response.json().get("detail")
            except Exception:
                detail = response.text
            return {"delivery_id": delivery_id, "error": detail, "status_code": response.status_code}
        response.raise_for_status()
        return response.json()


@mcp.tool(name="agent_context_health")
async def agent_context_health(
    bot_id: str,
    user_id: str | None = None,
) -> dict:
    """Inspect resident context estimate, headroom, thresholds, and capabilities."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{_APP_BASE_URL}/v1/agent-context/health",
            params={k: v for k, v in {"bot_id": bot_id, "user": user_id}.items() if v is not None},
            timeout=10.0,
        )
        response.raise_for_status()
        return response.json()


@mcp.tool(name="agent_context_reset")
async def agent_context_reset(
    bot_id: str,
    session_policy: str,
    reason: str = "agent-requested context maintenance",
    user_id: str | None = None,
) -> dict:
    """Safely reset an idle agent session without deleting durable history."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{_APP_BASE_URL}/v1/agent-context/reset",
            json={
                "bot_id": bot_id,
                "user": user_id,
                "session_policy": session_policy,
                "reason": reason,
            },
            timeout=15.0,
        )
        if response.status_code in (409, 422):
            return {"success": False, "status_code": response.status_code, "error": response.json().get("detail")}
        response.raise_for_status()
        return response.json()


@mcp.tool(name="agent_context_compact")
async def agent_context_compact(
    bot_id: str,
    idempotency_key: str,
    sender_bot_id: str = "",
) -> dict:
    """Queue one durable Claude /compact maintenance turn when the bot is idle.

    The sender comes from your signed turn; ``sender_bot_id`` is only needed
    by callers without one and must match it otherwise.
    """
    key = (idempotency_key or "").strip()
    if not key:
        return {"success": False, "error": "idempotency_key is required"}
    try:
        sender_bot_id = resolve_sender_bot_id(sender_bot_id)
    except SenderIdentityError as exc:
        return {"success": False, "error": str(exc)}
    health = await agent_context_health(bot_id)
    if not (health.get("capabilities") or {}).get("compact"):
        return {
            "success": False,
            "error": f"Backend {health.get('backend')!r} does not support compact",
        }
    return await _enqueue_durable(
        prefer_steer=False,
        target_bot_id=bot_id,
        message="/compact",
        sender_bot_id=sender_bot_id,
        max_tokens=None,
        temperature=0.0,
        timeout_seconds=1800.0,
        idempotency_key=key,
        project_id=None,
        task_id=None,
        message_kind="CONTEXT_MAINTENANCE",
        metadata={"context_action": "compact"},
        session_policy="continue",
        reset_session_before_delivery=None,
        retain_history=None,
        reset_reason=None,
    )


def _profile_rows() -> list:
    from llm_bawt.runtime_settings import BotProfileStore
    from llm_bawt.utils.config import Config

    try:
        store = BotProfileStore(Config())
        if store.engine is None:
            raise RuntimeError("bot profiles unavailable")
        return store.list_all()
    except Exception:
        raise BotAccessError("policy_unavailable") from None


def _live_states(profiles: list) -> dict[str, dict]:
    from llm_bawt.bot_live_state import BotIdentity, BotLiveStateReader
    from llm_bawt.utils.config import Config
    from llm_bawt.utils.db import get_shared_engine

    try:
        engine = get_shared_engine(Config())
    except Exception:
        logger.warning("bot live state: database unavailable", exc_info=True)
        engine = None
    return BotLiveStateReader(engine).read(
        BotIdentity(row.slug, row.bot_type, row.agent_backend) for row in profiles
    )


@mcp.tool(name="bots_list_available")
async def list_available_bots(
    tag: Annotated[str | None, Field(description="Only bots with this tag.")] = None,
    include_state: Annotated[bool, Field(
        description="Add state.turn.kind (idle/reserved/executing/ambiguous) and the other live fields."
    )] = True,
) -> list[dict]:
    """List bots available as inter-bot message targets, each with its tags.

    Read in-process (TASK-1329). Private and inactive bots are never listed. An
    unreadable policy or profile table raises ``policy_unavailable``, never an
    unfiltered or partial list. An unknown ``tag`` raises instead of matching
    nothing.

    ``state`` (TASK-1400) is an advisory, unlocked read: turn comes from the
    admission classifier, so it matches what a send would see at that moment.
    A source that can't be read is named in ``state.unavailable`` with null
    fields, never reported as idle. Task counts live in BawtHub (``tasks_list``).
    """
    # Policy first: a profile created after the snapshot is unknown, so hidden.
    access = await _policy_resolver().resolve()
    wanted = None
    if tag is not None:
        wanted = tag.strip().lower()
        if wanted not in access.tags.catalog:
            raise ValueError(f"unknown tag {tag!r}")
    profiles = [
        row for row in await asyncio.to_thread(_profile_rows)
        if access.tags.excluded_reason(row.slug) is None
        and (wanted is None or wanted in access.tags.tags.get(row.slug, ()))
    ]
    states = await asyncio.to_thread(_live_states, profiles) if include_state and profiles else {}
    listed = []
    for row in profiles:
        entry = {
            "slug": row.slug,
            "name": row.name or row.slug,
            "bot_type": normalize_bot_type(row.bot_type, row.agent_backend),
            "description": row.description or "",
            "default_model": row.default_model or "",
            "agent_backend": row.agent_backend,
            "tags": sorted(access.tags.tags.get(row.slug, ())),
        }
        if include_state:
            entry["state"] = states[row.slug]
        listed.append(entry)
    return listed
