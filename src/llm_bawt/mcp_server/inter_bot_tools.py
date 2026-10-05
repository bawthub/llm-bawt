"""MCP tools for immediate and durable inter-bot communication."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from .inter_bot_send_mode import SendModeError, resolve_send_mode
from .server import _get_storage, mcp

logger = logging.getLogger(__name__)
_APP_BASE_URL = "http://localhost:8642"


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
) -> dict:
    body = {
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
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{_APP_BASE_URL}/v1/inter-bot-deliveries",
                json=body,
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
            "delivery": "steer_or_idle" if prefer_steer else "when_idle",
            "bot_id": target_bot_id,
            "sender": sender_bot_id,
            "error": str(detail),
            "status_code": exc.response.status_code,
        }
    except Exception as exc:
        return {
            "success": False,
            "queued": False,
            "delivery": "steer_or_idle" if prefer_steer else "when_idle",
            "bot_id": target_bot_id,
            "sender": sender_bot_id,
            "error": str(exc) or exc.__class__.__name__,
        }


@mcp.tool(name="bots_send_message")
async def send_message_to_bot(
    target_bot_id: str,
    message: str,
    sender_bot_id: str = "unknown",
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
    * ``wait_for_reply=True`` (or ``fire_and_forget=False``) — when_idle delivery
      plus a bounded wait for the reply. A busy target is rejected without side
      effects unless the key already owns a delivery. Cannot be combined with
      ``delivery="steer_or_idle"``; task handoffs (``task_id``) require a key.

    ``force=True`` is accepted for compatibility but never permits concurrency.
    Results carry ``mode`` and ``dispatched`` so a caller can tell whether the
    target received anything.
    """
    try:
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

    durable_kwargs = dict(
        target_bot_id=target_bot_id,
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
    enqueue = _compat_hook("_enqueue_durable", _enqueue_durable)
    if not mode.wait:
        receipt = await enqueue(prefer_steer=mode.prefer_steer, **durable_kwargs)
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

    existing = None
    key = (idempotency_key or "").strip()
    if key:
        existing = await _compat_hook("_find_delivery_by_key", _find_delivery_by_key)(
            sender_bot_id.strip().lower() or "unknown",
            target_bot_id.strip().lower(),
            key,
        )
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
    sender_bot_id: str | None = None,
    target_bot_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> dict:
    """List durable inter-bot deliveries and lifecycle/error state."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{_APP_BASE_URL}/v1/inter-bot-deliveries",
            params={
                key: value for key, value in {
                    "sender_bot_id": sender_bot_id,
                    "target_bot_id": target_bot_id,
                    "status": status,
                    "limit": min(max(limit, 1), 200),
                }.items() if value is not None
            },
            timeout=10.0,
        )
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
    sender_bot_id: str = "unknown",
) -> dict:
    """Queue one durable Claude /compact maintenance turn when the bot is idle."""
    key = (idempotency_key or "").strip()
    if not key:
        return {"success": False, "error": "idempotency_key is required"}
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


@mcp.tool(name="bots_list_available")
async def list_available_bots() -> list[dict]:
    """List bots available as inter-bot message targets."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{_APP_BASE_URL}/v1/bots", timeout=10.0)
            response.raise_for_status()
            data = response.json()
        bots = data.get("data", []) if isinstance(data, dict) else data
        return [
            {
                "slug": bot.get("slug", "unknown"),
                "name": bot.get("name", bot.get("slug", "Unknown")),
                "bot_type": bot.get("bot_type", "chat"),
                "description": bot.get("description", ""),
                "default_model": bot.get("default_model", ""),
                "agent_backend": bot.get("agent_backend"),
            }
            for bot in bots if isinstance(bot, dict)
        ]
    except Exception as exc:
        logger.error("Failed to list available bots: %s", exc)
        return []
