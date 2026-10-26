"""``bots_send_message`` with ``targets``: one call, many bots (TASK-1431 r3 §7).

A ``targets`` that names exactly one explicit bot (not the sender) takes the
existing single-target path and returns today's flat receipt. Anything else is
one ``POST /v1/inter-bot-deliveries/batch``; the route owns every fan-out rule
(selector, private/inactive exclusion, key requirement, rate limit, cap), so
this module only builds the body and reports the route's answer.
"""

from __future__ import annotations

from typing import Any

import httpx

from llm_bawt.inter_bot_send_targets import SelectorError, TargetSelector

from .inter_bot_send_mode import SendMode
from .task_association import with_sender_turn


def single_explicit_target(targets: Any, sender_bot_id: str) -> str | None:
    """The one bot a selector names explicitly, else None.

    The sender itself is never returned: a selector leaves the sender out
    unless ``include_self``, so ``targets=[<self>]`` must not quietly become a
    self-send through the single path. Raises :class:`SelectorError`.
    """
    selector = TargetSelector.parse(targets)
    if len(selector.bots) != 1:
        return None
    target = selector.bots[0]
    return None if target == sender_bot_id.strip().lower() else target


def selector_error(exc: SelectorError, sender_bot_id: str) -> dict[str, Any]:
    return {"success": False, "dispatched": False, "error": str(exc), "sender": sender_bot_id}


async def send_batch(
    base_url: str,
    *,
    targets: Any,
    mode: SendMode,
    exclude: list[str],
    fields: dict[str, Any],
) -> dict[str, Any]:
    """POST one fan-out batch; a refusal comes back as ``dispatched: false``.

    ``fields`` are the common send fields (sender, message, key, …), the same
    ones the single path sends.
    """
    body = {
        **fields,
        "targets": targets,
        "exclude": exclude,
        "mode": mode.delivery,
        "reply_mode": mode.label,
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/v1/inter-bot-deliveries/batch", json=body,
                headers=with_sender_turn(), timeout=60.0,
            )
            data = response.json()
    except Exception as exc:
        return {"success": False, "dispatched": False, "mode": mode.delivery,
                "error": str(exc) or exc.__class__.__name__}
    if isinstance(data, dict) and "batch_id" in data:
        return {**data, "success": bool(data.get("dispatched")),
                "status_code": response.status_code}
    detail = data.get("detail") if isinstance(data, dict) else data
    error = detail.get("error") if isinstance(detail, dict) else None
    return {
        "success": False,
        "dispatched": False,
        "mode": mode.delivery,
        "error": error or str(detail),
        **({"detail": detail.get("detail")} if isinstance(detail, dict) else {}),
        "status_code": response.status_code,
    }
