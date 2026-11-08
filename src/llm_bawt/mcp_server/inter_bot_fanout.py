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


# Per-target fields an agent acts on; everything else (metadata, the batch
# record repeated per target, failure origin, timestamps) stays in the
# durable row behind ``bots_delivery_get`` (TASK-1484).
_TARGET_FIELDS = ("target_bot_id", "delivery_id", "status")


def _compact_target(receipt: dict[str, Any], common_note: Any) -> dict[str, Any]:
    out = {k: receipt[k] for k in _TARGET_FIELDS if receipt.get(k) is not None}
    if "target_bot_id" in out:
        out = {"target": out.pop("target_bot_id"), **out}
    if receipt.get("duplicate"):
        out["duplicate"] = True
    if receipt.get("success") is False:
        out["success"] = False
        if receipt.get("last_error"):
            out["last_error"] = receipt["last_error"]
    if receipt.get("note") and receipt["note"] != common_note:
        out["note"] = receipt["note"]
    return out


def compact_batch_receipt(result: dict[str, Any]) -> dict[str, Any]:
    """One line per target plus the batch id; full rows via ``bots_delivery_get``."""
    deliveries = [d for d in result.get("deliveries") or [] if isinstance(d, dict)]
    note = next((d.get("note") for d in deliveries if d.get("success") and d.get("note")), None)
    out = dict(result)
    if "deliveries" in result:
        out["deliveries"] = [_compact_target(d, note) for d in deliveries]
    if note:
        out["note"] = note
    if deliveries:
        out["full_receipt"] = "bots_delivery_get(delivery_id)"
    return out


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
        return compact_batch_receipt({**data, "success": bool(data.get("dispatched")),
                                      "status_code": response.status_code})
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
