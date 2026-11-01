"""Dormant gate framing on the existing durable ops receipt outbox.

No public admission path attaches this callback yet. A future trusted caller
must derive the destination from signed turn context, never public arguments,
and choose this direct callback OR the approval continuation, not both.
"""
from __future__ import annotations

import json
import logging

from .gate_plan import GatePlan

logger = logging.getLogger(__name__)
GATE_RECEIPT_KEY = "ops.batch_gate_receipt"
# Registry seed, not an instruction embedded in an executor or catalog row.
DEFAULT_GATE_RECEIPT = (
    "[Batch gate terminal wake: the JSON below identifies the ops job and exact "
    "repository/commit workload that ended. This callback is a wake, not SHIP "
    "authority or a verified gate receipt. Read the reconciled job and its "
    "trusted exact-SHA host gate proof before any publication decision. Do not "
    "rerun or push automatically. Lost/timed-out state does not prove the runner "
    "stopped; a retry needs positive retirement evidence and admission fencing.]"
)
_CALLBACK_FIELDS = {"kind", "bot_id", "user_id", "session_id", "plan"}


def gate_receipt_prompt(bot_id: str) -> str:
    """Resolve registered framing at delivery time, retaining the seed on failure."""
    try:
        from ..prompt_registry import get_prompt_resolver

        resolved = get_prompt_resolver().resolve(GATE_RECEIPT_KEY, scope_type="bot", scope_id=bot_id)
        if resolved and (resolved.body or "").strip():
            return resolved.body
    except Exception as error:
        logger.warning("ops prompt %s resolve failed, using seed: %s", GATE_RECEIPT_KEY, error)
    return DEFAULT_GATE_RECEIPT


def gate_callback(plan: GatePlan, *, bot_id: str, user_id: str, session_id: str) -> dict:
    """Internal serialization seam only; it confers no caller authority."""
    plan = GatePlan.from_dict(plan.to_dict())
    for value in (bot_id, user_id, session_id):
        if type(value) is not str or not value.strip() or len(value) > 256:
            raise ValueError("gate callback needs a bounded nonempty caller identity and session")
    return {"kind": "batch_gate", "bot_id": bot_id, "user_id": user_id,
            "session_id": session_id, "plan": plan.to_dict()}


def frame_gate_receipt(job, callback: dict, receipt: dict) -> tuple[str, dict]:
    """Check stored identity before enqueue; never infer runner success or retirement."""
    if callback.keys() != _CALLBACK_FIELDS or callback.get("kind") != "batch_gate":
        raise ValueError("invalid batch gate callback")
    plan = GatePlan.from_dict(callback["plan"])
    gate_callback(plan, bot_id=callback["bot_id"], user_id=callback["user_id"], session_id=callback["session_id"])
    if job.approval_request_id:
        raise ValueError("approved gate jobs use their approval continuation, not a direct callback")
    if (callback["bot_id"], callback["user_id"]) != (job.caller_bot_id, job.caller_user_id):
        raise ValueError("gate callback destination does not match the job caller")
    if GatePlan.from_dict(json.loads(job.args_json)) != plan:
        raise ValueError("gate callback plan does not match the job workload")
    return gate_receipt_prompt(callback["bot_id"]), {**receipt, "gate_plan": plan.to_dict()}
