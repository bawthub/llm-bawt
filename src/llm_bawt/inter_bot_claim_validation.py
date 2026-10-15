"""One identity check for durable-delivery entry and turn consumption.

Entry accepts a claim in STEERING or DISPATCHING; only a DISPATCHING
fallback may consume a reserved turn. The latter runs in save_turn's
transaction and locks the claim row so a lease takeover cannot interleave.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text as sa_text


def validate_delivery_claim(
    conn: Any,
    *,
    delivery_id: str,
    claim_token: str,
    target_bot_id: str,
    turn_id: str,
    user_message_id: str,
    bridge_request_id: str,
    for_turn: bool = False,
) -> bool:
    """Check exact identity, with a row lock when authorizing turn creation."""
    if not all((delivery_id, claim_token, target_bot_id, turn_id, user_message_id, bridge_request_id)):
        return False
    states = "status='DISPATCHING'" if for_turn else "status IN ('STEERING','DISPATCHING')"
    lock = " FOR UPDATE" if for_turn and conn.dialect.name == "postgresql" else ""
    row = conn.execute(sa_text("""
        SELECT payload_json FROM inter_bot_deliveries
        WHERE id=:id AND """ + states + """ AND claim_token=:claim_token
          AND target_bot_id=:target AND turn_id=:turn_id
          AND user_message_id=:message_id
    """ + lock), {
        "id": delivery_id,
        "claim_token": claim_token,
        "target": target_bot_id,
        "turn_id": turn_id,
        "message_id": user_message_id,
    }).scalar_one_or_none()
    if isinstance(row, str):  # SQLite tests; PostgreSQL JSONB returns a dict.
        row = json.loads(row)
    return isinstance(row, dict) and row.get("inter_bot_bridge_request_id") == bridge_request_id
