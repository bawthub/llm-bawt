"""First-bot consent and its one durable orientation turn (TASK-2524)."""
from __future__ import annotations

import json

from sqlalchemy import text

from ..bot_identities import SYSTEM_BOT_ID
from ..message_authorship import AuthorReference
from ..prompt_registry import get_prompt_resolver

PROMPT_KEY = "setup.first_chat"
PRODUCER = "setup"
SETTING_KEY = "setup.orientation"


def prepare_first_chat(config) -> str:
    resolved = get_prompt_resolver(config).resolve(PROMPT_KEY)
    if resolved is None:
        raise RuntimeError(f"Prompt '{PROMPT_KEY}' is not registered")
    return resolved.body


def record_choice(conn, profile, *, opted_in: bool, user_id: str | None, prompt: str | None, store):
    """Called inside the serialized bot-create transaction, after both readiness gates.

    The first choice survives even if that bot later disappears. The delivery
    row is the milestone receipt (queued/delivered/failed), not a second state
    machine. Neither a conflicting create nor a replay can submit another turn.
    """
    available = conn.execute(text("""
        SELECT NOT EXISTS (SELECT 1 FROM bot_profiles WHERE slug <> :slug)
           AND NOT EXISTS (SELECT 1 FROM runtime_settings
               WHERE scope_type='global' AND scope_id='*' AND key=:key)
    """), {"slug": profile.slug, "key": SETTING_KEY}).scalar_one()
    if not available:
        return None

    record = None
    if opted_in:
        if not user_id or not prompt or store is None:
            raise RuntimeError("First chat is unavailable; try again or turn off the introduction")
        access = conn.execute(text("""
            SELECT a.key FROM model_endpoints e JOIN access_paths a ON a.id=e.access_path_id
            WHERE e.id=:endpoint_id
        """), {"endpoint_id": profile.endpoint_id}).scalar_one()
        body = f"{prompt}\n\nChecked in setup: model {profile.default_model}, connection {access}."
        record, _ = store.enqueue(
            sender_bot_id=SYSTEM_BOT_ID,
            target_bot_id=profile.slug,
            author=AuthorReference.user(SYSTEM_BOT_ID),
            message=body,
            idempotency_key="setup:first-chat:v1",
            message_kind="SETUP_ORIENTATION",
            payload={
                "bot_id": profile.slug, "user": user_id,
                "messages": [{"role": "user", "content": body}],
                "prefer_steer": False, "reply_mode": "async", "stream": True,
                "augment_memory": False, "extract_memory": False,
            },
            metadata={"setup_owner_user_id": user_id},
            connection=conn, producer=PRODUCER, on_gate="hold",
        )

    conn.execute(text("""
        INSERT INTO runtime_settings (scope_type, scope_id, key, value_json, updated_at)
        VALUES ('global', '*', :key, :state, CURRENT_TIMESTAMP)
    """), {"key": SETTING_KEY, "state": json.dumps({
        "bot_id": profile.slug, "opted_in": opted_in, "milestone": "first_chat",
        "delivery_id": record.id if record is not None else None,
    })})
    return record
