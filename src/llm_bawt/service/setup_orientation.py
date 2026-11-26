"""First-bot consent and its one durable orientation turn (TASK-2524)."""
from __future__ import annotations

import json
import logging

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from ..bot_identities import SYSTEM_BOT_ID
from ..memory.postgresql import _sanitize_table_name
from ..message_authorship import AuthorReference, SETUP_ORIENTATION_AUTHOR_ID
from ..prompt_registry import get_prompt_resolver
from .dependencies import get_runtime_settings_store

PROMPT_KEY = "setup.first_chat"
PRODUCER = "setup"
SETTING_KEY = "setup.orientation"
ORIENTATION_PROMPT_KEY = "setup.orientation"

log = logging.getLogger(__name__)


def orientation_prompt(config, *, bot_id: str, user_id: str, session_id: str) -> str | None:
    """Resolve guidance only for the conversation that owns the setup welcome.

    The actual persisted welcome trigger is the binding, not the bot's active
    session pointer. This also works for prior opt-ins without a backfill. Call
    after saving the trigger so the welcome itself can pin the guidance.
    """
    try:
        settings = get_runtime_settings_store(config)
        if settings.engine is None:
            return None
        choice = settings.get_value("global", "*", SETTING_KEY)
        if not isinstance(choice, dict) or choice.get("opted_in") is not True or choice.get("bot_id") != bot_id:
            return None
        delivery_id = choice.get("delivery_id")
        if not isinstance(delivery_id, str) or not delivery_id:
            return None
        with settings.engine.connect() as conn:
            matched = conn.execute(text("""
                SELECT 1 FROM inter_bot_deliveries d
                JOIN messages m ON m.bot_id=:partition AND m.id=d.user_message_id
                JOIN sessions s ON s.id=m.session_id AND s.bot_id=:bot AND s.user_id=:user
                WHERE d.id=:delivery AND d.target_bot_id=:bot
                  AND d.metadata_json->>'_request_identity_producer'=:producer
                  AND d.metadata_json->>'setup_owner_user_id'=:user
                  AND s.id=:session AND m.role='user'
                  AND m.author_entity_type='system' AND m.author_entity_id=:author
            """), {"delivery": delivery_id, "bot": bot_id, "user": user_id,
                   "partition": _sanitize_table_name(bot_id), "session": session_id,
                   "producer": PRODUCER, "author": SETUP_ORIENTATION_AUTHOR_ID}).first()
        if matched is None:
            return None
        resolved = get_prompt_resolver(config).resolve(
            ORIENTATION_PROMPT_KEY, scope_type="bot", scope_id=bot_id,
        )
        return resolved.body if resolved else None
    except SQLAlchemyError as exc:
        # Orientation is optional; a missing/unavailable store must not block chat.
        log.warning("Could not resolve setup orientation (%s)", type(exc).__name__)
        return None


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
                # augment_memory=False would run the turn in local mode, which
                # skips the profile, so the welcome could not greet the person
                # by their profile name (TASK-2643). extract_memory=False only
                # skips this turn's own fact extraction; history summaries may
                # still include the cue.
                "augment_memory": True, "extract_memory": False,
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
