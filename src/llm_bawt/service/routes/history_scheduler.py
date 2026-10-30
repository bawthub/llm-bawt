"""Read-only, display-only scheduling origin enrichment (TASK-169).

No message text, message metadata or delivery metadata can confer scheduling
provenance. Only the durable occurrence -> delivery -> user-message join does.
This deliberately does not alter generic history filtering or model context.
"""
from datetime import datetime, timezone

from sqlalchemy import bindparam, text

from ..logging import get_service_logger

log = get_service_logger(__name__)
BATCH_SIZE = 500

_ORIGINS = text("""
    SELECT m.id AS message_id, o.job_id AS schedule_id,
           o.run_id AS occurrence_id, o.scheduled_for
    FROM prompt_occurrences o
    JOIN prompt_schedules p ON p.job_id = o.job_id
    JOIN inter_bot_deliveries d ON d.id = o.delivery_id
    JOIN messages m ON m.id = d.user_message_id AND m.bot_id = d.target_bot_id
    JOIN sessions s ON s.id = m.session_id AND s.bot_id = m.bot_id
    WHERE m.bot_id = :bot_id AND m.id IN :message_ids AND m.role = 'user'
      AND p.owner_user_id = :owner AND s.user_id = :owner
      AND ((d.author_entity_type = 'user' AND d.author_entity_id = :owner)
           OR (d.author_entity_type = 'bot' AND d.author_entity_id = p.creator_bot_id))
      AND o.session_id = m.session_id
""").bindparams(bindparam("message_ids", expanding=True))


def hydrate_scheduler_for_page(service, bot_id: str, page_messages: list[dict],
                               user_id: str | None = None) -> dict[str, dict]:
    """Bounded batched trusted origins for the requested owner's user rows.

    Tables may not exist yet during a rolling upgrade. Failure omits badges,
    never the timeline. No schema creation, store bootstrap or writes on reads.
    """
    owner = user_id if isinstance(user_id, str) else getattr(getattr(service, "config", None), "DEFAULT_USER", None)
    if not isinstance(owner, str) or not owner.strip():
        return {}
    owner = owner.strip()
    message_ids = list(dict.fromkeys(
        str(m["id"]) for m in page_messages if m.get("id") and m.get("role") == "user"
    ))
    if not message_ids:
        return {}
    try:
        from ...media.assets import _build_engine

        engine = _build_engine(service.config)
        if engine is None:
            return {}
        origins = {}
        with engine.connect() as conn:
            for start in range(0, len(message_ids), BATCH_SIZE):
                rows = conn.execute(_ORIGINS, {
                    "bot_id": bot_id, "owner": owner,
                    "message_ids": message_ids[start:start + BATCH_SIZE],
                }).mappings().all()
                for row in rows:
                    scheduled = row["scheduled_for"]
                    if isinstance(scheduled, str):
                        scheduled = datetime.fromisoformat(scheduled.replace("Z", "+00:00"))
                    if scheduled.tzinfo is None:
                        scheduled = scheduled.replace(tzinfo=timezone.utc)
                    origins[str(row["message_id"])] = {
                        "schedule_id": row["schedule_id"],
                        "occurrence_id": row["occurrence_id"],
                        "scheduled_for": scheduled.isoformat(),
                    }
        return origins
    except Exception as exc:
        log.warning("Failed to enrich scheduling origins for history page: %s", exc)
        return {}
