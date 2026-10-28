"""Approver-notice cards on an approver bot's history page (TASK-1661).

Only the durable delivery -> user-message join marks a row as a notice; message
text never does. Each card is re-checked with the one eligibility rule at read
time, so a revoked grant, a private requester or another user's card renders as
an unavailable stub carrying the request ID only. Read-only; failure omits the
cards, never the timeline.
"""
from sqlalchemy import bindparam, text

from ...bot_identities import SYSTEM_BOT_ID
from ..approval_fanout import REQUEST_ID_KEY, REQUEST_KIND, RESOLVED_KIND
from ..approver_eligibility import eligible_on
from ..logging import get_service_logger

log = get_service_logger(__name__)
BATCH_SIZE = 500

_NOTICES = text(f"""
    SELECT m.id AS message_id, d.message_kind AS kind,
           d.metadata_json->>'{REQUEST_ID_KEY}' AS request_id
    FROM inter_bot_deliveries d
    JOIN messages m ON m.id = d.user_message_id AND m.bot_id = d.target_bot_id
    JOIN sessions s ON s.id = m.session_id AND s.bot_id = m.bot_id
    WHERE m.bot_id = :bot_id AND m.id IN :message_ids AND m.role = 'user'
      AND d.sender_bot_id = :system AND d.message_kind IN (:request, :resolved)
      AND s.user_id = :owner AND d.payload_json->>'user' = :owner
""").bindparams(bindparam("message_ids", expanding=True))


def hydrate_approver_notices_for_page(service, bot_id: str, page_messages: list[dict],
                                      user_id: str | None = None) -> dict[str, dict]:
    """``{message_id: ApproverNotice-shaped dict}`` for this page's notice rows."""
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
        with engine.connect() as conn:
            found = []
            for start in range(0, len(message_ids), BATCH_SIZE):
                found += conn.execute(_NOTICES, {
                    "bot_id": bot_id, "owner": owner, "system": SYSTEM_BOT_ID,
                    "request": REQUEST_KIND, "resolved": RESOLVED_KIND,
                    "message_ids": message_ids[start:start + BATCH_SIZE],
                }).mappings().all()
            return {str(row["message_id"]): _notice(conn, bot_id, owner, row)
                    for row in found if row["request_id"]}
    except Exception as exc:
        log.warning("Failed to enrich approver notices for history page: %s", exc)
        return {}


def _notice(conn, approver: str, owner: str, found) -> dict:
    from .approval_policies import _store
    from .ops_approvals import _card, approver_detail

    request_id = str(found["request_id"])
    row = _store().get_request(request_id)
    card = _card(row) if row is not None and eligible_on(
        conn, approver=approver, row=row, user_id=owner) else None
    detail = approver_detail(row, card, approver=approver).model_dump(mode="json") if card else None
    return {"request_id": request_id, "kind": found["kind"],
            "available": detail is not None, "detail": detail}
