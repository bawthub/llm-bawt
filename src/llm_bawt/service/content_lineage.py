"""Known, server-verified delivery/continuation ancestry, never transcript roles."""
from __future__ import annotations

import logging

from sqlalchemy import JSON, text

from ..content_provenance import ContentTurn
from ..inter_bot_failure import FailureOrigin
from .dependencies import get_content_provenance_store

log = logging.getLogger(__name__)
APPROVAL_ORIGIN_KEY = "_content_approval_id"


def approval_source(row) -> ContentTurn | None:
    if row is None:
        return None
    values = [getattr(row, name, None) for name in (
        "bot_id", "user_id", "turn_id", "trigger_message_id",
    )]
    if any(not isinstance(value, str) or not value.strip() or len(value) > 128
           or value in {"unknown", "system", "none"} for value in values):
        return None
    return ContentTurn(*values)


def _inherit(service, sources, target):
    if not sources:
        # Legacy unsigned deliveries have no known clean/tainted origin. Do
        # not fabricate one from sender names or role=user.
        log.info("No verified content lineage available for automated turn")
        return
    store = getattr(service, "_content_provenance_store", None)
    if store is None:
        store = get_content_provenance_store(service.config)
    for source in set(sources):
        store.inherit(source, target)


def inherit_approval(service, row, target: ContentTurn):
    source = approval_source(row)
    _inherit(service, [source] if source else [], target)


def delivery_sources(service, delivery_id: str, target: ContentTurn) -> list[ContentTurn]:
    """Read the claimed head and its actual batch followers, not metadata IDs."""
    delivery_store = service._inter_bot_dispatcher.store
    with delivery_store.engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT * FROM inter_bot_deliveries
            WHERE id=:id OR batch_head_id=:id
        """).columns(metadata_json=JSON), {"id": delivery_id}).mappings().all()
    head = next((row for row in rows if row["id"] == delivery_id), None)
    if head is None or head["target_bot_id"] != target.bot_id or target.turn_id not in {
        head["turn_id"], head.get("target_turn_id"),
    }:
        raise ValueError("content lineage delivery target mismatch")
    sources = []
    for row in rows:
        if row["target_bot_id"] != target.bot_id:
            raise ValueError("content lineage batch target mismatch")
        origin = FailureOrigin.from_row(row)
        if origin is not None:
            sources.append(ContentTurn(
                origin.bot_id, origin.user_id, origin.turn_id, origin.trigger_message_id,
            ))
            continue
        # Busy MCP-result continuations are system-authored durable deliveries.
        # The protected ID leads back to the authoritative approval row.
        metadata = row.get("metadata_json") or {}
        approval_id = metadata.get(APPROVAL_ORIGIN_KEY)
        if approval_id is not None:
            from ..bot_identities import SYSTEM_BOT_ID

            if (row["sender_bot_id"] != SYSTEM_BOT_ID
                    or row["author_entity_type"] != "bot"
                    or row["author_entity_id"] != SYSTEM_BOT_ID
                    or metadata.get("approval_continuation_id") != approval_id):
                raise ValueError("content approval lineage mismatch")
            approval = service._tool_approval_policy_store.get_request(approval_id)
            if approval is None or approval.bot_id != target.bot_id or approval.user_id != target.user_id:
                raise ValueError("content approval lineage owner mismatch")
            source = approval_source(approval)
            if source is not None:
                sources.append(source)
    return sources


def inherit_delivery(service, delivery_id: str, target: ContentTurn):
    _inherit(service, delivery_sources(service, delivery_id, target), target)


def inherit_admitted_request(service, request, target: ContentTurn):
    """Call only after existing producer claim validation and turn admission."""
    claim = getattr(request, "_internal_approval_claim", None)
    if claim is not None:
        row = service._tool_approval_policy_store.get_request(claim[0])
        if row is None or row.bot_id != target.bot_id or row.user_id != target.user_id:
            raise ValueError("content continuation owner mismatch")
        inherit_approval(service, row, target)
    elif getattr(request, "inter_bot_delivery_id", None):
        inherit_delivery(service, request.inter_bot_delivery_id, target)
