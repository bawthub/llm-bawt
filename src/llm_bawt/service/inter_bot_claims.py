"""Shared validation for dispatcher-owned inter-bot correlation fields."""

from __future__ import annotations

from typing import Any


_CORRELATION_FIELDS = (
    "inter_bot_turn_id",
    "inter_bot_bridge_request_id",
    "inter_bot_claim_token",
    "inter_bot_timeout_seconds",
    "inter_bot_session_policy",
    "inter_bot_seed_session_id",
)


def validate_inter_bot_claim(service: Any, request: Any):
    """Reject spoofed durable authority and return its trusted bot author."""
    from ..message_authorship import AuthorReference

    approval_claim = getattr(request, "_internal_approval_claim", None)
    if approval_claim is not None:
        from .approval_continuations import validate_approval_continuation_claim

        validate_approval_continuation_claim(service, request, approval_claim)
        return None

    delivery_id = getattr(request, "inter_bot_delivery_id", None)
    if delivery_id:
        dispatcher = getattr(service, "_inter_bot_dispatcher", None)
        if dispatcher is None or not dispatcher.store.validate_claim(
            delivery_id=delivery_id,
            claim_token=getattr(request, "inter_bot_claim_token", None) or "",
            target_bot_id=(getattr(request, "bot_id", None) or service._default_bot),
            turn_id=getattr(request, "inter_bot_turn_id", None) or "",
            user_message_id=getattr(request, "user_message_id", None) or "",
            bridge_request_id=(
                getattr(request, "inter_bot_bridge_request_id", None) or ""
            ),
        ):
            raise ValueError("invalid or stale inter-bot delivery claim")
        record = dispatcher.store.get(delivery_id)
        if record is None:
            raise ValueError("inter-bot delivery claim disappeared")
        return record.author

    if any(getattr(request, field, None) for field in _CORRELATION_FIELDS):
        raise ValueError(
            "inter-bot correlation fields require a valid delivery claim"
        )

    internal_sender = getattr(request, "_internal_inter_bot_sender_id", None)
    if isinstance(internal_sender, str) and internal_sender.strip():
        return AuthorReference.bot(internal_sender)
    return None


def continuation_author(request: Any):
    """The first-party system author of an app-written continuation turn (TASK-1807).

    Keyed only on the Python-only approval claim, never on a request body field
    or on message text. The kind is the claimed row's: the claim validator
    (``validate_inter_bot_claim``, run before any writer) binds payload presence
    to ``row.request_kind``.
    """
    if getattr(request, "_internal_approval_claim", None) is None:
        return None
    from ..message_authorship import AuthorReference

    kind = "mcp" if getattr(request, "continuation_payload", None) is not None else "harness"
    return AuthorReference.system(f"approval-continuation:{kind}")


def turn_admission_claim(service: Any, request: Any, *, store: Any = None):
    """The trusted producer claim checked inside the turn's admission transaction.

    Approval claims retain the owner stamped on their reservation. Delivery
    claims are validated again under the turn's target lock and a row lock, so
    lease takeover cannot happen between entry validation and turn creation.
    """
    from .turn_admission import AdmissionClaim

    approval_claim = getattr(request, "_internal_approval_claim", None)
    if approval_claim is not None:
        if store is None:
            store = getattr(service, "_tool_approval_policy_store", None)
        fence_for = getattr(store, "continuation_claim_fence", None)
        if fence_for is None:
            raise RuntimeError("Approval store unavailable; refusing unfenced continuation turn")
        request_id, token = approval_claim
        return AdmissionClaim(owner=token, fence=fence_for(request_id, token))

    delivery_id = getattr(request, "inter_bot_delivery_id", None)
    if not delivery_id:
        return None
    dispatcher = getattr(service, "_inter_bot_dispatcher", None)
    delivery_store = getattr(dispatcher, "store", None)
    if delivery_store is None or getattr(delivery_store, "engine", None) is None:
        raise RuntimeError("Delivery store unavailable; refusing unfenced delivery turn")
    token = getattr(request, "inter_bot_claim_token", None) or ""
    target = getattr(request, "bot_id", None) or service._default_bot
    turn_id = getattr(request, "inter_bot_turn_id", None) or ""
    message_id = getattr(request, "user_message_id", None) or ""
    bridge_id = getattr(request, "inter_bot_bridge_request_id", None) or ""
    if not all((token, target, turn_id, message_id, bridge_id)):
        raise RuntimeError("Incomplete delivery claim; refusing unfenced delivery turn")

    from ..inter_bot_claim_validation import validate_delivery_claim

    def fence(conn):
        return validate_delivery_claim(
            conn, delivery_id=delivery_id, claim_token=token,
            target_bot_id=target, turn_id=turn_id,
            user_message_id=message_id, bridge_request_id=bridge_id,
            for_turn=True,
        )

    return AdmissionClaim(
        owner=token, fence=fence,
        allow_unowned_reservation=True, require_reservation=True,
        reservation_delivery_id=delivery_id,
    )
