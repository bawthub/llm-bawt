"""Who may see and decide another bot's ops approval card (TASK-1609, TASK-1661).

One rule for every approver-facing path: the ``/v1/ops-approvals`` list, the
exact-ID read and resolve, the approver fan-out pump, the delivery claim fence
and the approver chat card. A bot is an eligible approver of a card when, right
now:

* it holds the ``approve_ops`` grant;
* the tag snapshot is available (fail closed when it isn't);
* the card is a first-party ``ops_run`` card of the same user;
* the requester is not a private bot;
* the requester is not the approver itself.

Masking (``display_args``) is redaction, not authorization: callers check this
first and only then render.
"""
from __future__ import annotations

from ..approval_models import KIND_MCP, OPS_RUN_TOOL
from ..bot_tags import BehaviorTag, grants
from ..bot_tags.facade import BotTags


def is_ops_card(row) -> bool:
    """A first-party ``ops_run`` approval card."""
    return row.request_kind == KIND_MCP and row.tool_name == OPS_RUN_TOOL


def card_visible(row, *, user_id: str, tags) -> bool:
    """The user's card from a known, non-private requester; fails closed."""
    return (row is not None and row.user_id == user_id and tags.available
            and not tags.has(row.bot_id, BehaviorTag.PRIVATE))


def approver_eligible(*, holds: bool, tags, row, approver: str, user_id: str) -> bool:
    """Whether ``approver`` may see and decide ``row`` for ``user_id`` now."""
    return (holds and card_visible(row, user_id=user_id, tags=tags)
            and is_ops_card(row) and row.bot_id != approver)


def holder_and_tags(conn, bot: str):
    """Whether ``bot`` holds ``approve_ops`` now, and the tag snapshot, on ``conn``."""
    return grants.holds_grant(conn, bot, BehaviorTag.APPROVE_OPS), BotTags().snapshot(conn)


def eligible_on(conn, *, approver: str, row, user_id: str) -> bool:
    """``approver_eligible`` evaluated live on ``conn`` (claim fence, pump, history)."""
    holds, tags = holder_and_tags(conn, approver)
    return approver_eligible(holds=holds, tags=tags, row=row, approver=approver, user_id=user_id)
