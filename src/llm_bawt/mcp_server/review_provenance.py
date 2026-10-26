"""Model provenance for a task review, resolved from the reviewer's signed turn (TASK-1249).

Nothing here comes from tool arguments or the review body. The reviewer's
model is the one its signed turn ran on (``turn_logs``). The requester is the
bot whose delivery for this task that turn provably received (an accepted idle
turn, or a steer whose current generation started; a mere claim does not
count), and its model is the model of the sending turn recorded on the
delivery (``sender_turn_id``).

The author's model needs a commit-to-turn record that does not exist yet
(TASK-1462), so it and ``crossProvider`` stay null with a reason note. A
``requesterModel`` is never used as the author's model.

Every unresolved field is null and carries a ``<field>_unresolved:<reason>``
note. Notes are sorted so a retry from the same turn produces identical
provenance (BawtHub rejects a keyed replay whose provenance differs).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from llm_bawt.bot_access import BotAccessError

logger = logging.getLogger(__name__)

SIGNED_TURN = "signed_turn"
UNSIGNED = "unsigned"
AUTHOR_UNRESOLVED = "author_unresolved:no_commit_turn_record"

# Candidate requesting deliveries: claimed for this turn, for this task, from
# another bot. A claim alone proves nothing (target_turn_id is written before
# transport), so _received() keeps only rows this turn provably got.
_REQUESTER_SQL = """
    SELECT sender_bot_id, sender_turn_id, status, delivery_mode,
           transport_accepted_at IS NOT NULL AS accepted, metadata_json
      FROM inter_bot_deliveries
     WHERE target_turn_id = :turn
       AND target_bot_id = :bot
       AND sender_bot_id <> :bot
       AND lower(task_id) = ANY(:tasks)
       AND status IN ('STEERING', 'DISPATCHING', 'DELIVERED')
       AND delivery_mode IN ('steer', 'turn')
"""


def _received(row: Any) -> bool:
    """Durable receipt: an accepted idle turn, or a steer whose generation started."""
    from llm_bawt.inter_bot_delivery_steer_fence import steer_started

    if row["delivery_mode"] == "turn":
        return bool(row["accepted"]) and row["status"] in {"DISPATCHING", "DELIVERED"}
    return steer_started(row)


def signed_review_turn(bot_id: str, arguments: dict) -> Any | None:
    """The reviewer's signed turn, or None for an unsigned caller.

    A supplied but bad credential, or one signed for another bot than the
    ``bot_id`` argument, raises ``caller_unavailable``; it never falls back to
    unsigned.
    """
    from .bot_access_identity import trusted_caller
    from .task_association import current_task_turn_context

    caller = trusted_caller("tasks_reviews_add", arguments)
    if caller is None:
        return None
    if caller.bot_id != bot_id:
        raise BotAccessError("caller_unavailable")
    if caller.kind != "turn":
        return None
    try:
        turn = current_task_turn_context()
    except Exception:
        raise BotAccessError("caller_unavailable") from None
    if (turn.bot_id, turn.user_id) != (caller.bot_id, caller.user_id):
        raise BotAccessError("caller_unavailable")
    return turn


def _engine():
    from llm_bawt.utils.config import Config
    from llm_bawt.utils.db import get_shared_engine

    return get_shared_engine(Config())


def turn_model(turn_id: str, bot_id: str) -> str | None:
    """The endpoint ref a turn ran on, from its own bot's turn log."""
    from sqlalchemy import text

    with _engine().connect() as conn:
        return conn.execute(text(
            "SELECT model FROM turn_logs WHERE id = :id AND bot_id = :bot"
        ), {"id": turn_id, "bot": bot_id}).scalar()


def requesting_deliveries(turn_id: str, bot_id: str, task_refs: list[str]) -> list[tuple[str, str | None]]:
    """Distinct (sender, sender turn) of task deliveries this turn provably received."""
    from sqlalchemy import text

    with _engine().connect() as conn:
        rows = conn.execute(text(_REQUESTER_SQL), {
            "turn": turn_id, "bot": bot_id, "tasks": [ref.lower() for ref in task_refs],
        }).mappings().all()
    return sorted({(row["sender_bot_id"], row["sender_turn_id"]) for row in rows if _received(row)},
                  key=lambda pair: (pair[0], pair[1] or ""))


def provider_of(ref: str | None) -> str | None:
    """Model vendor of an exact endpoint ref; never guessed from a bare model key."""
    if not ref:
        return None
    from llm_bawt.utils.config import Config

    endpoint = Config().ensure_model_catalog().endpoint_by_ref(ref)
    return endpoint.model.vendor if endpoint else None


def _guarded(note: str, notes: list[str], call: Callable[[], Any]) -> Any:
    try:
        return call()
    except Exception:
        logger.warning("review provenance: %s", note, exc_info=True)
        notes.append(note)
        return None


def _reviewer(turn: Any, notes: list[str]) -> dict:
    model = _guarded("reviewer_unresolved:turn_log_unavailable", notes,
                     lambda: turn_model(turn.turn_id, turn.bot_id))
    if model is None:
        if "reviewer_unresolved:turn_log_unavailable" not in notes:
            notes.append("reviewer_unresolved:no_turn_log")
        return {"reviewerModel": None, "reviewerProvider": None}
    provider = _guarded("reviewer_unresolved:catalog_unavailable", notes, lambda: provider_of(model))
    if provider is None and "reviewer_unresolved:catalog_unavailable" not in notes:
        notes.append("reviewer_unresolved:unknown_endpoint")
    return {"reviewerModel": model, "reviewerProvider": provider}


def _requester(turn: Any, task_refs: list[str], notes: list[str]) -> dict:
    empty = {"requesterBotId": None, "requesterModel": None}
    if not task_refs:
        notes.append("requester_unresolved:task_unavailable")
        return empty
    rows = _guarded("requester_unresolved:deliveries_unavailable", notes,
                    lambda: requesting_deliveries(turn.turn_id, turn.bot_id, task_refs))
    if rows is None:
        return empty
    senders = {sender for sender, _ in rows}
    if not senders:
        notes.append("requester_unresolved:no_delivery")
        return empty
    if len(senders) > 1:
        notes.append("requester_unresolved:multiple_senders")
        return empty
    sender_turns = {sender_turn for _, sender_turn in rows}
    if sender_turns == {None}:
        notes.append("requester_unresolved:no_sender_turn")
        return empty
    if len(sender_turns) > 1:  # includes an unsigned send beside a signed one
        notes.append("requester_unresolved:multiple_sender_turns")
        return empty
    sender = senders.pop()
    model = _guarded("requester_unresolved:turn_log_unavailable", notes,
                     lambda: turn_model(sender_turns.pop(), sender))
    if model is None and "requester_unresolved:turn_log_unavailable" not in notes:
        notes.append("requester_unresolved:no_turn_log")
    return {"requesterBotId": sender, "requesterModel": model}


def resolve_review_provenance(turn: Any | None, task_refs: list[str]) -> dict:
    """BawtHub review provenance fields (camelCase, wire-ready) for a caller.

    ``turn`` comes from :func:`signed_review_turn` (None = unsigned).
    ``task_refs`` are the reviewed task's cuid and shortId (empty when the task
    could not be read). Database or catalog failures only add notes.
    """
    if turn is None:
        return {"provenance": UNSIGNED}
    notes: list[str] = []
    fields = {
        "provenance": SIGNED_TURN,
        **_reviewer(turn, notes),
        **_requester(turn, task_refs, notes),
        "authorModel": None,
        "authorProvider": None,
        "crossProvider": None,
    }
    notes.append(AUTHOR_UNRESOLVED)
    return {**fields, "provenanceNotes": sorted(notes)}
