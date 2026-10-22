"""Retry of a failed durable delivery as a new row (manual Retry, restart retry).

A retry is a *new* outbox row with new message/turn/bridge request IDs, not a
republication of the old Redis command. The original row and its failure remain
inspectable; one original delivery can create at most one retry, whichever
producer created it.

Two producers share one core and differ only by :class:`RetryPolicy`:

* **manual** (TASK-1111): the user presses Retry on a failed bot callback. The
  child replays the original message.
* **restart** (TASK-1231B): the target's bridge host shut down mid-turn or
  before the turn started (``abort_cause == HOST_SHUTDOWN`` on the turn row).
  The child carries the original request behind a note that it was
  interrupted, so a bot that had started checks its progress instead of
  repeating side effects, and a send that never started is not lost. Ordinary
  chat is eligible. The child is authored by the reserved ``system`` identity,
  never by the original author: words the user did not write are not
  attributed to them. The original author is kept in
  :data:`RETRY_ORIGINAL_AUTHOR`, so the reply-channel note still routes to a
  bot sender and an ordinary-chat child still reaches its user's view.
  Depth is one: a restart child killed again stays FAILED (visible, no loop).
"""

from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from sqlalchemy import text as sa_text
from sqlalchemy.engine import Connection

from agent_bridge.abort_contract import ABORT_CAUSE_KEY, HOST_SHUTDOWN

from .bot_identities import SYSTEM_BOT_ID
from .inter_bot_delivery import FAILED, InterBotDeliveryStore
from .inter_bot_delivery_record import DeliveryRecord
from .message_authorship import AuthorReference

logger = logging.getLogger(__name__)

RESTART_RETRY_PROMPT_KEY = "inter_bot.restart_retry.continue"
DEFAULT_RESTART_RETRY_CONTINUE = (
    "[Restart retry: the service hosting your turn on the request below "
    "restarted before the turn finished. If you had already started it, check "
    "what you completed and continue from there without repeating side "
    "effects. If you had not started it, handle it now.]"
)


class DeliveryRetryConflict(ValueError):
    """The delivery cannot safely be resent as a new turn."""


@dataclass(frozen=True)
class RetryPolicy:
    """What distinguishes one retry producer from another."""

    key_prefix: str
    marker: str
    # Metadata keys that make an original ineligible (hidden from this producer).
    refused_metadata: frozenset[str]
    # Builds the child's (message, payload) from the original's.
    build: Callable[[DeliveryRecord, dict[str, Any]], tuple[str, dict[str, Any]]]
    # Who the child's message is attributed to; None keeps the original author.
    author: AuthorReference | None = None

    def key(self, original_id: str) -> str:
        return f"{self.key_prefix}:{original_id}"


RETRY_MARKERS = ("manual_retry_of", "restart_retry_of")
# Stamped by RequestIdentityResolver over any caller metadata, so a sender
# cannot forge a retry child with a matching key and marker.
RETRY_PRODUCER = "retry"
# Who wrote the request a system-authored retry child carries.
RETRY_ORIGINAL_AUTHOR = "retry_original_author"
# Why a host-shutdown FAILED delivery did or did not come back (audit only).
RESTART_RETRY_OUTCOME = "restart_retry_outcome"
_EXECUTION_METADATA = frozenset({
    "maintenance_barrier", RETRY_ORIGINAL_AUTHOR, RESTART_RETRY_OUTCOME, *RETRY_MARKERS,
})


def is_system_retry(record: Any) -> bool:
    """A ``system``-authored row written by the retry producer (restart child).

    The producer key is server-stamped over caller metadata, so this cannot be
    forged by a sender.
    """
    author = getattr(record, "author", None)
    metadata = getattr(record, "metadata", None) or {}
    return (author is not None and author.entity_type == "bot"
            and author.entity_id == SYSTEM_BOT_ID
            and metadata.get("_request_identity_producer") == RETRY_PRODUCER)


def retry_original_author(record: Any) -> AuthorReference | None:
    """The author of the request a system-authored retry child carries.

    Trusted only when :func:`is_system_retry`; anywhere else the key is
    caller data and is ignored.
    """
    if not is_system_retry(record):
        return None
    value = (record.metadata or {}).get(RETRY_ORIGINAL_AUTHOR)
    if not isinstance(value, Mapping):
        return None
    entity_type, entity_id = value.get("entity_type"), value.get("entity_id")
    if not isinstance(entity_id, str) or not entity_id:
        return None
    if entity_type == "user":
        return AuthorReference.user(entity_id)
    if entity_type == "bot" and entity_id not in {SYSTEM_BOT_ID, "unknown"}:
        return AuthorReference.bot(entity_id)
    return None


def retry_reply_to(record: Any) -> str | None:
    """The bot a system-authored retry child replies to, or None."""
    author = retry_original_author(record)
    return author.entity_id if author is not None and author.entity_type == "bot" else None


def _original_author_of(original: DeliveryRecord) -> AuthorReference | None:
    # A retry of a retry keeps the first author.
    inherited = retry_original_author(original)
    if inherited is not None:
        return inherited
    author = original.author
    if author.entity_type in {"user", "bot"} and author.entity_id not in {SYSTEM_BOT_ID, "unknown", None}:
        return author
    return None


def _replay(original: DeliveryRecord, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return original.message, payload


def resolve_restart_retry_text(target_bot_id: str | None) -> str:
    """The live "you were interrupted, continue" text, falling back to the seed.

    A registry or DB failure must never strand an interrupted delivery, so any
    error or blank body uses :data:`DEFAULT_RESTART_RETRY_CONTINUE`.
    """
    try:
        from .prompt_registry import get_prompt_resolver

        resolved = get_prompt_resolver().resolve(
            RESTART_RETRY_PROMPT_KEY,
            scope_type="bot" if target_bot_id else "global", scope_id=target_bot_id,
        )
        if resolved and (resolved.body or "").strip():
            return resolved.body
    except Exception as error:  # registry/DB failure must not block the retry
        logger.warning("restart-retry prompt resolve failed, using seed: %s", error)
    return DEFAULT_RESTART_RETRY_CONTINUE


def _continue(original: DeliveryRecord, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Prefix the original request with the interruption note.

    The request itself is always kept, parts and attachments included: the
    host may have shut down before the turn started, and then the original
    request is the whole job.
    """
    note = resolve_restart_retry_text(original.target_bot_id)
    message = f"{note}\n\n{original.message}"
    messages = list(payload.get("messages") or [])
    last_user = next((i for i in range(len(messages) - 1, -1, -1)
                      if isinstance(messages[i], dict) and messages[i].get("role") == "user"), None)
    if last_user is None:
        messages.append({"role": "user", "content": message})
    else:
        content = messages[last_user].get("content")
        if isinstance(content, list):
            content = [{"type": "text", "text": note}, *content]
        elif isinstance(content, str):
            content = f"{note}\n\n{content}"
        else:
            content = message
        messages[last_user] = {**messages[last_user], "content": content}
    return message, {**payload, "messages": messages}


MANUAL_RETRY = RetryPolicy(
    key_prefix="manual-retry", marker="manual_retry_of",
    refused_metadata=frozenset({"prompt_schedule", "ordinary_chat", "maintenance_user_queue"}),
    build=_replay,
)

RESTART_RETRY = RetryPolicy(
    key_prefix="restart-retry", marker="restart_retry_of",
    # Ordinary chat is eligible. A /new (session intent) must not run twice, and
    # a restart child is not restart-retried again (no chains).
    refused_metadata=frozenset({
        "prompt_schedule", "maintenance_user_queue",
        "new_session_intent", "new_session_execution", "restart_retry_of",
    }),
    build=_continue,
    author=AuthorReference.bot(SYSTEM_BOT_ID),
)


def restart_abort_cause(turn: Any) -> str | None:
    """Return ``HOST_SHUTDOWN`` when ``turn`` was killed by its host shutting down.

    Reads the per-attempt ``request_json.abort_cause`` audit stamped by the
    stream consumer. Only an ended ``aborted`` turn qualifies, and any explicit
    ``abort_request`` wins: a Stop the user asked for is never retried.
    """
    if turn is None or getattr(turn, "status", None) != "aborted" or getattr(turn, "ended_at", None) is None:
        return None
    raw = getattr(turn, "request_json", None)
    try:
        request = json.loads(raw) if isinstance(raw, str) and raw else raw
    except (TypeError, ValueError):
        return None
    if not isinstance(request, Mapping) or request.get("abort_request"):
        return None
    audit = request.get(ABORT_CAUSE_KEY)
    cause = audit.get("cause") if isinstance(audit, Mapping) else None
    return HOST_SHUTDOWN if cause == HOST_SHUTDOWN else None


def _turn_row(conn, turn_id: str | None) -> Any:
    if not turn_id:
        return None
    return conn.execute(sa_text(
        "SELECT status, ended_at, request_json FROM turn_logs WHERE id=:id"
    ), {"id": turn_id}).mappings().first()


def _existing_child(conn, original: DeliveryRecord, policy: RetryPolicy) -> str | None:
    """A retry child of ``original`` made by a *different* producer, if any."""
    others = [marker for marker in RETRY_MARKERS if marker != policy.marker]
    clauses = " OR ".join(f"metadata_json->>'{marker}' = :id" for marker in others)
    return conn.execute(sa_text(f"""
        SELECT id FROM inter_bot_deliveries
        WHERE target_bot_id=:target AND ({clauses})
          AND metadata_json->>'_request_identity_producer' = :producer
        LIMIT 1
    """), {"target": original.target_bot_id, "id": original.id,
          "producer": RETRY_PRODUCER}).scalar_one_or_none()


def _retry(
    store: InterBotDeliveryStore, delivery_id: str, policy: RetryPolicy,
    *, eligible: Callable[[Any, DeliveryRecord, dict[str, Any]], bool] | None = None,
    connection: Connection | None = None,
) -> tuple[DeliveryRecord, bool] | None:
    """Atomically copy one failed delivery into a new durable delivery.

    Acquire the target lock before the original row lock (the cancel path uses
    that order too). The deterministic key covers ambiguous outcomes and makes
    repeated calls return the same child. ``connection`` joins the caller's
    transaction, so a claim can be failed and retried in one commit. Every
    refusal happens before the first write, so a caller may catch
    :class:`DeliveryRetryConflict` and still commit its own writes.
    """
    if store.engine is None:
        raise RuntimeError("Inter-bot delivery database unavailable")
    if connection is not None and connection.engine is not store.engine:
        raise ValueError("delivery retry requires the shared admission engine")
    with (nullcontext(connection) if connection is not None else store.engine.begin()) as conn:
        target = conn.execute(sa_text(
            "SELECT target_bot_id FROM inter_bot_deliveries WHERE id=:id"
        ), {"id": delivery_id}).scalar_one_or_none()
        if target is None:
            return None
        from .service.turn_admission import lock_target

        lock_target(conn, target)
        row = conn.execute(sa_text(
            "SELECT * FROM inter_bot_deliveries WHERE id=:id FOR UPDATE"
        ), {"id": delivery_id}).mappings().first()
        if row is None or row["target_bot_id"] != target:
            return None
        original = DeliveryRecord.from_mapping(row)
        metadata = original.metadata
        if (any(key in metadata for key in policy.refused_metadata)
                or original.message_kind == "CONTEXT_MAINTENANCE"):
            return None
        if original.status != FAILED:
            raise DeliveryRetryConflict("Only failed deliveries can be retried")
        # A second reset can archive an entirely different session. The user
        # must submit an explicit new delivery if the old one requested a reset.
        if original.session_policy != "continue":
            raise DeliveryRetryConflict("Session-reset deliveries cannot be retried here")
        payload = row["payload_json"]
        if not isinstance(payload, dict):
            raise DeliveryRetryConflict("Original delivery payload is unavailable")
        if eligible is not None and not eligible(conn, original, payload):
            return None
        sibling = _existing_child(conn, original, policy)
        if sibling is not None:
            raise DeliveryRetryConflict(f"Delivery was already retried as {sibling}")
        # The old turn's compaction barrier is execution state, not a caller
        # instruction. enqueue stamps the new delivery's current barrier.
        retry_metadata = {key: value for key, value in metadata.items()
                          if key not in _EXECUTION_METADATA}
        if isinstance(retry_metadata.get("ordinary_chat"), dict):
            # The digest/assistant flags describe the original request only.
            retry_metadata["ordinary_chat"] = {
                key: value for key, value in retry_metadata["ordinary_chat"].items()
                if key not in {"digest", "assistant_supplied"}
            }
        retry_metadata[policy.marker] = original.id
        original_author = _original_author_of(original)
        if original_author is not None:
            retry_metadata[RETRY_ORIGINAL_AUTHOR] = original_author.to_dict()
        message, retry_payload = policy.build(original, dict(payload))
        return store.enqueue(
            sender_bot_id=original.sender_bot_id,
            target_bot_id=original.target_bot_id,
            message=message,
            payload=retry_payload,
            author=policy.author or original.author,
            idempotency_key=policy.key(original.id),
            project_id=original.project_id,
            task_id=original.task_id,
            message_kind=original.message_kind,
            metadata=retry_metadata,
            max_attempts=original.max_attempts,
            session_policy=original.session_policy,
            connection=conn,
            producer=RETRY_PRODUCER,
        )


def retry_failed_delivery(
    store: InterBotDeliveryStore, delivery_id: str,
) -> tuple[DeliveryRecord, bool] | None:
    """Manual Retry of one failed bot callback (replays the original message).

    Returns None for missing/hidden delivery IDs and raises for unsafe states.
    """
    return _retry(store, delivery_id, MANUAL_RETRY)


def _killed_by_host_shutdown(conn, original: DeliveryRecord, payload: dict[str, Any]) -> bool:
    # A standalone reset or a /new follow-up must never run a second time.
    if payload.get("inter_bot_reset_only") or payload.get("inter_bot_typed_follow_up"):
        return False
    return restart_abort_cause(_RowView(_turn_row(conn, original.turn_id))) == HOST_SHUTDOWN


class _RowView:
    """Attribute view over a turn_logs mapping for :func:`restart_abort_cause`."""

    def __init__(self, row: Mapping[str, Any] | None):
        self._row = row or {}

    def __getattr__(self, name: str) -> Any:
        return self._row.get(name)


def restart_retry_delivery(
    store: InterBotDeliveryStore, delivery_id: str, *, connection: Connection | None = None,
) -> tuple[DeliveryRecord, bool] | None:
    """Resume one FAILED delivery whose turn its host killed while shutting down.

    Self-validating and idempotent (key ``restart-retry:<id>``), so both the
    dispatcher and a recovery sweep may call it. Returns None when the
    delivery is missing, hidden from this producer, or was not killed by a host
    shutdown; raises :class:`DeliveryRetryConflict` for unsafe states. Pass
    ``connection`` to commit it with the caller's ``fail_claim``.
    """
    return _retry(store, delivery_id, RESTART_RETRY, eligible=_killed_by_host_shutdown,
                  connection=connection)


def fail_claim_with_restart_retry(
    store: InterBotDeliveryStore, record: DeliveryRecord, claim_token: str, error: str,
) -> tuple[DeliveryRecord | None, DeliveryRecord | None]:
    """Fail a claimed delivery and, if its host shut down, insert its restart retry.

    The one owner of the order target lock -> FAILED -> child. Both writes
    commit together, so a child never exists without its FAILED original. The
    child insert runs in a SAVEPOINT: if it is refused or errors, the FAILED
    write still commits and the delivery stays visibly failed. For a turn its
    host shut down, and only then, the outcome (``retried``/``refused``/
    ``error``) is written to the FAILED row's :data:`RESTART_RETRY_OUTCOME`
    in the same transaction, so "why didn't this come back?" is answerable
    from the database. Any other failure (a user Stop, a plain abort) never
    gets an outcome: there was no restart to retry. Returns
    ``(failed, child)``; ``failed`` is None when the claim was lost.
    """
    if store.engine is None:
        raise RuntimeError("Inter-bot delivery database unavailable")
    from .service.turn_admission import lock_target

    with store.engine.begin() as conn:
        lock_target(conn, record.target_bot_id)
        failed = store.fail_claim(record.id, claim_token, error, connection=conn)
        if failed is None:
            return None, None
        child, outcome, shutdown = None, None, False
        try:
            # Every read and write that can fail stays inside the SAVEPOINT,
            # so the outer transaction (the FAILED write) is always usable.
            # Whether the host killed this turn is decided first, from the
            # turn row: a refusal from ``_retry`` (e.g. a session reset) says
            # nothing about why the turn ended (Vex 1231B DESIGN r2 C1).
            with conn.begin_nested():
                shutdown = restart_abort_cause(
                    _RowView(_turn_row(conn, record.turn_id))) == HOST_SHUTDOWN
                result = (restart_retry_delivery(store, record.id, connection=conn)
                          if shutdown else None)
            if result is not None:
                child = result[0]
                outcome = {"outcome": "retried", "child_delivery_id": child.id}
            elif shutdown:
                outcome = {"outcome": "refused", "reason": (
                    "not eligible: a session reset or /new, schedule, maintenance, "
                    "context maintenance, or already a restart retry")}
        except DeliveryRetryConflict as conflict:
            logger.info("delivery %s not restart-retried: %s", record.id, conflict)
            if shutdown:
                outcome = {"outcome": "refused", "reason": str(conflict)}
        except Exception as exc:
            logger.exception("restart retry of delivery %s failed; it stays FAILED", record.id)
            if shutdown:
                outcome = {"outcome": "error", "reason": str(exc) or exc.__class__.__name__}
        if outcome is not None:
            row = conn.execute(sa_text("""
                UPDATE inter_bot_deliveries
                SET metadata_json = COALESCE(metadata_json, '{}'::jsonb)
                    || jsonb_build_object(CAST(:key AS text), CAST(:outcome AS jsonb))
                WHERE id=:id RETURNING *
            """), {"id": record.id, "key": RESTART_RETRY_OUTCOME,
                  "outcome": json.dumps(outcome)}).mappings().first()
            if row is not None:
                failed = DeliveryRecord.from_mapping(row)
    return failed, child


def verified_retry_child(store: Any, original: DeliveryRecord) -> DeliveryRecord | None:
    """The one retry child (manual or restart) of ``original``, or None.

    Readers that follow a FAILED original to its retry (answer route, ordinary
    chat receipts) use this single rule: deterministic key, matching marker,
    same sender/target, and server-stamped retry provenance.
    """
    for policy in (MANUAL_RETRY, RESTART_RETRY):
        rows = store.list(
            sender_bot_id=original.sender_bot_id, target_bot_id=original.target_bot_id,
            idempotency_key=policy.key(original.id), limit=1,
        )
        child = rows[0] if rows else None
        if (child is not None
                and child.metadata.get(policy.marker) == original.id
                and child.metadata.get("_request_identity_producer") == RETRY_PRODUCER):
            return child
    return None
