"""Approval notices in approver bots' chats (TASK-1661).

A pending ``ops_run`` approval card used to wait until a person or a grant
holder happened to look. This leader-only pump puts it in front of every bot
that may decide it, and tells them when it is decided.

* Who: exactly the eligible approvers (:mod:`.approver_eligibility`), checked
  live at enqueue, again at claim (``claim_fence``) and again on every read.
  No new authority: the notice only says the card exists; deciding it goes
  through ``ops_approval_resolve``, which re-checks everything.
* Request phase: each (pending card, eligible approver) pair gets one row in
  ``approval_approver_fanout`` and one ``APPROVAL_REQUEST`` delivery keyed
  ``approval-request:<request>:<approver>``. The row and the delivery commit
  in one transaction under the approver's ``lock_target``.
* Resolution phase: once the card leaves pending, a still-QUEUED request
  notice is cancelled (nothing to decide any more). A notice that already
  reached the approver gets one ``steer_only`` ``APPROVAL_RESOLVED`` follow-up
  keyed ``approval-resolved:<request>:<approver>``; it steers a running turn
  and is settled not-needed otherwise. The resolver itself gets none.
* Revocation: an approver that is no longer eligible while the card is still
  pending has its queued notice cancelled and its row marked ``revoked``.
* Masked text only: a notice carries the request id, requester, operation and
  ``display_args`` (``x-sensitive`` values masked), never the stored prompt,
  subject or raw arguments, and never the resolver's reason.
* A pair whose pass fails rolls back whole and is retried on the next pass.
* ``approver_fanout_mode``: ``notify`` (default), ``report`` (logs the pending
  count; writes nothing) or ``off``.

The table is created once (``ensure_fanout_schema``). On that create, the cards
already pending are baselined for the holders of that moment, so history
never notifies.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text as sa_text
from sqlmodel import Session

from ..approval_models import KIND_MCP, OPS_RUN_TOOL, REQ_PENDING, ToolApprovalRequest
from ..bot_identities import SYSTEM_BOT_ID
from ..bot_tags import BehaviorTag
from ..inter_bot_delivery_ledger import DeliveryTransitionEvents
from ..inter_bot_delivery_record import DeliveryRecord
from ..inter_bot_steer_only import STEER_ONLY_KEY
from ..utils.db import is_postgresql
from ..utils.schema import SchemaBootstrapGuard, ensure_relations
from .approver_eligibility import approver_eligible, holder_and_tags

log = logging.getLogger(__name__)

TABLE = "approval_approver_fanout"
REQUEST_KIND = "APPROVAL_REQUEST"
RESOLVED_KIND = "APPROVAL_RESOLVED"
KINDS = frozenset({REQUEST_KIND, RESOLVED_KIND})
REQUEST_PROMPT_KEY = "approvals.approver_request"
RESOLVED_PROMPT_KEY = "approvals.approver_resolved"
REQUEST_KEY_PREFIX = "approval-request:"
RESOLVED_KEY_PREFIX = "approval-resolved:"
#: The delivery metadata key naming the approval request a notice is about.
REQUEST_ID_KEY = "approval_request_id"
BATCH_SIZE = 20
MODE_SETTING = "approver_fanout_mode"
NOTIFY, REPORT, OFF = "notify", "report", "off"
MODES = (NOTIFY, REPORT, OFF)
INELIGIBLE = "approval_ineligible"
WITHDRAWN = "approval_resolved_before_delivery"

# Row states: the request notice is out; then exactly one terminal state.
REQUESTED = "requested"
BASELINE, REVOKED, CANCELLED, FOLLOWED_UP, SELF = (
    "baseline", "revoked", "cancelled", "followed_up", "resolver")

_schema_guard = SchemaBootstrapGuard()
_RELATIONS = {
    TABLE: f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            request_id TEXT NOT NULL,
            approver_bot_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            state TEXT NOT NULL,
            request_delivery_id TEXT,
            resolved_delivery_id TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (request_id, approver_bot_id)
        )
    """,
    f"ix_{TABLE}_open": f"""
        CREATE INDEX IF NOT EXISTS ix_{TABLE}_open ON {TABLE} (request_id)
        WHERE state = '{REQUESTED}'
    """,
}
_PENDING_OPS = f"""
    request_kind = '{KIND_MCP}' AND tool_name = '{OPS_RUN_TOOL}' AND status = '{REQ_PENDING}'
    AND operations_snapshot_json IS NOT NULL
"""


def ensure_fanout_schema(engine) -> None:
    """Create the fan-out table once per engine (PostgreSQL only)."""
    if is_postgresql(engine):
        _schema_guard.run(engine, "approval-approver-fanout", _bootstrap)


def _bootstrap(conn) -> None:
    created = ensure_relations(conn, _RELATIONS)
    if TABLE not in created:
        return
    holders = _holders(conn)
    if not holders:
        return
    baselined = conn.execute(sa_text(f"""
        INSERT INTO {TABLE} (request_id, approver_bot_id, user_id, state)
        SELECT r.id, h.bot, r.user_id, '{BASELINE}'
        FROM tool_approval_requests r CROSS JOIN unnest(CAST(:holders AS text[])) AS h(bot)
        WHERE {_PENDING_OPS}
        ON CONFLICT DO NOTHING
    """), {"holders": sorted(holders)}).rowcount
    log.info("%s created; %d already-pending approver pairs baselined (never notified)", TABLE, baselined)


def _holders(conn) -> frozenset[str]:
    """Bots holding ``approve_ops`` now (the tag snapshot; private bots excluded)."""
    from ..bot_tags.facade import BotTags

    tags = BotTags().snapshot(conn)
    if not tags.available:
        return frozenset()
    return frozenset(bot for bot in tags.known_bots - tags.private_bots
                     if tags.has(bot, BehaviorTag.APPROVE_OPS))


def claim_fence(conn, head: dict[str, Any]) -> str | None:
    """Why a claimed approval notice must not run now, or None to let it run.

    Called by the delivery claim on the target's ``lock_target`` transaction
    (Codex B1): a grant revoked or a requester made private after enqueue
    never reaches the approver. A request notice also needs the card pending.
    """
    if head.get("message_kind") not in KINDS:
        return None
    metadata = head.get("metadata_json") or {}
    payload = head.get("payload_json") if isinstance(head.get("payload_json"), dict) else {}
    row = _request(conn, metadata.get(REQUEST_ID_KEY))
    holds, tags = holder_and_tags(conn, head["target_bot_id"])
    if not approver_eligible(holds=holds, tags=tags, row=row, approver=head["target_bot_id"],
                             user_id=payload.get("user")):
        return INELIGIBLE
    if head["message_kind"] == REQUEST_KIND and row.status != REQ_PENDING:
        return WITHDRAWN
    return None


def _request(conn, request_id: Any) -> ToolApprovalRequest | None:
    """The approval row, read on ``conn`` (inside the caller's lock), or None."""
    if not isinstance(request_id, str) or not request_id:
        return None
    with Session(bind=conn) as session:
        row = session.get(ToolApprovalRequest, request_id)
        if row is not None:
            session.expunge(row)
        return row


@dataclass
class PassResult:
    notified: list[DeliveryRecord] = field(default_factory=list)
    settled: int = 0
    errors: int = 0


class ApprovalFanoutPump:
    """Request and resolution notices for eligible approver bots."""

    def __init__(self, store: Any, *, batch_size: int = BATCH_SIZE) -> None:
        self.store = store
        self.batch_size = batch_size
        self._last_report: int | None = None

    @property
    def engine(self):
        return self.store.engine

    def mode(self) -> str:
        from ..runtime_setting_resolution import resolve_global_runtime_setting

        try:
            mode = str(resolve_global_runtime_setting(self.store.config, MODE_SETTING) or NOTIFY).strip().lower()
        except Exception:
            log.exception("Unreadable %s; approver notices report only this pass", MODE_SETTING)
            return REPORT
        if mode not in MODES:
            log.warning("Unknown %s %r; approver notices report only", MODE_SETTING, mode)
            return REPORT
        return mode

    def run_pass(self) -> PassResult:
        """One bounded pass; the notices it enqueued (the caller announces them)."""
        result = PassResult()
        if not is_postgresql(self.engine):
            return result
        mode = self.mode()
        if mode == OFF:
            return result
        if mode == REPORT:
            pending = len(self._new_pairs())
            if pending != self._last_report:
                log.info("approver notices (report only): %d pending pairs", pending)
                self._last_report = pending
            return result
        for request_id, approver in self._new_pairs():
            self._step(result, self._notify, request_id, approver)
        for request_id, approver in self._open_rows():
            self._step(result, self._settle, request_id, approver)
        if result.notified or result.settled or result.errors:
            log.info("approver notices: %d enqueued, %d settled, %d errors",
                     len(result.notified), result.settled, result.errors)
        return result

    @staticmethod
    def _step(result: PassResult, action, request_id: str, approver: str) -> None:
        try:
            outcome = action(request_id, approver)
        except Exception:  # rolled back whole: retried next pass
            log.exception("approver notice %s -> %s failed", request_id, approver)
            result.errors += 1
            return
        if isinstance(outcome, DeliveryRecord):
            result.notified.append(outcome)
        elif outcome:
            result.settled += 1

    def _new_pairs(self) -> list[tuple[str, str]]:
        """Pending ops cards x current holders with no fan-out row yet, oldest first."""
        with self.engine.connect() as conn:
            holders = _holders(conn)
            if not holders:
                return []
            return [(r["id"], r["bot"]) for r in conn.execute(sa_text(f"""
                SELECT r.id, h.bot FROM tool_approval_requests r
                CROSS JOIN unnest(CAST(:holders AS text[])) AS h(bot)
                WHERE {_PENDING_OPS} AND r.bot_id <> h.bot
                  AND NOT EXISTS (SELECT 1 FROM {TABLE} f
                                  WHERE f.request_id = r.id AND f.approver_bot_id = h.bot)
                ORDER BY r.created_at, r.id, h.bot LIMIT :limit
            """), {"holders": sorted(holders), "limit": self.batch_size}).mappings()]

    def _open_rows(self) -> list[tuple[str, str]]:
        with self.engine.connect() as conn:
            return [(r["request_id"], r["approver_bot_id"]) for r in conn.execute(sa_text(f"""
                SELECT request_id, approver_bot_id FROM {TABLE} WHERE state = '{REQUESTED}'
                ORDER BY updated_at LIMIT :limit
            """), {"limit": self.batch_size}).mappings()]

    def _notify(self, request_id: str, approver: str) -> DeliveryRecord | bool:
        """Record the pair and enqueue its request notice, or record why not."""
        from .turn_admission import lock_target

        with self.engine.begin() as conn:
            lock_target(conn, approver)
            row = _request(conn, request_id)
            if row is None or row.status != REQ_PENDING:
                return False
            holds, tags = holder_and_tags(conn, approver)
            if not approver_eligible(holds=holds, tags=tags, row=row, approver=approver,
                                     user_id=row.user_id):
                return False  # not an approver of this card; a later grant is a new pair
            inserted = conn.execute(sa_text(f"""
                INSERT INTO {TABLE} (request_id, approver_bot_id, user_id, state)
                VALUES (:req, :bot, :user, '{REQUESTED}') ON CONFLICT DO NOTHING
            """), {"req": request_id, "bot": approver, "user": row.user_id}).rowcount
            if not inserted:
                return False
            card = notice_card(row)
            if card is None:
                self._finish(conn, request_id, approver, INELIGIBLE)
                return True
            record = self._enqueue(conn, row, approver, REQUEST_KIND, card)
            conn.execute(sa_text(f"""
                UPDATE {TABLE} SET request_delivery_id=:delivery, updated_at=now()
                WHERE request_id=:req AND approver_bot_id=:bot
            """), {"delivery": record.id, "req": request_id, "bot": approver})
            return record

    def _settle(self, request_id: str, approver: str) -> DeliveryRecord | bool:
        """Close one open pair: revoke, withdraw, or follow up with the decision."""
        from .turn_admission import lock_target

        with self.engine.begin() as conn:
            lock_target(conn, approver)
            pair = conn.execute(sa_text(f"""
                SELECT * FROM {TABLE} WHERE request_id=:req AND approver_bot_id=:bot
                  AND state = '{REQUESTED}' FOR UPDATE
            """), {"req": request_id, "bot": approver}).mappings().first()
            if pair is None:
                return False
            row = _request(conn, request_id)
            holds, tags = holder_and_tags(conn, approver)
            eligible = approver_eligible(holds=holds, tags=tags, row=row, approver=approver,
                                         user_id=pair["user_id"])
            pending = row is not None and row.status == REQ_PENDING
            if pending and eligible:
                conn.execute(sa_text(f"UPDATE {TABLE} SET updated_at=now() WHERE request_id=:req"
                                     " AND approver_bot_id=:bot"), {"req": request_id, "bot": approver})
                return False
            cancelled = self._cancel_queued(conn, pair["request_delivery_id"],
                                            INELIGIBLE if not eligible else WITHDRAWN)
            if not eligible:
                return self._finish(conn, request_id, approver, REVOKED)
            if cancelled:
                return self._finish(conn, request_id, approver, CANCELLED)
            if row.resolved_by_bot == approver:
                return self._finish(conn, request_id, approver, SELF)
            card = notice_card(row)
            if card is None:
                return self._finish(conn, request_id, approver, REVOKED)
            record = self._enqueue(conn, row, approver, RESOLVED_KIND, card)
            conn.execute(sa_text(f"""
                UPDATE {TABLE} SET state='{FOLLOWED_UP}', resolved_delivery_id=:delivery,
                    updated_at=now() WHERE request_id=:req AND approver_bot_id=:bot
            """), {"delivery": record.id, "req": request_id, "bot": approver})
            return record

    @staticmethod
    def _finish(conn, request_id: str, approver: str, state: str) -> bool:
        conn.execute(sa_text(f"""
            UPDATE {TABLE} SET state=:state, updated_at=now()
            WHERE request_id=:req AND approver_bot_id=:bot
        """), {"state": state, "req": request_id, "bot": approver})
        return True

    @staticmethod
    def _cancel_queued(conn, delivery_id: str | None, reason: str) -> bool:
        """Cancel a still-queued request notice; True when the approver never saw it.

        Unseen covers a notice cancelled here, one already fenced at claim
        (CANCELLED) and one that failed: none of them gets a follow-up.
        """
        if not delivery_id:
            return False
        cancelled = conn.execute(sa_text("""
            UPDATE inter_bot_deliveries SET status='CANCELLED', next_retry_at=NULL,
                last_error=:reason, updated_at=now()
            WHERE id=:id AND status='QUEUED' RETURNING *
        """), {"id": delivery_id, "reason": reason}).mappings().first()
        if cancelled is not None:
            DeliveryTransitionEvents.stage(conn, cancelled, reason_code="approval_resolved")
            return True
        status = conn.execute(sa_text("SELECT status FROM inter_bot_deliveries WHERE id=:id"),
                              {"id": delivery_id}).scalar()
        return status in (None, "CANCELLED", "FAILED")

    def _enqueue(self, conn, row, approver: str, kind: str, card: dict[str, Any]) -> DeliveryRecord:
        resolved = kind == RESOLVED_KIND
        variables = {**card, "status": row.status, "resolved_by": _resolver_label(row)} if resolved else card
        message = notice_text(RESOLVED_PROMPT_KEY if resolved else REQUEST_PROMPT_KEY, approver, variables)
        prefix = RESOLVED_KEY_PREFIX if resolved else REQUEST_KEY_PREFIX
        payload = {
            "bot_id": approver, "user": row.user_id, "session_id": None,
            "messages": [{"role": "user", "content": message}],
            "prefer_steer": True, "reply_mode": "async",
            "extract_memory": False, "augment_memory": False, "stream": True,
        }
        if resolved:
            payload[STEER_ONLY_KEY] = True
        record, _ = self.store.enqueue(
            sender_bot_id=SYSTEM_BOT_ID, target_bot_id=approver, message=message,
            idempotency_key=f"{prefix}{row.id}:{approver}", message_kind=kind,
            on_gate="hold", connection=conn,
            metadata={REQUEST_ID_KEY: row.id, "requested_by": row.bot_id,
                      "operation": card["operation"]},
            payload=payload,
        )
        return record


def _resolver_label(row) -> str:
    """Who decided, for other approvers: a bot slug or "a person" (never the reason)."""
    return row.resolved_by_bot or "a person"


def notice_card(row) -> dict[str, Any] | None:
    """The masked prompt variables for one card, or None when it can't be shown safely."""
    from .routes.ops_approvals import _card

    card = _card(row)
    if card is None:
        return None
    return {"request_id": card.request_id, "requested_by": card.requested_by,
            "operation": card.operation, "title": card.title or card.operation,
            "risk_level": card.risk_level or "unknown", "severity": card.severity,
            "args": json.dumps(card.args, ensure_ascii=False, sort_keys=True, default=str)}


def notice_text(key: str, bot_id: str, variables: dict[str, Any]) -> str:
    """The registry body of ``key`` for ``bot_id``, rendered; raises when blank.

    The seed is never sent in its place (prompt rule): the caller's
    transaction rolls back and the pair is retried next pass.
    """
    from ..prompt_registry import get_prompt_resolver

    rendered = get_prompt_resolver().render(key=key, variables=variables, scope_type="bot", scope_id=bot_id)
    if not (rendered or "").strip():
        raise ValueError(f"prompt {key} rendered blank for {bot_id}")
    return rendered
