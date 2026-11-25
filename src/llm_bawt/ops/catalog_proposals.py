"""Bot-proposed ops catalog edits that a person approves (TASK-2630).

Catalog writes run privileged host commands, so only a person may make them
(BawtHub's admin check in front of the service bearer). Bots still find the
edits that are needed. This module lets a bot record the exact change, one
batch of operation edits with a reason, and lets the person apply the whole
batch with one decision.

Every item names the operation version it was written against. A decision
applies nothing if any operation has moved since; the batch is marked stale
and the bot proposes again. Approval writes through
:meth:`OpsStore.update_operation` with ``expect_version``, attributed to the
person, so revision history, catalog notifications and pending-approval
invalidation behave exactly like a manual edit. Bots can never decide.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from sqlalchemy import Column, DateTime, Text, update
from sqlmodel import Field, Session, SQLModel, select

from ..utils.schema import SchemaBootstrapGuard
from .store import _OP_WRITABLE

PENDING, APPROVED, REJECTED, STALE = "pending", "approved", "rejected", "stale"
# The slug is immutable and identifies the target; everything else may change.
EDITABLE = frozenset(_OP_WRITABLE - {"slug"})
MAX_ITEMS = 50
MAX_REASON = 500


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class OpsCatalogProposal(SQLModel, table=True):
    """One proposed edit of one operation; items of a batch share ``batch_id``."""
    __tablename__ = "ops_catalog_proposals"
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    batch_id: str = Field(index=True)
    position: int
    operation_slug: str = Field(index=True)
    base_version: int
    before_json: str = Field(sa_column=Column(Text, nullable=False))
    after_json: str = Field(sa_column=Column(Text, nullable=False))
    reason: str = Field(sa_column=Column(Text, nullable=False))
    status: str = Field(default=PENDING, index=True)
    proposed_by: str
    decided_by: str | None = None
    decided_at: datetime | None = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))
    result_version: int | None = None
    created_at: datetime = Field(default_factory=_utcnow,
                                 sa_column=Column(DateTime(timezone=True), nullable=False))


class ProposalError(ValueError):
    """A proposal or decision that cannot be recorded as given."""


_guard = SchemaBootstrapGuard()


def ensure_schema(engine) -> None:
    def create(conn):
        SQLModel.metadata.create_all(conn, tables=[OpsCatalogProposal.__table__])
    _guard.run(engine, "ops-catalog-proposals-v1", create)


def _norm(field: str, value: Any) -> Any:
    # The store canonicalizes these two from dicts; compare in stored form.
    if field in ("args_schema_json", "args_defaults_json") and isinstance(value, dict):
        from .validation import canonical_json
        return canonical_json(value)
    return value


def propose(store, *, items: Sequence[Mapping[str, Any]], reason: str, proposed_by: str) -> dict[str, Any]:
    """Record one pending batch. Each item: ``slug``, ``expect_version`` and
    ``changes`` (field -> new value). Refuses unknown fields, no-op changes,
    a moved base version and anything the catalog would reject on save."""
    store._require()
    reason = (reason or "").strip()
    if not reason or len(reason) > MAX_REASON:
        raise ProposalError(f"a proposal needs a reason of 1..{MAX_REASON} characters")
    if not proposed_by:
        raise ProposalError("proposer identity required")
    if not items or len(items) > MAX_ITEMS:
        raise ProposalError(f"a batch has 1..{MAX_ITEMS} items")
    ensure_schema(store.engine)
    batch_id = uuid.uuid4().hex
    rows, seen = [], set()
    for position, item in enumerate(items):
        slug, version, changes = item.get("slug"), item.get("expect_version"), item.get("changes")
        if not isinstance(slug, str) or slug in seen:
            raise ProposalError("each item needs a distinct operation slug")
        seen.add(slug)
        if type(version) is not int:
            raise ProposalError(f"{slug}: expect_version must be the integer version you read")
        if not isinstance(changes, Mapping) or not changes:
            raise ProposalError(f"{slug}: changes must name at least one field")
        unknown = sorted(set(changes) - EDITABLE)
        if unknown:
            raise ProposalError(f"{slug}: fields not editable: {', '.join(unknown)}")
        current = store.get_operation_by_slug(slug)
        if current is None or current.soft_deleted_at is not None:
            raise ProposalError(f"{slug}: no such operation")
        if current.version != version:
            raise ProposalError(f"{slug}: operation is at version {current.version}, not {version}; re-read it")
        clean = {k: _norm(k, v) for k, v in changes.items()}
        before = {k: getattr(current, k) for k in clean}
        after = {k: v for k, v in clean.items() if before[k] != v}
        if not after:
            raise ProposalError(f"{slug}: changes match the current operation")
        _dry_run(store, slug, after)
        rows.append(OpsCatalogProposal(
            batch_id=batch_id, position=position, operation_slug=slug, base_version=version,
            before_json=json.dumps({k: before[k] for k in after}), after_json=json.dumps(after),
            reason=reason, proposed_by=proposed_by))
    with Session(store.engine) as session:
        session.add_all(rows)
        _emit(store, session, batch_id, PENDING, proposed_by, len(rows), reason)
        session.commit()
    return {"batch_id": batch_id, "status": PENDING, "items": len(rows)}


def _dry_run(store, slug: str, after: Mapping[str, Any]) -> None:
    """Run the edit through the real write path, flush, and roll back, so a
    proposal can never record an edit the catalog would refuse on approval."""
    with Session(store.engine) as session:
        try:
            row = store._lock_operation(session, slug)
            store.apply_update_in(session, row, store._clean_op(dict(after)), actor="dry-run",
                                  side_effects=False)
        except ValueError as exc:
            raise ProposalError(f"{slug}: {exc}") from exc
        finally:
            session.rollback()


def _emit(store, session, batch_id: str, status: str, actor: str, count: int, reason: str) -> None:
    events = getattr(store, "events", None)
    if events is None:
        return
    word = {PENDING: "awaiting approval", APPROVED: "approved", REJECTED: "rejected", STALE: "stale"}[status]
    noun = "edit" if count == 1 else "edits"
    events.append_at_commit(
        session.connection(), event_key=f"ops-catalog-proposal:{batch_id}:{status}",
        kind="approval", source="ops", user_id=None,
        tone={PENDING: "info", APPROVED: "success"}.get(status, "warning"),
        summary=f"Ops catalog · {count} {noun} {word}: {reason}"[:500],
        actor_type="bot" if status == PENDING else "user", actor_id=actor,
        subject_type="ops_catalog_proposal", subject_id=batch_id)


def _item_api(row: OpsCatalogProposal) -> dict[str, Any]:
    return {"id": row.id, "operation_slug": row.operation_slug, "base_version": row.base_version,
            "before": json.loads(row.before_json), "after": json.loads(row.after_json),
            "status": row.status, "result_version": row.result_version}


def list_batches(store, *, status: str | None = PENDING, limit: int = 50) -> list[dict[str, Any]]:
    if store.engine is None:
        return []
    ensure_schema(store.engine)
    with Session(store.engine) as session:
        stmt = select(OpsCatalogProposal)
        if status:
            stmt = stmt.where(OpsCatalogProposal.status == status)
        rows = session.exec(stmt.order_by(OpsCatalogProposal.created_at.desc(),
                                          OpsCatalogProposal.position)).all()
    batches: dict[str, dict[str, Any]] = {}
    for row in rows:
        b = batches.get(row.batch_id)
        if b is None:
            if len(batches) >= limit:
                continue
            b = batches[row.batch_id] = {
                "batch_id": row.batch_id, "reason": row.reason, "proposed_by": row.proposed_by,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "decided_by": row.decided_by,
                "decided_at": row.decided_at.isoformat() if row.decided_at else None,
                "items": []}
        b["items"].append(_item_api(row))
    for b in batches.values():
        states = {i["status"] for i in b["items"]}
        b["status"] = states.pop()
    return list(batches.values())


def _refuse_bot(engine, actor: str) -> None:
    if not actor or not actor.strip():
        raise PermissionError("a person must decide")
    from ..memory.guidance import _decider
    with engine.connect() as conn:
        try:
            _decider(conn, actor, actor)
        except PermissionError as exc:
            raise PermissionError("a bot cannot decide catalog edits") from exc


def decide(store, *, batch_id: str, decided_by: str, approve: bool) -> dict[str, Any]:
    """One-shot decision for a whole batch, in one transaction.

    The proposal rows and every target operation are locked (operations in
    slug order), every base version is rechecked under the locks, and the
    edits, revisions, pending-approval invalidation and the decision commit
    together: all applied, or none (``stale`` when an operation moved). A
    repeated decision reports the recorded result and changes nothing.
    """
    store._require()
    ensure_schema(store.engine)
    _refuse_bot(store.engine, decided_by)
    with Session(store.engine) as session:
        items = session.exec(select(OpsCatalogProposal).where(
            OpsCatalogProposal.batch_id == batch_id).order_by(OpsCatalogProposal.position)).all()
        if not items:
            raise LookupError("no such proposal")
        # Locks the batch rows (write lock first, also on SQLite) and refuses a second decision.
        claimed = session.execute(update(OpsCatalogProposal).where(
            OpsCatalogProposal.batch_id == batch_id, OpsCatalogProposal.status == PENDING,
        ).values(decided_by=decided_by, decided_at=_utcnow())
            .execution_options(synchronize_session=False)).rowcount
        if claimed != len(items):
            session.rollback()
            return {"batch_id": batch_id, "status": _recorded_status(store, batch_id), "recorded": True}
        reason, count = items[0].reason, len(items)
        targets = sorted(items, key=lambda i: i.operation_slug)
        if not approve:
            outcome, versions, moved = REJECTED, {}, []
        else:
            ops = {i.operation_slug: store._lock_operation(session, i.operation_slug) for i in targets}
            moved = [slug for slug, op in ops.items()
                     if op is None or op.soft_deleted_at is not None
                     or op.version != next(i.base_version for i in items if i.operation_slug == slug)]
            versions = {}
            if moved:
                outcome = STALE
            else:
                outcome = APPROVED
                try:
                    for i in targets:
                        row = store.apply_update_in(session, ops[i.operation_slug],
                                                    store._clean_op(json.loads(i.after_json)),
                                                    actor=decided_by)
                        versions[i.id] = row.version
                except ValueError as exc:
                    session.rollback()
                    raise ProposalError(f"nothing applied: {exc}") from exc
        for i in items:
            session.execute(update(OpsCatalogProposal).where(OpsCatalogProposal.id == i.id)
                            .values(status=outcome, result_version=versions.get(i.id))
                            .execution_options(synchronize_session=False))
        applied = {i.operation_slug: versions[i.id] for i in items if i.id in versions}
        _emit(store, session, batch_id, outcome, decided_by, count, reason)
        session.commit()
    out: dict[str, Any] = {"batch_id": batch_id, "status": outcome}
    if outcome == STALE:
        out["moved"] = moved
    if outcome == APPROVED:
        out["applied"] = applied
    return out


def _recorded_status(store, batch_id: str) -> str:
    with Session(store.engine) as session:
        states = set(session.exec(select(OpsCatalogProposal.status).where(
            OpsCatalogProposal.batch_id == batch_id)).all())
    return states.pop() if states else "unknown"
