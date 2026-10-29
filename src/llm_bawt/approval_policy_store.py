"""Approval policy persistence, schema, and compilation."""

from __future__ import annotations
import json
import logging
from typing import Any
from sqlalchemy import text, func
from sqlalchemy.exc import SQLAlchemyError
from .approval_validation import validate_policy
from agent_bridge.approval import PolicyBundle, compute_etag
from sqlmodel import Session, SQLModel, select
from .approval_models import (
    ApprovalStoreUnavailable,
    CONT_NOT_NEEDED,
    EXEC_FAILED,
    EXEC_NOT_APPLICABLE,
    KIND_HARNESS,
    KIND_MCP,
    REQ_PENDING,
    REQ_SUPERSEDED,
    ToolApprovalDecision,
    ToolApprovalPolicy,
    ToolApprovalPolicyRevision,
    ToolApprovalRequest,
    _new_id,
    _utcnow,
)
from .system_events import SystemEventEmitter
from .utils.config import Config, has_database_credentials
from .utils.schema import SchemaBootstrapGuard

from .approval_request_store import ApprovalRequestStoreMixin
from .approval_defaults import _DEFAULT_POLICIES, _OPS_DEFAULT_POLICIES

# Fields a caller may set on create/update. Anything else is ignored.
_POLICY_WRITABLE = {
    "enabled",
    "backend_scope",
    "tool_name",
    "matcher_type",
    "pattern",
    "field",
    "action",
    "severity",
    "category",
    "approval_prompt",
    "order",
}


logger = logging.getLogger(__name__)


class ToolApprovalPolicyStore(ApprovalRequestStoreMixin):
    """DB access for approval policies + request audit log."""

    _schema_guard = SchemaBootstrapGuard()

    def __init__(self, config: Config, engine: Any = None):
        """``engine`` overrides credential-derived connection resolution.

        Used by the tenant seeder (and tests), which already owns a connected
        engine and must not re-resolve one from a partial config.
        """
        self.config = config
        self.engine = None
        self.events: SystemEventEmitter | None = None
        if engine is not None:
            self.engine = engine
            self._ensure_tables_exist()
            self._ensure_event_ledger()
            return
        if not has_database_credentials(config):
            return
        try:
            from .utils.db import get_shared_engine

            self.engine = get_shared_engine(config)  # TASK-202: shared pool
            if self.engine is None:
                return
            self._ensure_tables_exist()
            self._ensure_event_ledger()
        except Exception as e:  # noqa: BLE001
            self.engine = None
            logger.warning("Tool approval policies DB unavailable: %s", e)

    def _ensure_event_ledger(self) -> None:
        try:
            self.events = SystemEventEmitter(self.engine)
        except Exception:
            logger.exception("Approval event ledger unavailable; approvals continue")

    def _ensure_tables_exist(self) -> None:
        if self.engine is None:
            return

        def bootstrap(conn) -> None:
            SQLModel.metadata.create_all(
                bind=conn,
                tables=[
                    ToolApprovalPolicy.__table__,
                    ToolApprovalRequest.__table__,
                    ToolApprovalPolicyRevision.__table__,
                    ToolApprovalDecision.__table__,
                ],
            )
            self._migrate_add_columns(conn)

        self._schema_guard.run(self.engine, "tool-approval-policy-store", bootstrap)

    def _migrate_add_columns(self, conn) -> None:
        """Add columns introduced after initial schema creation.

        TASK-639 extends ``tool_approval_requests`` with the MCP-kind execution
        + continuation state so the same table serves both the classic bridge-
        hook (harness) flow and the new server-side MCP flow. Idempotent via
        ``ADD COLUMN IF NOT EXISTS``; safe to run every bootstrap.

        Postgres-only: SQLite (test env) got the full column list from the
        preceding ``create_all`` and doesn't accept ``ADD COLUMN IF NOT EXISTS``
        syntax. Skip cleanly there.
        """
        if conn.dialect.name != "postgresql":
            return
        migrations = [
            # request kind + SDK/MCP correlation
            f"ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            f"request_kind VARCHAR(16) NOT NULL DEFAULT '{KIND_HARNESS}'",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "tool_use_id VARCHAR(128)",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "mcp_server VARCHAR(128)",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "invocation_hash VARCHAR(64)",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "caller_context_json TEXT",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "continuation_capable BOOLEAN NOT NULL DEFAULT FALSE",
            # execution state machine
            f"ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            f"execution_state VARCHAR(16) NOT NULL DEFAULT '{EXEC_NOT_APPLICABLE}'",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "execution_attempts INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "execution_started_at TIMESTAMP WITH TIME ZONE",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "execution_finished_at TIMESTAMP WITH TIME ZONE",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "execution_error TEXT",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "result_json TEXT",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "result_is_error BOOLEAN",
            # continuation outbox
            f"ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            f"continuation_state VARCHAR(16) NOT NULL DEFAULT '{CONT_NOT_NEEDED}'",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "continuation_attempts INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "continuation_last_error TEXT",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "continuation_next_attempt_at TIMESTAMP WITH TIME ZONE",
            "ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS "
            "continuation_delivered_at TIMESTAMP WITH TIME ZONE",
            # indexes matching the SQLModel Field(index=True) declarations
            "CREATE INDEX IF NOT EXISTS ix_tool_approval_requests_request_kind "
            "ON tool_approval_requests (request_kind)",
            "CREATE INDEX IF NOT EXISTS ix_tool_approval_requests_tool_use_id "
            "ON tool_approval_requests (tool_use_id)",
            "CREATE INDEX IF NOT EXISTS ix_tool_approval_requests_execution_state "
            "ON tool_approval_requests (execution_state)",
            "CREATE INDEX IF NOT EXISTS ix_tool_approval_requests_continuation_state "
            "ON tool_approval_requests (continuation_state)",
            "CREATE INDEX IF NOT EXISTS "
            "ix_tool_approval_requests_continuation_next_attempt_at "
            "ON tool_approval_requests (continuation_next_attempt_at)",
        ]
        for name, sql_type in {
            "resolution_message": "TEXT NOT NULL DEFAULT ''",
            "resolved_via": "TEXT NOT NULL DEFAULT 'user'",  # TASK-1609
            "continuation_owner": "VARCHAR(16) NOT NULL DEFAULT 'client'",
            "operations_snapshot_json": "TEXT",
            "execution_claim_token": "VARCHAR(64)",
            "continuation_claim_token": "VARCHAR(64)",
            "continuation_id": "VARCHAR(128)",
            "grant_state": "VARCHAR(24) NOT NULL DEFAULT 'pending'",
            "grant_error": "TEXT",
        }.items():
            migrations.append(
                f"ALTER TABLE tool_approval_requests ADD COLUMN IF NOT EXISTS {name} {sql_type}"
            )
        for name in (
            "session_key",
            "session_id",
            "request_id",
            "tool_use_id",
            "outcome",
        ):
            migrations.append(
                f"ALTER TABLE tool_approval_decisions ADD COLUMN IF NOT EXISTS {name} TEXT"
            )
        for stmt in migrations:
            conn.execute(text(stmt))

        # TASK-873: pre-existing unroutable MCP rows remain as audit records,
        # but cannot stay actionable. Terminalize them before validating the
        # database invariant that every new pending MCP approval is routable.
        invalid_identity = """
            bot_id IS NULL OR btrim(bot_id) = '' OR lower(btrim(bot_id)) IN ('unknown','none','null','undefined')
            OR user_id IS NULL OR btrim(user_id) = '' OR lower(btrim(user_id)) IN ('unknown','none','null','undefined')
            OR turn_id IS NULL OR btrim(turn_id) = '' OR lower(btrim(turn_id)) IN ('unknown','none','null','undefined')
            OR trigger_message_id IS NULL OR btrim(trigger_message_id) = '' OR lower(btrim(trigger_message_id)) IN ('unknown','none','null','undefined')
            OR session_key IS NULL OR btrim(session_key) = '' OR lower(btrim(session_key)) IN ('unknown','none','null','undefined')
            OR backend IS NULL OR btrim(backend) = '' OR lower(btrim(backend)) IN ('unknown','none','null','undefined')
            OR tool_use_id IS NULL OR btrim(tool_use_id) = '' OR lower(btrim(tool_use_id)) IN ('unknown','none','null','undefined')
            OR mcp_server IS NULL OR btrim(mcp_server) = '' OR lower(btrim(mcp_server)) IN ('unknown','none','null','undefined')
            OR invocation_hash IS NULL OR invocation_hash !~ '^[0-9a-f]{64}$'
            OR caller_context_json IS NULL OR btrim(caller_context_json) = ''
            OR continuation_capable IS NOT TRUE
        """
        conn.execute(
            text(f"""
            UPDATE tool_approval_requests
               SET status = '{REQ_SUPERSEDED}',
                   resolved_at = COALESCE(resolved_at, NOW()),
                   resolution_message = CASE WHEN btrim(COALESCE(resolution_message, '')) = ''
                       THEN 'TASK-873: superseded unroutable legacy MCP approval'
                       ELSE resolution_message END,
                   execution_state = '{EXEC_FAILED}',
                   execution_finished_at = COALESCE(execution_finished_at, NOW()),
                   execution_error = COALESCE(execution_error,
                       'Unroutable legacy MCP approval context; execution was blocked')
             WHERE request_kind = '{KIND_MCP}'
               AND status = '{REQ_PENDING}'
               AND ({invalid_identity})
        """)
        )
        conn.execute(
            text(f"""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                     WHERE conname = 'ck_tool_approval_requests_pending_mcp_routable'
                       AND conrelid = 'tool_approval_requests'::regclass
                ) THEN
                    ALTER TABLE tool_approval_requests
                    ADD CONSTRAINT ck_tool_approval_requests_pending_mcp_routable
                    CHECK (
                        request_kind <> '{KIND_MCP}' OR status <> '{REQ_PENDING}' OR NOT ({invalid_identity})
                    ) NOT VALID;
                END IF;
            END $$
        """)
        )
        conn.execute(
            text("""
            ALTER TABLE tool_approval_requests
            VALIDATE CONSTRAINT ck_tool_approval_requests_pending_mcp_routable
        """)
        )

    # ---- policy CRUD -------------------------------------------------------

    def list_all(self) -> list[ToolApprovalPolicy]:
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval policy database unavailable")
        with Session(self.engine) as session:
            return list(
                session.exec(
                    select(ToolApprovalPolicy).order_by(
                        ToolApprovalPolicy.order_index, ToolApprovalPolicy.id
                    )
                ).all()
            )

    def get(self, policy_id: str) -> ToolApprovalPolicy | None:
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval policy database unavailable")
        with Session(self.engine) as session:
            return session.get(ToolApprovalPolicy, policy_id)

    def _clean(self, data: dict[str, Any]) -> dict[str, Any]:
        data = dict(data)
        if "order_index" in data:
            if "order" in data and data["order"] != data["order_index"]:
                raise ValueError("order and order_index disagree")
            data["order"] = data.pop("order_index")
        out = {k: v for k, v in data.items() if k in _POLICY_WRITABLE}
        return out

    def create(
        self, data: dict[str, Any], actor: str | None = None
    ) -> ToolApprovalPolicy:
        return self._create_with_id(_new_id(), data, actor=actor)

    def _create_with_id(
        self,
        policy_id: str,
        data: dict[str, Any],
        *,
        actor: str | None = None,
    ) -> ToolApprovalPolicy:
        """Create one policy with a caller-selected stable id.

        Public CRUD still generates opaque ids. Catalog defaults use this seam
        so each seed is independently insert-if-missing and never overwrites an
        operator-edited row on later startups.
        """
        if self.engine is None:
            raise RuntimeError("Tool approval policies DB unavailable")
        clean = validate_policy(self._clean(data))
        now = _utcnow()
        row = ToolApprovalPolicy(
            id=policy_id,
            enabled=bool(clean.get("enabled", True)),
            backend_scope=str(clean.get("backend_scope", "*") or "*"),
            tool_name=str(clean.get("tool_name", "*") or "*"),
            matcher_type=str(clean.get("matcher_type", "always")),
            pattern=str(clean.get("pattern", "") or ""),
            field=str(clean.get("field", "") or ""),
            action=str(clean.get("action", "require_approval")),
            severity=str(clean.get("severity", "medium")),
            category=clean.get("category"),
            approval_prompt=clean.get("approval_prompt"),
            order_index=int(clean.get("order", 100) or 0),
            version=1,
            created_at=now,
            updated_at=now,
            created_by=actor,
            updated_by=actor,
        )
        with Session(self.engine) as session:
            session.add(row)
            self._revision(session, row, "create", actor)
            session.commit()
            session.refresh(row)
            return row

    def update(
        self, policy_id: str, data: dict[str, Any], actor: str | None = None
    ) -> ToolApprovalPolicy | None:
        if self.engine is None:
            raise RuntimeError("Tool approval policies DB unavailable")
        clean = self._clean(data)
        with Session(self.engine) as session:
            row = session.exec(
                select(ToolApprovalPolicy)
                .where(ToolApprovalPolicy.id == policy_id)
                .with_for_update()
            ).first()
            if row is None:
                return None
            self._revision(session, row, "baseline", row.updated_by)
            clean = validate_policy({**row.to_api(), **clean})
            if "enabled" in clean:
                row.enabled = bool(clean["enabled"])
            if "backend_scope" in clean:
                row.backend_scope = str(clean["backend_scope"] or "*")
            if "tool_name" in clean:
                row.tool_name = str(clean["tool_name"] or "*")
            if "matcher_type" in clean:
                row.matcher_type = str(clean["matcher_type"])
            if "pattern" in clean:
                row.pattern = str(clean["pattern"] or "")
            if "field" in clean:
                row.field = str(clean["field"] or "")
            if "action" in clean:
                row.action = str(clean["action"])
            if "severity" in clean:
                row.severity = str(clean["severity"])
            if "category" in clean:
                row.category = clean["category"]
            if "approval_prompt" in clean:
                row.approval_prompt = clean["approval_prompt"]
            if "order" in clean:
                row.order_index = int(clean["order"] or 0)
            row.version += 1
            row.updated_at = _utcnow()
            row.updated_by = actor
            self._revision(session, row, "update", actor)
            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    def delete(self, policy_id: str) -> bool:
        if self.engine is None:
            raise RuntimeError("Tool approval policies DB unavailable")
        with Session(self.engine) as session:
            row = session.exec(
                select(ToolApprovalPolicy)
                .where(ToolApprovalPolicy.id == policy_id)
                .with_for_update()
            ).first()
            if row is None:
                return False
            self._revision(session, row, "baseline", row.updated_by)
            row.version += 1
            self._revision(session, row, "delete", None)
            session.delete(row)
            session.commit()
            return True

    # ---- bundle compilation ------------------------------------------------

    def compile_bundle(self) -> PolicyBundle:
        """Compile all rows into the versioned bundle a bridge consumes."""
        try:
            policies = [row.to_policy() for row in self.list_all()]
        except SQLAlchemyError as exc:
            raise ApprovalStoreUnavailable(
                "Approval policy database unavailable"
            ) from exc
        etag = compute_etag(1, policies)
        return PolicyBundle(version=1, etag=etag, policies=policies)

    # ---- seeding -----------------------------------------------------------

    def seed_defaults(self) -> int:
        """Insert missing starter policies without overwriting operator edits.

        The legacy Bash rules retain their original whole-table-empty bootstrap
        behavior to avoid duplicating random-id seeds in existing databases.
        TASK-639 ops rules use stable ids and are inserted independently, so a
        newly introduced default appears on upgrade while an existing (possibly
        operator-edited or disabled) row is preserved byte-for-byte.
        """
        if self.engine is None:
            raise RuntimeError("Tool approval policies DB unavailable")
        seeded = 0
        if not self.list_all():
            for default in _DEFAULT_POLICIES:
                self.create(default, actor="seed")
                seeded += 1
        for default in _OPS_DEFAULT_POLICIES:
            policy_id = str(default["id"])
            if self.get(policy_id) is not None or self.list_revisions(policy_id):
                continue
            payload = {key: value for key, value in default.items() if key != "id"}
            self._create_with_id(policy_id, payload, actor="seed")
            seeded += 1
        return seeded

    def _revision(self, session, row, change, actor):
        key = f"{row.id}:{row.version}"
        if session.get(ToolApprovalPolicyRevision, key) is None:
            session.add(
                ToolApprovalPolicyRevision(
                    id=key,
                    policy_id=row.id,
                    version=row.version,
                    change=change,
                    actor=actor,
                    snapshot_json=json.dumps(row.to_api(), sort_keys=True),
                )
            )

    def list_revisions(self, policy_id):
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval policy database unavailable")
        with Session(self.engine) as session:
            return list(
                session.exec(
                    select(ToolApprovalPolicyRevision)
                    .where(ToolApprovalPolicyRevision.policy_id == policy_id)
                    .order_by(ToolApprovalPolicyRevision.version.desc())
                ).all()
            )

    def page_revisions(self, policy_id, *, limit=50, offset=0):
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval policy database unavailable")
        limit, offset = min(max(limit, 1), 200), max(offset, 0)
        predicate = ToolApprovalPolicyRevision.policy_id == policy_id
        with Session(self.engine) as session:
            total = session.exec(
                select(func.count())
                .select_from(ToolApprovalPolicyRevision)
                .where(predicate)
            ).one()
            rows = list(
                session.exec(
                    select(ToolApprovalPolicyRevision)
                    .where(predicate)
                    .order_by(ToolApprovalPolicyRevision.version.desc())
                    .offset(offset)
                    .limit(limit)
                ).all()
            )
            return rows, total

    def record_bridge_decision(
        self, event, *, bot_id=None, user_id=None, session_id=None, turn_id=None
    ):
        """Commit a redacted run event, idempotent across replay of that event.

        No current-bundle lookup: record the version actually evaluated, whose
        immutable snapshot is available from the policy revision endpoint.
        Errors propagate to the app consumer's bounded best-effort retry loop.
        """
        from hashlib import sha256
        from agent_bridge.approval_audit import audit_subject

        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval policy database unavailable")
        meta = event.raw
        if meta.get("action") not in ("allow", "deny", "require_approval"):
            raise ValueError("Invalid bridge audit action")
        key = sha256(
            f"bridge-decision:{event.run_id}:{event.event_id}".encode()
        ).hexdigest()
        with Session(self.engine) as session:
            if session.get(ToolApprovalDecision, key) is not None:
                return
            session.add(
                ToolApprovalDecision(
                    id=key,
                    created_at=event.timestamp,
                    backend=event.provider or "claude-code",
                    tool_name=event.tool_name or "",
                    action=meta["action"],
                    outcome=meta.get("outcome"),
                    severity=meta.get("severity") or "medium",
                    subject=audit_subject(meta.get("subject") or ""),
                    policy_id=meta.get("policy_id"),
                    policy_version=meta.get("policy_version"),
                    bundle_etag=meta.get("bundle_etag") or "",
                    invocation_hash=meta.get("invocation_hash") or "",
                    source="bridge",
                    bot_id=bot_id,
                    user_id=user_id,
                    turn_id=turn_id,
                    session_key=event.session_key,
                    session_id=session_id,
                    request_id=event.run_id,
                    tool_use_id=event.tool_use_id,
                )
            )
            session.commit()

    def record_decision(
        self,
        *,
        decision,
        bundle,
        backend,
        tool_name,
        invocation_hash,
        bot_id=None,
        user_id=None,
        turn_id=None,
        source="mcp",
    ):
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval policy database unavailable")
        from agent_bridge.approval_audit import audit_subject

        policy = decision.policy
        row = ToolApprovalDecision(
            backend=backend,
            tool_name=tool_name,
            action=decision.action.value,
            severity=decision.severity.value,
            subject=audit_subject(decision.subject),
            policy_id=policy.id if policy else None,
            policy_version=policy.version if policy else None,
            policy_snapshot_json=json.dumps(policy.to_dict(), sort_keys=True)
            if policy
            else None,
            bundle_etag=bundle.etag,
            invocation_hash=invocation_hash,
            bot_id=bot_id,
            user_id=user_id,
            turn_id=turn_id,
            source=source,
        )
        with Session(self.engine) as session:
            session.add(row)
            session.commit()

    def list_decisions(self, *, limit=50, offset=0):
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval policy database unavailable")
        limit, offset = min(max(limit, 1), 200), max(offset, 0)
        with Session(self.engine) as session:
            total = session.exec(
                select(func.count()).select_from(ToolApprovalDecision)
            ).one()
            rows = list(
                session.exec(
                    select(ToolApprovalDecision)
                    .order_by(
                        ToolApprovalDecision.created_at.desc(), ToolApprovalDecision.id
                    )
                    .offset(offset)
                    .limit(limit)
                ).all()
            )
            return rows, total
