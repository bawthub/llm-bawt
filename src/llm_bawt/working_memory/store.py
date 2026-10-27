"""PostgreSQL store for working memory (design sections 5, 6, 11.3, 12, 13).

Every write runs in one transaction that first locks the namespace row, then
decides (capability, CAS, idempotency, caps) under that lock, and only a real
write bumps the namespace version. Anything that is not a real write rolls the
transaction back, so a refused, conflicting, replayed or no-op request leaves
no row behind, not even the namespace bootstrap row (5.3).

Lock order: namespace row -> item row -> change insert (13). No session or turn
row is ever locked here.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import SQLAlchemyError

from ..service.providers.crypto import derive_key
from ..utils.schema import SchemaBootstrapGuard, ensure_columns, ensure_relations
from .task_links import TaskInfo
from .contracts import (
    ARCHIVED_STATUSES,
    MAX_OPEN_ITEMS,
    OPEN_STATUSES,
    WorkingMemoryError,
    create_fingerprint,
    normalize_changes,
    normalize_create,
    normalize_idempotency_key,
    normalize_identity,
    normalize_item_id,
)

logger = logging.getLogger(__name__)

ActorKind = Literal["bot", "user"]
Authorize = Callable[[Connection], Any]
"""A policy gate run on the store's connection; it refuses by raising."""

_RELATIONS = {
    "working_memory_namespaces": """
        CREATE TABLE IF NOT EXISTS working_memory_namespaces (
            bot_id VARCHAR(128) NOT NULL,
            user_id VARCHAR(128) NOT NULL,
            version BIGINT NOT NULL DEFAULT 0,
            PRIMARY KEY (bot_id, user_id)
        )
    """,
    "working_memory_items": """
        CREATE TABLE IF NOT EXISTS working_memory_items (
            id UUID PRIMARY KEY,
            bot_id VARCHAR(128) NOT NULL,
            user_id VARCHAR(128) NOT NULL,
            kind VARCHAR(32) NOT NULL CHECK (kind IN ('personal', 'initiative_link')),
            initiative_task_id VARCHAR(128),
            objective TEXT NOT NULL,
            status VARCHAR(16) NOT NULL
                CHECK (status IN ('planned', 'active', 'blocked', 'done', 'cancelled')),
            next_action TEXT,
            blockers JSONB NOT NULL DEFAULT '[]'::jsonb,
            checkpoint TEXT,
            priority INTEGER,
            refs JSONB NOT NULL DEFAULT '[]'::jsonb,
            revision INTEGER NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            archived_at TIMESTAMPTZ,
            removed_at TIMESTAMPTZ,
            CHECK ((kind = 'initiative_link') = (initiative_task_id IS NOT NULL)),
            CHECK ((status IN ('done', 'cancelled')) = (archived_at IS NOT NULL))
        )
    """,
    "ix_working_memory_items_namespace": """
        CREATE INDEX IF NOT EXISTS ix_working_memory_items_namespace
        ON working_memory_items (bot_id, user_id)
    """,
    "working_memory_task_suppressions": """
        CREATE TABLE IF NOT EXISTS working_memory_task_suppressions (
            bot_id VARCHAR(128) NOT NULL,
            user_id VARCHAR(128) NOT NULL,
            task_digest CHAR(64) NOT NULL,
            key_version SMALLINT NOT NULL DEFAULT 1,
            key_check CHAR(64),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (bot_id, user_id, key_version, task_digest)
        )
    """,
    "working_memory_task_purge_audit": """
        CREATE TABLE IF NOT EXISTS working_memory_task_purge_audit (
            item_id UUID PRIMARY KEY,
            task_digest CHAR(64),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """,
    "working_memory_changes": """
        CREATE TABLE IF NOT EXISTS working_memory_changes (
            seq BIGSERIAL PRIMARY KEY,
            bot_id VARCHAR(128) NOT NULL,
            user_id VARCHAR(128) NOT NULL,
            ns_version BIGINT NOT NULL,
            item_id UUID NOT NULL,
            revision INTEGER NOT NULL,
            kind VARCHAR(16) NOT NULL
                CHECK (kind IN ('create', 'update', 'archive', 'removed', 'purged')),
            actor_kind VARCHAR(8) NOT NULL CHECK (actor_kind IN ('bot', 'user')),
            actor_id VARCHAR(128) NOT NULL,
            session_id VARCHAR(64),
            turn_id VARCHAR(128),
            idempotency_key UUID,
            create_fingerprint CHAR(64),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CHECK (kind = 'create' OR (idempotency_key IS NULL AND create_fingerprint IS NULL))
        )
    """,
    "ix_working_memory_changes_ns_version": """
        CREATE INDEX IF NOT EXISTS ix_working_memory_changes_ns_version
        ON working_memory_changes (bot_id, user_id, ns_version)
    """,
    "ix_working_memory_changes_item": """
        CREATE INDEX IF NOT EXISTS ix_working_memory_changes_item
        ON working_memory_changes (item_id)
    """,
    "ux_working_memory_changes_create_key": """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_working_memory_changes_create_key
        ON working_memory_changes (bot_id, user_id, idempotency_key)
        WHERE kind = 'create'
    """,
}

_TASK_COLUMNS = {
    "canonical_task_id": ("VARCHAR(128)", None),
    "task_short_id": ("VARCHAR(64)", None),
    "task_title": ("VARCHAR(160)", None),
    "task_status": ("VARCHAR(32)", None),
}
_SUPPRESSION_COLUMNS = {"key_check": ("CHAR(64)", None)}
_TASK_RELATIONS = {
    "ux_working_memory_live_task_links": """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_working_memory_live_task_links
        ON working_memory_items (bot_id, user_id, canonical_task_id)
        WHERE kind = 'initiative_link' AND canonical_task_id IS NOT NULL
          AND removed_at IS NULL AND status IN ('planned', 'active', 'blocked')
    """,
}

_ITEM_COLUMNS = (
    "id, kind, initiative_task_id, canonical_task_id, task_short_id, task_title, task_status,"
    " objective, status, next_action, blockers, checkpoint, priority, refs,"
    " revision, created_at, updated_at, archived_at, removed_at"
)
# Read order (8.1 / L9 OI-7): status group, then priority (NULL last), newest update, id.
_ORDER = (
    " ORDER BY CASE status WHEN 'active' THEN 0 WHEN 'blocked' THEN 1 WHEN 'planned' THEN 2"
    " WHEN 'done' THEN 3 ELSE 4 END, priority ASC NULLS LAST, updated_at DESC, id"
)
# One statement is one snapshot, so the two values agree at any isolation level.
_EXTERNAL_CHANGES = text(
    "SELECT (SELECT max(ns_version) FROM working_memory_changes"
    " WHERE bot_id = :bot AND user_id = :user AND ns_version > :after"
    " AND session_id IS DISTINCT FROM :session),"
    " (SELECT version FROM working_memory_namespaces WHERE bot_id = :bot AND user_id = :user)"
)

_schema_guard = SchemaBootstrapGuard()


@contextmanager
def _storage_errors():
    """A database failure is ``unavailable``, never an empty result (6.1, 14)."""
    try:
        yield
    except SQLAlchemyError as exc:
        logger.warning("working memory storage failure: %s", exc)
        raise WorkingMemoryError("unavailable", "working memory unavailable") from exc


@dataclass(frozen=True)
class Actor:
    """Who wrote: a bot through its trusted turn capability, or the owner over REST."""

    kind: ActorKind
    actor_id: str
    session_id: str | None = None
    turn_id: str | None = None


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _suppression_key() -> bytes:
    return derive_key("working-memory-task-suppression/v1")


def _key_check(key: bytes) -> str:
    """Opaque sentinel so rotating the persisted master key fails closed."""
    return hmac.new(key, b"working-memory-task-suppression/key-check/v1", hashlib.sha256).hexdigest()


def _task_digest(bot: str, user: str, task_id: str) -> str:
    """Opaque stable suppression key; never store item text or a task reference."""
    data = json.dumps([bot, user, task_id], separators=(",", ":"), ensure_ascii=False)
    return hmac.new(_suppression_key(), data.encode("utf-8"), hashlib.sha256).hexdigest()


def item_dict(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "kind": row["kind"],
        "initiative_task_id": row["initiative_task_id"],
        "canonical_task_id": row["canonical_task_id"],
        "task_short_id": row["task_short_id"],
        "task_title": row["task_title"],
        "task_status": row["task_status"],
        "objective": row["objective"],
        "status": row["status"],
        "next_action": row["next_action"],
        "blockers": _json(row["blockers"]) or [],
        "checkpoint": row["checkpoint"],
        "priority": row["priority"],
        "refs": _json(row["refs"]) or [],
        "revision": row["revision"],
        "created_at": _iso(row["created_at"]),
        "updated_at": _iso(row["updated_at"]),
        "archived_at": _iso(row["archived_at"]),
        "removed_at": _iso(row["removed_at"]),
    }


class _Write:
    """One namespace-locked write transaction; commits only after a real write."""

    def __init__(self, conn: Connection, bot: str, user: str) -> None:
        self.conn = conn
        self.bot = bot
        self.user = user
        self.version: int | None = None

    def bump(self) -> int:
        """The real-write marker: the returned value is this write's ``ns_version``."""
        self.version = self.conn.execute(text(
            "UPDATE working_memory_namespaces SET version = version + 1"
            " WHERE bot_id = :bot AND user_id = :user RETURNING version"
        ), {"bot": self.bot, "user": self.user}).scalar_one()
        return self.version

    def change(self, actor: Actor, item_id: str, revision: int, kind: str, *,
               key: str | None = None, fingerprint: str | None = None) -> None:
        self.conn.execute(text("""
            INSERT INTO working_memory_changes
                (bot_id, user_id, ns_version, item_id, revision, kind, actor_kind, actor_id,
                 session_id, turn_id, idempotency_key, create_fingerprint)
            VALUES (:bot, :user, :version, :item, :revision, :kind, :actor_kind, :actor_id,
                    :session, :turn, CAST(:key AS uuid), :fingerprint)
        """), {
            "bot": self.bot, "user": self.user, "version": self.version, "item": item_id,
            "revision": revision, "kind": kind, "actor_kind": actor.kind, "actor_id": actor.actor_id,
            "session": actor.session_id, "turn": actor.turn_id, "key": key, "fingerprint": fingerprint,
        })

    def open_count(self) -> int:
        return self.conn.execute(text(
            "SELECT count(*) FROM working_memory_items"
            " WHERE bot_id = :bot AND user_id = :user AND removed_at IS NULL"
            " AND status IN ('planned', 'active', 'blocked')"
        ), {"bot": self.bot, "user": self.user}).scalar_one()

    def lock_item(self, item_id: str) -> Mapping[str, Any] | None:
        return self.conn.execute(text(
            f"SELECT {_ITEM_COLUMNS} FROM working_memory_items"
            " WHERE id = :id AND bot_id = :bot AND user_id = :user FOR UPDATE"
        ), {"id": item_id, "bot": self.bot, "user": self.user}).mappings().first()

    def _check_suppression_key(self) -> str:
        try:
            expected = _key_check(_suppression_key())
        except Exception as exc:
            raise WorkingMemoryError("unavailable", "task suppression key unavailable") from exc
        checks = self.conn.execute(text(
            "SELECT DISTINCT key_version, key_check FROM working_memory_task_suppressions"
            " WHERE bot_id=:bot AND user_id=:user"
        ), {"bot": self.bot, "user": self.user}).all()
        if any(version != 1 or check is None or not hmac.compare_digest(check, expected)
               for version, check in checks):
            raise WorkingMemoryError("task_link_audit_pending", "task suppression key changed; audit required")
        return expected

    def suppressed(self, digest: str) -> bool:
        self._check_suppression_key()
        return self.conn.execute(text(
            "SELECT 1 FROM working_memory_task_suppressions"
            " WHERE bot_id=:bot AND user_id=:user AND key_version=1 AND task_digest=:digest"
        ), {"bot": self.bot, "user": self.user, "digest": digest}).first() is not None

    def suppress(self, digest: str) -> None:
        check = self._check_suppression_key()
        self.conn.execute(text(
            "INSERT INTO working_memory_task_suppressions (bot_id,user_id,task_digest,key_check)"
            " VALUES (:bot,:user,:digest,:check) ON CONFLICT DO NOTHING"
        ), {"bot": self.bot, "user": self.user, "digest": digest, "check": check})


class WorkingMemoryStore:
    """Items, the content-free change log and the per-namespace version."""

    def __init__(self, engine) -> None:
        if engine is None or getattr(getattr(engine, "dialect", None), "name", None) != "postgresql":
            raise WorkingMemoryError("unavailable", "working memory unavailable")
        self.engine = engine

    def ensure_schema(self) -> None:
        def bootstrap(conn: Connection) -> None:
            ensure_relations(conn, _RELATIONS)
            ensure_columns(conn, "working_memory_items", _TASK_COLUMNS)
            ensure_columns(conn, "working_memory_task_suppressions", _SUPPRESSION_COLUMNS)
            ensure_relations(conn, _TASK_RELATIONS)

        _schema_guard.run(self.engine, "working-memory", bootstrap)

    # -- reads --------------------------------------------------------------

    def list_items(self, bot_id: str, user_id: str, *, include_archived: bool = False,
                   include_removed: bool = False, authorize: Authorize | None = None) -> dict[str, Any]:
        """Version and items from one REPEATABLE READ snapshot, so they agree (8.1, 13)."""
        bot, user = normalize_identity(bot_id, user_id)
        where = ["bot_id = :bot", "user_id = :user"]
        if not include_removed:
            where.append("removed_at IS NULL")
        if not include_archived:
            where.append("status IN ('planned', 'active', 'blocked')")
        with _storage_errors():
            self.ensure_schema()
            with self.engine.connect() as conn:
                conn = conn.execution_options(isolation_level="REPEATABLE READ")
                with conn.begin():
                    conn.execute(text("SET TRANSACTION READ ONLY"))
                    if authorize is not None:
                        authorize(conn)
                    return self._snapshot(conn, bot, user, where, include_removed)

    def render_snapshot(self, bot_id: str, user_id: str) -> dict[str, Any]:
        """Open items for the prompt section, from one REPEATABLE READ snapshot (8.1).

        Like :meth:`list_items` for open items, plus each item's
        ``last_session_id``: the session of its latest change, None for an
        owner edit. Read-only; takes no lock.
        """
        bot, user = normalize_identity(bot_id, user_id)
        where = ["bot_id = :bot", "user_id = :user", "removed_at IS NULL",
                 "status IN ('planned', 'active', 'blocked')"]
        with _storage_errors():
            self.ensure_schema()
            with self.engine.connect() as conn:
                conn = conn.execution_options(isolation_level="REPEATABLE READ")
                with conn.begin():
                    conn.execute(text("SET TRANSACTION READ ONLY"))
                    out = self._snapshot(conn, bot, user, where, False)
                    ids = [item["id"] for item in out["items"]]
                    last = {} if not ids else {
                        str(r["item_id"]): r["session_id"] for r in conn.execute(text(
                            "SELECT DISTINCT ON (item_id) item_id, session_id FROM working_memory_changes"
                            " WHERE bot_id = :bot AND user_id = :user AND item_id = ANY(CAST(:ids AS uuid[]))"
                            " ORDER BY item_id, seq DESC"
                        ), {"bot": bot, "user": user, "ids": ids}).mappings()
                    }
        for item in out["items"]:
            item["last_session_id"] = last.get(item["id"])
        return out

    def external_changes(self, bot_id: str, user_id: str, *, after: int, session_id: str,
                         conn: Connection | None = None) -> tuple[int | None, int]:
        """``(highest external ns_version above after, namespace version)`` from one snapshot (9.2).

        External means not written in ``session_id``: another session of the
        bot, or the owner over REST. One statement, so both values come from
        one snapshot; it takes no lock. With ``conn`` it runs on the caller's
        connection and transaction (the notice mark, which already holds its
        session row and must not check out a second pooled connection);
        without it, on a connection of its own.
        """
        bot, user = normalize_identity(bot_id, user_id)
        params = {"bot": bot, "user": user, "after": int(after), "session": session_id}
        with _storage_errors():
            if conn is not None:
                row = conn.execute(_EXTERNAL_CHANGES, params).one()
            else:
                self.ensure_schema()
                with self.engine.connect() as own:
                    row = own.execute(_EXTERNAL_CHANGES, params).one()
        external, version = row
        return (None if external is None else int(external)), int(version or 0)

    @staticmethod
    def _snapshot(conn: Connection, bot: str, user: str, where: list[str], include_removed: bool) -> dict[str, Any]:
        version = conn.execute(text(
            "SELECT version FROM working_memory_namespaces WHERE bot_id = :bot AND user_id = :user"
        ), {"bot": bot, "user": user}).scalar()
        rows = conn.execute(text(
            f"SELECT {_ITEM_COLUMNS} FROM working_memory_items WHERE {' AND '.join(where)}{_ORDER}"
        ), {"bot": bot, "user": user}).mappings().all()
        out: dict[str, Any] = {"version": int(version or 0), "items": [item_dict(r) for r in rows]}
        if include_removed:
            # Hard-deleted items, content-free: only the id and when (6.2, 11.3).
            purged = conn.execute(text(
                "SELECT item_id, created_at FROM working_memory_changes"
                " WHERE bot_id = :bot AND user_id = :user AND kind = 'purged' ORDER BY seq"
            ), {"bot": bot, "user": user}).mappings().all()
            out["purged"] = [{"id": str(r["item_id"]), "purged_at": _iso(r["created_at"])} for r in purged]
        return out

    # -- writes -------------------------------------------------------------

    @contextmanager
    def _write(self, bot: str, user: str, authorize: Authorize | None):
        """Lock, decide, then bump (5.3). Commit only when ``bump`` ran.

        ``authorize`` is the caller's policy gate. It runs under the namespace
        lock on the same connection, so each write gets a fresh policy check.
        That's not a drain barrier: a write already authorized when the
        setting is turned off can still commit.
        """
        with _storage_errors():
            self.ensure_schema()
            with self.engine.connect() as conn:
                transaction = conn.begin()
                try:
                    conn.execute(text(
                        "INSERT INTO working_memory_namespaces (bot_id, user_id) VALUES (:bot, :user)"
                        " ON CONFLICT (bot_id, user_id) DO NOTHING"
                    ), {"bot": bot, "user": user})
                    conn.execute(text(
                        "SELECT version FROM working_memory_namespaces"
                        " WHERE bot_id = :bot AND user_id = :user FOR UPDATE"
                    ), {"bot": bot, "user": user})
                    if authorize is not None:
                        authorize(conn)
                    write = _Write(conn, bot, user)
                    yield write
                except BaseException:
                    transaction.rollback()
                    raise
                if write.version is None:
                    transaction.rollback()
                else:
                    transaction.commit()

    def replay_legacy_link(self, bot_id: str, user_id: str, fields: Mapping[str, Any], *,
                           idempotency_key: Any, authorize: Authorize) -> dict[str, Any] | None:
        """Preserve an old TASK-N create fingerprint before new creates canonicalize."""
        bot, user = normalize_identity(bot_id, user_id)
        key = normalize_idempotency_key(idempotency_key)
        fingerprint = create_fingerprint(normalize_create(fields))
        with self._write(bot, user, authorize) as write:
            prior = write.conn.execute(text(
                "SELECT item_id, create_fingerprint FROM working_memory_changes"
                " WHERE bot_id=:bot AND user_id=:user AND kind='create' AND idempotency_key=CAST(:key AS uuid)"
            ), {"bot": bot, "user": user, "key": key}).mappings().first()
            if prior is None or prior["create_fingerprint"] != fingerprint:
                return None
            return self._replay(write, prior, fingerprint)

    def prepare_links(self, bot_id: str, user_id: str, resolve: Callable[[str], TaskInfo], *,
                      authorize: Authorize) -> None:
        """Resolve legacy identifiers before locking; audit and backfill under one lock.

        No link is allowed until the index exists, every retained link has a
        canonical ID, and every known removal has a suppression marker. Unknown
        historical purges are unidentifiable: refuse that namespace, not a guess.
        """
        bot, user = normalize_identity(bot_id, user_id)
        with _storage_errors():
            self.ensure_schema()  # index creation must succeed before the gate opens
            with self.engine.connect() as conn:
                authorize(conn)
                if conn.execute(text("SELECT to_regclass('ux_working_memory_live_task_links')")).scalar() is None:
                    raise WorkingMemoryError("task_link_audit_pending", "live task-link index is missing")
                original = conn.execute(text(
                    "SELECT id, initiative_task_id, canonical_task_id, status, removed_at, revision"
                    " FROM working_memory_items WHERE bot_id=:bot AND user_id=:user"
                    " AND kind='initiative_link' ORDER BY created_at,id LIMIT 257"
                ), {"bot": bot, "user": user}).mappings().all()
                unknown_purge = conn.execute(text(
                    "SELECT 1 FROM working_memory_changes c LEFT JOIN working_memory_task_purge_audit a"
                    " ON a.item_id=c.item_id WHERE c.bot_id=:bot AND c.user_id=:user"
                    " AND c.kind='purged' AND a.item_id IS NULL LIMIT 1"
                ), {"bot": bot, "user": user}).first()
        if len(original) > 256 or unknown_purge is not None:
            raise WorkingMemoryError("task_link_audit_pending", "historical working-memory links need owner audit")
        legacy = {r["initiative_task_id"] for r in original if not r["canonical_task_id"]}
        resolved = {ref: resolve(ref) for ref in legacy}
        digests = {info.id: _task_digest(bot, user, info.id) for info in resolved.values()}
        with self._write(bot, user, authorize) as write:
            current = write.conn.execute(text(
                "SELECT id, initiative_task_id, canonical_task_id, status, removed_at, revision"
                " FROM working_memory_items WHERE bot_id=:bot AND user_id=:user"
                " AND kind='initiative_link' ORDER BY created_at,id LIMIT 257"
            ), {"bot": bot, "user": user}).mappings().all()
            if [tuple(r.values()) for r in current] != [tuple(r.values()) for r in original]:
                # A concurrent canonical create/backfill already passed this same
                # namespace audit; do not reject a legitimate racing association.
                # New legacy rows still need fresh resolution off this lock.
                if len(current) > 256 or any(not r["canonical_task_id"] for r in current):
                    raise WorkingMemoryError("task_link_audit_pending", "links changed during audit; retry")
            # Also check the previously empty namespace: a concurrent purge is
            # serialized by this lock and must never be missed by the gate.
            if write.conn.execute(text(
                "SELECT 1 FROM working_memory_changes c LEFT JOIN working_memory_task_purge_audit a"
                " ON a.item_id=c.item_id WHERE c.bot_id=:bot AND c.user_id=:user"
                " AND c.kind='purged' AND a.item_id IS NULL LIMIT 1"
            ), {"bot": bot, "user": user}).first():
                raise WorkingMemoryError("task_link_audit_pending", "historical purge needs owner audit")
            groups: dict[str, list[Mapping[str, Any]]] = {}
            for row in current:
                task_id = row["canonical_task_id"] or resolved[row["initiative_task_id"]].id
                groups.setdefault(task_id, []).append(row)
            actor = Actor("bot", bot)  # migration is external to every pinned session
            for task_id, rows in groups.items():
                if any(r["removed_at"] is not None for r in rows):
                    write.suppress(digests.get(task_id) or _task_digest(bot, user, task_id))
                live = [r for r in rows if r["removed_at"] is None and r["status"] in OPEN_STATUSES]
                # Prefer an owner-authored item, then the oldest. Migration does
                # not discard the other item's content or its original create key.
                owner_items = {r["item_id"] for r in write.conn.execute(text(
                    "SELECT DISTINCT item_id FROM working_memory_changes"
                    " WHERE bot_id=:bot AND user_id=:user AND actor_kind='user'"
                    " AND item_id = ANY(CAST(:ids AS uuid[]))"
                ), {"bot": bot, "user": user, "ids": [str(r["id"]) for r in live]}).mappings()} if live else set()
                keeper = next((r for r in live if r["id"] in owner_items), live[0] if live else None)
                losers = live if any(r["removed_at"] is not None for r in rows) else [
                    r for r in live if r is not keeper
                ]
                archived_ids = {r["id"] for r in losers}
                for loser in losers:
                    write.bump()
                    write.conn.execute(text(
                        "UPDATE working_memory_items SET status='cancelled', archived_at=now(),"
                        " revision=revision+1, updated_at=now() WHERE id=:id"
                    ), {"id": loser["id"]})
                    write.change(actor, str(loser["id"]), loser["revision"] + 1, "archive")
                for row in rows:
                    if row["canonical_task_id"] is not None:
                        continue
                    info = resolved[row["initiative_task_id"]]
                    write.bump()
                    write.conn.execute(text(
                        "UPDATE working_memory_items SET canonical_task_id=:task,"
                        " initiative_task_id=:task, task_short_id=:short, task_title=:title,"
                        " task_status=:status, revision=revision+1, updated_at=now() WHERE id=:id"
                    ), {"id": row["id"], "task": info.id, "short": info.short_id,
                        "title": info.title[:160], "status": info.status})
                    write.change(actor, str(row["id"]), row["revision"] + 1 + int(row["id"] in archived_ids), "update")

    def create(self, bot_id: str, user_id: str, fields: Mapping[str, Any], *, idempotency_key: Any,
               actor: Actor, authorize: Authorize | None = None, task: TaskInfo | None = None,
               auto: bool = False) -> dict[str, Any]:
        """Idempotent create keyed by a UUID and the immutable create fingerprint (6.1)."""
        bot, user = normalize_identity(bot_id, user_id)
        key = normalize_idempotency_key(idempotency_key)
        payload = normalize_create(fields)
        if task is not None:
            if payload["kind"] != "initiative_link" or payload["initiative_task_id"] != task.id:
                raise WorkingMemoryError("invalid_request", "link task identity must be canonical")
            try:
                digest = _task_digest(bot, user, task.id)  # fail before a namespace write
            except Exception as exc:
                raise WorkingMemoryError("unavailable", "task suppression key unavailable") from exc
        else:
            digest = None
        fingerprint = create_fingerprint(payload)
        with self._write(bot, user, authorize) as write:
            prior = write.conn.execute(text(
                "SELECT item_id, create_fingerprint FROM working_memory_changes"
                " WHERE bot_id = :bot AND user_id = :user AND kind = 'create'"
                " AND idempotency_key = CAST(:key AS uuid)"
            ), {"bot": bot, "user": user, "key": key}).mappings().first()
            if prior is not None:
                return self._replay(write, prior, fingerprint)
            if digest is not None:
                if write.suppressed(digest):
                    raise WorkingMemoryError("removed", "this task link was removed")
                existing = write.conn.execute(text(
                    f"SELECT {_ITEM_COLUMNS} FROM working_memory_items"
                    " WHERE bot_id=:bot AND user_id=:user AND canonical_task_id=:task"
                    " AND kind='initiative_link' AND removed_at IS NULL"
                    " ORDER BY CASE WHEN status IN ('planned','active','blocked')"
                    " THEN 0 ELSE 1 END, created_at,id LIMIT 1"
                ), {"bot": bot, "user": user, "task": task.id}).mappings().first()
                if existing is not None:
                    if not auto or existing["status"] in ARCHIVED_STATUSES:
                        raise WorkingMemoryError("already_linked", "this task already has a working-memory link",
                                                 item_id=str(existing["id"]))
                    return {"item": item_dict(existing), "version": None, "replayed": True}
            if write.open_count() + 1 > MAX_OPEN_ITEMS:
                raise WorkingMemoryError(
                    "cap_exceeded", f"at most {MAX_OPEN_ITEMS} open items", cap="open_items", limit=MAX_OPEN_ITEMS,
                )
            item_id = str(uuid.uuid4())
            version = write.bump()
            row = write.conn.execute(text(f"""
                INSERT INTO working_memory_items
                    (id, bot_id, user_id, kind, initiative_task_id, canonical_task_id,
                     task_short_id, task_title, task_status, objective, status, next_action,
                     priority, refs, revision)
                VALUES (:id, :bot, :user, :kind, :task, :canonical,
                        :short, :title, :task_status, :objective, :status, :next_action,
                        :priority, CAST(:refs AS jsonb), 1)
                RETURNING {_ITEM_COLUMNS}
            """), {
                "id": item_id, "bot": bot, "user": user, "kind": payload["kind"],
                "task": payload["initiative_task_id"], "canonical": task.id if task else None,
                "short": task.short_id if task else None, "title": task.title[:160] if task else None,
                "task_status": task.status if task else None, "objective": payload["objective"],
                "status": payload["status"], "next_action": payload["next_action"],
                "priority": payload["priority"], "refs": json.dumps(payload["refs"]),
            }).mappings().one()
            write.change(actor, item_id, 1, "create", key=key, fingerprint=fingerprint)
            return {"item": item_dict(row), "version": version, "replayed": False}

    @staticmethod
    def _replay(write: _Write, prior: Mapping[str, Any], fingerprint: str) -> dict[str, Any]:
        """Decide a retried key from the create row alone; never writes (6.1)."""
        item_id = str(prior["item_id"])
        if prior["create_fingerprint"] is None:
            raise WorkingMemoryError("removed", "that item was deleted", item_id=item_id)
        if prior["create_fingerprint"] != fingerprint:
            raise WorkingMemoryError(
                "idempotency_conflict", "idempotency_key was already used for a different create",
                item_id=item_id,
            )
        row = write.conn.execute(text(
            f"SELECT {_ITEM_COLUMNS} FROM working_memory_items WHERE id = :id"
        ), {"id": item_id}).mappings().first()
        if row is None or row["removed_at"] is not None:
            raise WorkingMemoryError("removed", "that item was deleted", item_id=item_id)
        return {"item": item_dict(row), "version": None, "replayed": True}

    def refresh_task(self, bot_id: str, user_id: str, item_id: str, info: TaskInfo, *,
                     expected_revision: int, authorize: Authorize) -> bool:
        """Apply a verified task snapshot without reopening/overwriting a user's item."""
        bot, user = normalize_identity(bot_id, user_id)
        with self._write(bot, user, authorize) as write:
            row = write.lock_item(item_id)
            if (row is None or row["kind"] != "initiative_link" or row["removed_at"] is not None
                    or row["status"] in ARCHIVED_STATUSES or row["revision"] != expected_revision
                    or row["canonical_task_id"] != info.id):
                return False
            status = ("done" if info.status == "COMPLETED" else "cancelled"
                      if info.status == "CANCELLED" or
                      (info.assignee is not None and info.assignee.lower() != bot) else row["status"])
            title = info.title[:160]
            if (row["status"], row["task_short_id"], row["task_title"], row["task_status"]) == (
                    status, info.short_id, title, info.status):
                return False
            write.bump()
            updated = write.conn.execute(text(
                "UPDATE working_memory_items SET status=:state,"
                " archived_at=CASE WHEN :state IN ('done','cancelled') THEN now() ELSE archived_at END,"
                " task_short_id=:short,task_title=:title,task_status=:task_status,"
                " revision=revision+1,updated_at=now() WHERE id=:id RETURNING revision"
            ), {"id": item_id, "state": status, "short": info.short_id, "title": title,
                "task_status": info.status}).scalar_one()
            write.change(Actor("bot", bot), item_id, updated,
                         "archive" if status in ARCHIVED_STATUSES else "update")
            return True

    def update(self, bot_id: str, user_id: str, item_id: Any, *, expected_revision: Any,
               changes: Mapping[str, Any], actor: Actor, authorize: Authorize | None = None) -> dict[str, Any]:
        """CAS update; ``done``/``cancelled`` archive, a reopen counts toward the cap."""
        bot, user = normalize_identity(bot_id, user_id)
        item = normalize_item_id(item_id)
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
            raise WorkingMemoryError("invalid_request", "expected_revision must be an integer")
        with self._write(bot, user, authorize) as write:
            row = write.lock_item(item)
            if row is None:
                raise WorkingMemoryError("not_found", "no such working-memory item")
            if row["removed_at"] is not None:
                raise WorkingMemoryError("removed", "that item was deleted", item_id=item)
            if row["revision"] != expected_revision:
                raise WorkingMemoryError(
                    "revision_conflict", "the item changed; re-read it and retry", item=item_dict(row),
                )
            normalized = normalize_changes(row["kind"], changes)
            current = item_dict(row)
            delta = {field: value for field, value in normalized.items() if current[field] != value}
            if not delta:
                return {"item": current, "version": None, "changed": False}
            status = delta.get("status", row["status"])
            reopening = row["status"] in ARCHIVED_STATUSES and status in OPEN_STATUSES
            if reopening and row["kind"] == "initiative_link":
                if actor.kind != "user" or not row["canonical_task_id"]:
                    raise WorkingMemoryError("already_linked", "archived task links require owner review")
                try:
                    digest = _task_digest(bot, user, row["canonical_task_id"])
                except Exception as exc:
                    raise WorkingMemoryError("unavailable", "task suppression key unavailable") from exc
                if write.suppressed(digest):
                    raise WorkingMemoryError("removed", "this task link was removed")
                other = write.conn.execute(text(
                    "SELECT id FROM working_memory_items WHERE bot_id=:bot AND user_id=:user"
                    " AND canonical_task_id=:task AND kind='initiative_link' AND id<>CAST(:id AS uuid)"
                    " AND removed_at IS NULL AND status IN ('planned','active','blocked') LIMIT 1"
                ), {"bot": bot, "user": user, "task": row["canonical_task_id"], "id": item}).first()
                if other is not None:
                    raise WorkingMemoryError("already_linked", "this task already has a live link",
                                             item_id=str(other[0]))
            if reopening and write.open_count() + 1 > MAX_OPEN_ITEMS:
                raise WorkingMemoryError(
                    "cap_exceeded", f"at most {MAX_OPEN_ITEMS} open items", cap="open_items", limit=MAX_OPEN_ITEMS,
                )
            version = write.bump()
            assignments = ["revision = revision + 1", "updated_at = now()"]
            params: dict[str, Any] = {"id": item}
            for field, value in delta.items():
                if field in ("blockers", "refs"):
                    assignments.append(f"{field} = CAST(:{field} AS jsonb)")
                    params[field] = json.dumps(value)
                else:
                    assignments.append(f"{field} = :{field}")
                    params[field] = value
            if "status" in delta:
                assignments.append(
                    "archived_at = CASE WHEN :status IN ('done', 'cancelled') THEN now() ELSE NULL END"
                )
            updated = write.conn.execute(text(
                f"UPDATE working_memory_items SET {', '.join(assignments)} WHERE id = :id"
                f" RETURNING {_ITEM_COLUMNS}"
            ), params).mappings().one()
            archived = status in ARCHIVED_STATUSES and row["status"] not in ARCHIVED_STATUSES
            write.change(actor, item, updated["revision"], "archive" if archived else "update")
            return {"item": item_dict(updated), "version": version, "changed": True}

    def delete(self, bot_id: str, user_id: str, item_id: Any, *, hard: bool, actor: Actor,
               authorize: Authorize | None = None) -> dict[str, Any]:
        """Soft delete (final) or hard delete (purge); repeats are idempotent (6.2, 11.3)."""
        bot, user = normalize_identity(bot_id, user_id)
        item = normalize_item_id(item_id)
        with self._write(bot, user, authorize) as write:
            row = write.lock_item(item)
            if row is not None and row["kind"] == "initiative_link":
                if not row["canonical_task_id"]:
                    raise WorkingMemoryError("task_link_audit_pending", "resolve this legacy task link before deletion")
                try:
                    digest = _task_digest(bot, user, row["canonical_task_id"])
                except Exception as exc:
                    raise WorkingMemoryError("unavailable", "task suppression key unavailable") from exc
            else:
                digest = None
            if row is None:
                purged = write.conn.execute(text(
                    "SELECT 1 FROM working_memory_changes WHERE item_id = :id AND bot_id = :bot"
                    " AND user_id = :user AND kind = 'purged'"
                ), {"id": item, "bot": bot, "user": user}).first()
                if purged is None:
                    raise WorkingMemoryError("not_found", "no such working-memory item")
                return {"item_id": item, "state": "purged", "version": None, "changed": False}
            if not hard:
                if row["removed_at"] is not None:
                    return {"item_id": item, "state": "removed", "version": None, "changed": False}
                version = write.bump()
                if digest is not None:
                    write.suppress(digest)
                write.conn.execute(text(
                    "UPDATE working_memory_items SET removed_at = now() WHERE id = :id"
                ), {"id": item})
                write.change(actor, item, row["revision"], "removed")
                return {"item_id": item, "state": "removed", "version": version, "changed": True}
            version = write.bump()
            if digest is not None:
                write.suppress(digest)
            write.conn.execute(text(
                "INSERT INTO working_memory_task_purge_audit (item_id,task_digest)"
                " VALUES (CAST(:item AS uuid),:digest) ON CONFLICT DO NOTHING"
            ), {"item": item, "digest": digest})
            write.conn.execute(text("DELETE FROM working_memory_items WHERE id = :id"), {"id": item})
            write.conn.execute(text(
                "UPDATE working_memory_changes SET create_fingerprint = NULL"
                " WHERE item_id = :id AND kind = 'create'"
            ), {"id": item})
            write.change(actor, item, row["revision"], "purged")
            return {"item_id": item, "state": "purged", "version": version, "changed": True}


__all__ = ["Actor", "WorkingMemoryStore", "item_dict"]
