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

from ..utils.schema import SchemaBootstrapGuard, ensure_relations
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

_ITEM_COLUMNS = (
    "id, kind, initiative_task_id, objective, status, next_action, blockers, checkpoint, priority, refs,"
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


def item_dict(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "kind": row["kind"],
        "initiative_task_id": row["initiative_task_id"],
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


class WorkingMemoryStore:
    """Items, the content-free change log and the per-namespace version."""

    def __init__(self, engine) -> None:
        if engine is None or getattr(getattr(engine, "dialect", None), "name", None) != "postgresql":
            raise WorkingMemoryError("unavailable", "working memory unavailable")
        self.engine = engine

    def ensure_schema(self) -> None:
        _schema_guard.run(self.engine, "working-memory", lambda conn: ensure_relations(conn, _RELATIONS))

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

    def create(self, bot_id: str, user_id: str, fields: Mapping[str, Any], *, idempotency_key: Any,
               actor: Actor, authorize: Authorize | None = None) -> dict[str, Any]:
        """Idempotent create keyed by a UUID and the immutable create fingerprint (6.1)."""
        bot, user = normalize_identity(bot_id, user_id)
        key = normalize_idempotency_key(idempotency_key)
        payload = normalize_create(fields)
        fingerprint = create_fingerprint(payload)
        with self._write(bot, user, authorize) as write:
            prior = write.conn.execute(text(
                "SELECT item_id, create_fingerprint FROM working_memory_changes"
                " WHERE bot_id = :bot AND user_id = :user AND kind = 'create'"
                " AND idempotency_key = CAST(:key AS uuid)"
            ), {"bot": bot, "user": user, "key": key}).mappings().first()
            if prior is not None:
                return self._replay(write, prior, fingerprint)
            if write.open_count() + 1 > MAX_OPEN_ITEMS:
                raise WorkingMemoryError(
                    "cap_exceeded", f"at most {MAX_OPEN_ITEMS} open items", cap="open_items", limit=MAX_OPEN_ITEMS,
                )
            item_id = str(uuid.uuid4())
            version = write.bump()
            row = write.conn.execute(text(f"""
                INSERT INTO working_memory_items
                    (id, bot_id, user_id, kind, initiative_task_id, objective, status, next_action,
                     priority, refs, revision)
                VALUES (:id, :bot, :user, :kind, :task, :objective, :status, :next_action,
                        :priority, CAST(:refs AS jsonb), 1)
                RETURNING {_ITEM_COLUMNS}
            """), {
                "id": item_id, "bot": bot, "user": user, "kind": payload["kind"],
                "task": payload["initiative_task_id"], "objective": payload["objective"],
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
                write.conn.execute(text(
                    "UPDATE working_memory_items SET removed_at = now() WHERE id = :id"
                ), {"id": item})
                write.change(actor, item, row["revision"], "removed")
                return {"item_id": item, "state": "removed", "version": version, "changed": True}
            version = write.bump()
            write.conn.execute(text("DELETE FROM working_memory_items WHERE id = :id"), {"id": item})
            write.conn.execute(text(
                "UPDATE working_memory_changes SET create_fingerprint = NULL"
                " WHERE item_id = :id AND kind = 'create'"
            ), {"id": item})
            write.change(actor, item, row["revision"], "purged")
            return {"item_id": item, "state": "purged", "version": version, "changed": True}


__all__ = ["Actor", "WorkingMemoryStore", "item_dict"]
