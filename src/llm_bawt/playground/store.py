"""The ``playground_artifacts`` table: artifacts bots send the user (TASK-2190).

A row is the artifact's metadata plus its content: either inline text
(``inline_content``) or a blob in the media object store under
``playground/<id>``. Playground owns that blob outright. It is never keyed by
content hash, so deleting it can't touch a chat attachment holding the same
bytes.

Mail-style lifecycle (TASK-2195). ``opened_at`` is the read flag: opening
sets it and keeps the artifact; mark-unread clears it. ``archived_at`` hides
a row from the Inbox; ``trashed_at`` moves it to Trash, where it can be
restored. Only "delete forever" or the trash purge removes the content, which
sets ``expired_at`` and leaves a content-gone tombstone row. The blob key
stays set until the blob is really gone, so a failed delete is retried.

Revisions (TASK-2625). The sender can replace a live artifact's content in
place. ``revision`` counts versions (NULL on older rows means 1) and
``updated_at`` is when the last revision landed. Read and folder state are
kept, so a revision never re-notifies. A replaced blob goes to
``playground_retired_blobs`` and the sweep deletes it after a grace period.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    and_,
    func,
    insert,
    inspect,
    or_,
    select,
    text,
    update,
)
from sqlalchemy.engine import Engine

from ..media.assets import _encode_crockford
from ..service.commit_wake import notify_in_transaction
from ..utils.schema import SchemaBootstrapGuard

logger = logging.getLogger(__name__)

TABLE_NAME = "playground_artifacts"
BLOB_PREFIX = "playground/"
KINDS = ("image", "video", "audio", "pdf", "html", "markdown", "text", "file")

metadata = MetaData()
artifacts = Table(
    TABLE_NAME,
    metadata,
    Column("id", String(40), primary_key=True),
    Column("user_id", String(128), nullable=False),
    Column("sender_bot_id", String(64), nullable=False),
    Column("title", String(200), nullable=False),
    Column("note", String(300)),
    Column("kind", String(16), nullable=False),
    Column("mime", String(200), nullable=False),
    Column("filename", String(255)),
    Column("size_bytes", BigInteger, nullable=False),
    Column("blob_key", String(80)),
    Column("inline_content", Text),
    Column("view_token_hash", String(64)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("opened_at", DateTime(timezone=True)),
    Column("expired_at", DateTime(timezone=True)),
    Column("archived_at", DateTime(timezone=True)),
    Column("trashed_at", DateTime(timezone=True)),
    Column("revision", Integer),
    Column("updated_at", DateTime(timezone=True)),
    Index(f"idx_{TABLE_NAME}_user_created", "user_id", "created_at"),
    Index(f"idx_{TABLE_NAME}_opened_live", "opened_at"),
)

RETIRED_TABLE = "playground_retired_blobs"
#: Blobs a revision replaced (TASK-2625). Recorded in the revision's own
#: transaction, so no committed swap can lose track of the old bytes; the
#: sweep deletes each after a grace period, so a read that picked the old key
#: just before the swap still finishes.
retired_blobs = Table(
    RETIRED_TABLE,
    metadata,
    Column("blob_key", String(80), primary_key=True),
    Column("artifact_id", String(40), nullable=False),
    Column("retired_at", DateTime(timezone=True), nullable=False),
    Index(f"idx_{RETIRED_TABLE}_retired", "retired_at"),
)

#: Columns safe to return to any reader: never content or the token hash.
PUBLIC_COLUMNS = (
    "id", "user_id", "sender_bot_id", "title", "note", "kind", "mime", "filename",
    "size_bytes", "created_at", "opened_at", "expired_at", "archived_at", "trashed_at",
    "revision", "updated_at",
)

#: Inbox views. ``None`` (the default list) is the Inbox, so unread counts
#: built from the default list follow the Inbox.
VIEWS = ("inbox", "archive", "trash", "all")
_DATE_COLUMNS = ("created_at", "opened_at", "expired_at", "archived_at", "trashed_at", "updated_at")
#: Nullable columns added after the table first shipped: name -> SQL type.
_ADDED_COLUMNS = {
    "archived_at": "timestamp",
    "trashed_at": "timestamp",
    "revision": "INTEGER",
    "updated_at": "timestamp",
}


def new_artifact_id() -> str:
    """``pg_<ulid>``: sortable by creation time, like media asset ids."""
    ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    return "pg_" + _encode_crockford(ms, 10) + _encode_crockford(
        int.from_bytes(os.urandom(10), "big"), 16)


def blob_key_for(artifact_id: str) -> str:
    return f"{BLOB_PREFIX}{artifact_id}"


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def status_of(row: dict[str, Any]) -> str:
    """Read state: ``unopened`` (unread), ``opened`` (read) or ``expired`` (content gone)."""
    if row.get("expired_at") is not None:
        return "expired"
    return "opened" if row.get("opened_at") is not None else "unopened"


def folder_of(row: dict[str, Any]) -> str:
    if row.get("expired_at") is not None:
        return "deleted"
    if row.get("trashed_at") is not None:
        return "trash"
    return "archive" if row.get("archived_at") is not None else "inbox"


def public_view(row: dict[str, Any]) -> dict[str, Any]:
    """The metadata a list or History entry shows; no content."""
    out = {name: row.get(name) for name in PUBLIC_COLUMNS}
    for name in _DATE_COLUMNS:
        value = out[name]
        if isinstance(value, datetime):
            if value.tzinfo is None:  # SQLite drops the zone
                value = value.replace(tzinfo=timezone.utc)
            out[name] = value.isoformat()
    out["revision"] = out["revision"] or 1
    out["status"] = status_of(row)
    out["folder"] = folder_of(row)
    return out


def _add_columns(conn) -> None:
    """Add later nullable columns to a table created before them; data untouched."""
    present = {c["name"] for c in inspect(conn).get_columns(TABLE_NAME)}
    for name, kind in _ADDED_COLUMNS.items():
        if name in present:
            continue
        if kind == "timestamp":
            kind = "TIMESTAMP WITH TIME ZONE" if conn.dialect.name == "postgresql" else "TIMESTAMP"
        conn.execute(text(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {name} {kind}"))


#: The newest revision whose DDL this bootstrap covers; older states keep the
#: runtime bootstrap so the added columns exist before any query uses them.
PLAYGROUND_SCHEMA_REVISION = "0007_playground_revisions"


def bootstrap_playground_schema(conn) -> None:
    """Idempotent DDL shared by runtime compatibility mode and revision 0005."""
    metadata.create_all(conn, checkfirst=True)
    _add_columns(conn)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class OpenRefused(Exception):
    """The artifact can't be opened: ``not_found`` or ``expired``."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class ReviseRefused(Exception):
    """A revision can't land: ``not_found``, ``trashed`` or ``expired``."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class PlaygroundStore:
    _schema_guard = SchemaBootstrapGuard()

    def __init__(self, engine: Engine):
        self.engine = engine
        from ..utils.alembic_state import owns

        if not owns(engine, PLAYGROUND_SCHEMA_REVISION):
            self._schema_guard.run(engine, "playground-artifacts-v2", bootstrap_playground_schema)

    # -- write ---------------------------------------------------------------

    def insert(self, row: dict[str, Any]) -> dict[str, Any]:
        values = {**row, "created_at": row.get("created_at") or _now()}
        with self.engine.begin() as conn:
            notify_in_transaction(conn)  # TASK-1184: open mailboxes refresh (topic ``playground``)
            conn.execute(insert(artifacts).values(**values))
        return values

    def revise(self, artifact_id: str, *, user_id: str, sender_bot_id: str,
               values: dict[str, Any]) -> dict[str, Any]:
        """Replace a live artifact's content and bump its revision (TASK-2625).

        Only the sender of that user's artifact may revise it. Read and folder
        state are kept. A replaced blob is recorded as retired in the same
        transaction; the sweep deletes it after a grace period. Raises
        ``ReviseRefused`` with ``not_found`` (also for another sender's
        artifact), ``trashed`` or ``expired``.
        """
        mine = [artifacts.c.id == artifact_id, artifacts.c.user_id == user_id,
                artifacts.c.sender_bot_id == sender_bot_id]
        with self.engine.begin() as conn:
            current = conn.execute(
                select(artifacts.c.blob_key, artifacts.c.expired_at, artifacts.c.trashed_at)
                .where(*mine).with_for_update()
            ).mappings().first()
            if current is None:
                raise ReviseRefused("not_found")
            if current["expired_at"] is not None:
                raise ReviseRefused("expired")
            if current["trashed_at"] is not None:
                raise ReviseRefused("trashed")
            notify_in_transaction(conn)  # open viewers and mailboxes refresh (topic ``playground``)
            row = conn.execute(
                update(artifacts)
                .where(*mine, artifacts.c.expired_at.is_(None), artifacts.c.trashed_at.is_(None))
                .values(**values, revision=func.coalesce(artifacts.c.revision, 1) + 1,
                        updated_at=_now())
                .returning(*artifacts.c)
            ).mappings().first()
            if row is None:  # unreachable while the row lock is held
                raise ReviseRefused("not_found")
            previous = current["blob_key"]
            if previous and previous != row["blob_key"]:
                conn.execute(insert(retired_blobs).values(
                    blob_key=previous, artifact_id=artifact_id, retired_at=_now()))
        return dict(row)

    def retired_before(self, cutoff: datetime, limit: int = 500) -> list[str]:
        """Retired blob keys past their grace period (the sweep's work list)."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(retired_blobs.c.blob_key)
                .where(retired_blobs.c.retired_at < cutoff)
                .order_by(retired_blobs.c.retired_at)
                .limit(limit)
            ).all()
        return [r[0] for r in rows]

    def forget_retired(self, blob_key: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(retired_blobs.delete().where(retired_blobs.c.blob_key == blob_key))

    def open(self, artifact_id: str, user_id: str) -> tuple[dict[str, Any], str]:
        """Open (or reopen) a live artifact and mint a view token (TASK-2195).

        The first open sets ``opened_at`` (new -> seen); later opens keep it.
        A deleted artifact raises ``OpenRefused("expired")``.
        """
        token = secrets.token_urlsafe(32)
        mine = and_(artifacts.c.id == artifact_id, artifacts.c.user_id == user_id)
        with self.engine.begin() as conn:
            notify_in_transaction(conn)  # TASK-1184: open mailboxes refresh (topic ``playground``)
            row = conn.execute(
                update(artifacts)
                .where(mine, artifacts.c.expired_at.is_(None))
                .values(opened_at=func.coalesce(artifacts.c.opened_at, _now()),
                        view_token_hash=hash_token(token))
                .returning(*artifacts.c)
            ).mappings().first()
        if row is not None:
            return dict(row), token
        if self.get(artifact_id, user_id) is None:
            raise OpenRefused("not_found")
        raise OpenRefused("expired")

    def mark_expired(self, artifact_id: str, user_id: str | None = None, *,
                     in_trash: bool = False, trashed_before: datetime | None = None,
                     ) -> tuple[dict[str, Any] | None, bool]:
        """Expire one artifact's content in the database; idempotent.

        ``in_trash``/``trashed_before`` are checked in the same UPDATE, so a
        concurrent restore can never be overwritten. Returns ``(row, changed)``:
        the row (None when no such artifact exists for this user) and whether
        this call performed the transition.
        """
        cond = [artifacts.c.id == artifact_id]
        if user_id is not None:
            cond.append(artifacts.c.user_id == user_id)
        guard = [artifacts.c.expired_at.is_(None)]
        if in_trash or trashed_before is not None:
            guard.append(artifacts.c.trashed_at.is_not(None))
        if trashed_before is not None:
            guard.append(artifacts.c.trashed_at < trashed_before)
        with self.engine.begin() as conn:
            notify_in_transaction(conn)  # TASK-1184: open mailboxes refresh (topic ``playground``)
            changed = conn.execute(
                update(artifacts)
                .where(*cond, *guard)
                .values(expired_at=_now(), inline_content=None, view_token_hash=None)
            ).rowcount > 0
            row = conn.execute(select(*artifacts.c).where(*cond)).mappings().first()
        return (dict(row) if row is not None else None), changed

    def clear_blob_key(self, artifact_id: str, blob_key: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(update(artifacts)
                         .where(artifacts.c.id == artifact_id, artifacts.c.blob_key == blob_key)
                         .values(blob_key=None))

    def set_state(self, artifact_id: str, user_id: str, action: str) -> dict[str, Any] | None:
        """Apply a mailbox action to a live artifact; None when not found.

        ``read``/``unread``, ``archive``/``unarchive``, ``trash``/``restore``.
        Idempotent; a content-gone row is returned unchanged.
        """
        now = _now()
        values = {
            "read": {"opened_at": func.coalesce(artifacts.c.opened_at, now)},
            "unread": {"opened_at": None},
            "archive": {"archived_at": func.coalesce(artifacts.c.archived_at, now)},
            "unarchive": {"archived_at": None},
            "trash": {"trashed_at": func.coalesce(artifacts.c.trashed_at, now)},
            "restore": {"trashed_at": None},
        }[action]
        mine = [artifacts.c.id == artifact_id, artifacts.c.user_id == user_id]
        with self.engine.begin() as conn:
            notify_in_transaction(conn)  # TASK-1184: open mailboxes refresh (topic ``playground``)
            conn.execute(update(artifacts).where(*mine, artifacts.c.expired_at.is_(None)).values(**values))
            row = conn.execute(select(*artifacts.c).where(*mine)).mappings().first()
        return dict(row) if row is not None else None

    def trashed_before(self, cutoff: datetime) -> list[str]:
        """Live artifacts trashed before ``cutoff`` (the purge's work list)."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(artifacts.c.id)
                .where(artifacts.c.trashed_at.is_not(None), artifacts.c.trashed_at < cutoff,
                       artifacts.c.expired_at.is_(None))
            ).all()
        return [r[0] for r in rows]

    def pending_blob_deletes(self, limit: int = 500) -> list[dict[str, Any]]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(artifacts.c.id, artifacts.c.blob_key)
                .where(artifacts.c.expired_at.is_not(None), artifacts.c.blob_key.is_not(None))
                .limit(limit)
            ).mappings().all()
        return [dict(r) for r in rows]

    # -- read ----------------------------------------------------------------

    def get(self, artifact_id: str, user_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(*artifacts.c).where(
                artifacts.c.id == artifact_id, artifacts.c.user_id == user_id)).mappings().first()
        return dict(row) if row is not None else None

    def viewable(self, artifact_id: str, user_id: str, token: str) -> dict[str, Any] | None:
        """The live row when ``token`` is its view token; None otherwise."""
        row = self.get(artifact_id, user_id)
        if row is None or row["expired_at"] is not None or not row["view_token_hash"]:
            return None
        if not hmac.compare_digest(row["view_token_hash"], hash_token(token or "")):
            return None
        return row

    def list(self, user_id: str, *, bot_id: str | None = None, status: str | None = None,
             view: str | None = None, query: str | None = None,
             limit: int = 200) -> list[dict[str, Any]]:
        """Newest first. ``view`` defaults to the Inbox; ``query`` matches
        title, note and sender, case-insensitively."""
        c = artifacts.c
        q = select(*[c[name] for name in PUBLIC_COLUMNS]).where(c.user_id == user_id)
        view = view or "inbox"
        if view == "inbox":
            q = q.where(c.archived_at.is_(None), c.trashed_at.is_(None), c.expired_at.is_(None))
        elif view == "archive":
            q = q.where(c.archived_at.is_not(None), c.trashed_at.is_(None), c.expired_at.is_(None))
        elif view == "trash":
            q = q.where(c.trashed_at.is_not(None), c.expired_at.is_(None))
        if bot_id:
            q = q.where(c.sender_bot_id == bot_id)
        if status == "unopened":
            q = q.where(c.opened_at.is_(None), c.expired_at.is_(None))
        elif status == "expired":
            q = q.where(c.expired_at.is_not(None))
        if query:
            like = "%" + query.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            q = q.where(or_(func.lower(c.title).like(like, escape="\\"),
                            func.lower(func.coalesce(c.note, "")).like(like, escape="\\"),
                            func.lower(c.sender_bot_id).like(like, escape="\\")))
        q = q.order_by(c.created_at.desc(), c.id.desc()).limit(limit)
        with self.engine.connect() as conn:
            return [dict(r) for r in conn.execute(q).mappings().all()]

    def senders(self, user_id: str) -> list[str]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(artifacts.c.sender_bot_id, func.max(artifacts.c.created_at).label("last"))
                .where(artifacts.c.user_id == user_id)
                .group_by(artifacts.c.sender_bot_id)
                .order_by(func.max(artifacts.c.created_at).desc())
            ).all()
        return [r[0] for r in rows]
