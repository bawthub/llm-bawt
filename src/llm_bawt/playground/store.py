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
    Index(f"idx_{TABLE_NAME}_user_created", "user_id", "created_at"),
    Index(f"idx_{TABLE_NAME}_opened_live", "opened_at"),
)

#: Columns safe to return to any reader: never content or the token hash.
PUBLIC_COLUMNS = (
    "id", "user_id", "sender_bot_id", "title", "note", "kind", "mime", "filename",
    "size_bytes", "created_at", "opened_at", "expired_at", "archived_at", "trashed_at",
)

#: Inbox views. ``None`` (the default list) is the Inbox, so unread counts
#: built from the default list follow the Inbox.
VIEWS = ("inbox", "archive", "trash", "all")
_DATE_COLUMNS = ("created_at", "opened_at", "expired_at", "archived_at", "trashed_at")
_ADDED_COLUMNS = ("archived_at", "trashed_at")


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
    out["status"] = status_of(row)
    out["folder"] = folder_of(row)
    return out


def _add_columns(conn) -> None:
    """Add the TASK-2195 columns to a table created before them; data untouched."""
    present = {c["name"] for c in inspect(conn).get_columns(TABLE_NAME)}
    for name in _ADDED_COLUMNS:
        if name not in present:
            conn.execute(text(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {name} TIMESTAMP WITH TIME ZONE"
                              if conn.dialect.name == "postgresql"
                              else f"ALTER TABLE {TABLE_NAME} ADD COLUMN {name} TIMESTAMP"))


def _bootstrap(conn) -> None:
    metadata.create_all(conn, checkfirst=True)
    _add_columns(conn)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class OpenRefused(Exception):
    """The artifact can't be opened: ``not_found`` or ``expired``."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class PlaygroundStore:
    _schema_guard = SchemaBootstrapGuard()

    def __init__(self, engine: Engine):
        self.engine = engine
        self._schema_guard.run(engine, "playground-artifacts-v2", _bootstrap)

    # -- write ---------------------------------------------------------------

    def insert(self, row: dict[str, Any]) -> dict[str, Any]:
        values = {**row, "created_at": row.get("created_at") or _now()}
        with self.engine.begin() as conn:
            conn.execute(insert(artifacts).values(**values))
        return values

    def open(self, artifact_id: str, user_id: str) -> tuple[dict[str, Any], str]:
        """Open (or reopen) a live artifact and mint a view token (TASK-2195).

        The first open sets ``opened_at`` (new -> seen); later opens keep it.
        A deleted artifact raises ``OpenRefused("expired")``.
        """
        token = secrets.token_urlsafe(32)
        mine = and_(artifacts.c.id == artifact_id, artifacts.c.user_id == user_id)
        with self.engine.begin() as conn:
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

    def mark_expired(self, artifact_id: str, user_id: str | None = None) -> dict[str, Any] | None:
        """Expire one artifact's content in the database; idempotent.

        Returns the row (with any blob key still to delete), or None when no
        such artifact exists for this user.
        """
        cond = [artifacts.c.id == artifact_id]
        if user_id is not None:
            cond.append(artifacts.c.user_id == user_id)
        with self.engine.begin() as conn:
            conn.execute(
                update(artifacts)
                .where(*cond, artifacts.c.expired_at.is_(None))
                .values(expired_at=_now(), inline_content=None, view_token_hash=None)
            )
            row = conn.execute(select(*artifacts.c).where(*cond)).mappings().first()
        return dict(row) if row is not None else None

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
