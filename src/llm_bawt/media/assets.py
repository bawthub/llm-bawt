"""PostgreSQL store for the ``media_assets`` table.

This is the canonical registry for normalized, content-addressed media
blobs that get attached to chat messages, tool outputs, or agent payloads.
The actual bytes live on disk (see :mod:`llm_bawt.media.storage`); only
metadata is persisted here.

TASK-222 foundation — read together with:

- ``media_generations`` table (legacy job-tracker; different concern).
- ``{bot}_messages.attachments`` JSONB column added in the same migration:
  every chat-history row carries a tiny array of ``{"asset_id": "ma_...",
  "kind": "image"}`` references that join back to this table on read.

The table is idempotent: it's created on first ``MediaAssetStore`` init
and reused thereafter — same pattern as ``MediaGenerationStore`` and the
per-bot memory tables. There is intentionally no Alembic in this project.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Column, DateTime, Integer, text
from sqlmodel import Field, SQLModel

from ..utils.config import Config
from ..utils.schema import SchemaBootstrapGuard

logger = logging.getLogger(__name__)

TABLE_NAME = "media_assets"

# Allowed values for media_assets.source. Stored as a CHECK constraint so
# typos in producer code surface as a DB error instead of silently writing
# garbage that the GC job can't reason about.
ALLOWED_SOURCES = ("chat_upload", "tool_generated", "agent_attachment")

# Allowed values for media_assets.kind (TASK-847). Canonical list lives with
# the strategy registry; mirrored here so the DB layer validates without
# importing Pillow-backed code.
ALLOWED_KINDS = ("image", "file")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id                  TEXT PRIMARY KEY,
    sha256              TEXT NOT NULL,
    mime_type           TEXT NOT NULL,
    original_mime_type  TEXT,
    size_bytes          INTEGER NOT NULL,
    width               INTEGER,
    height              INTEGER,
    kind                TEXT NOT NULL DEFAULT 'image',
    filename            TEXT,
    storage_key         TEXT,
    source              TEXT NOT NULL,
    owner_user_id       TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at          TIMESTAMPTZ,
    CONSTRAINT media_assets_source_check
        CHECK (source IN ('chat_upload', 'tool_generated', 'agent_attachment'))
)
"""

# TASK-847: columns added after the table first shipped. ``CREATE TABLE IF
# NOT EXISTS`` is a no-op on existing installs, so each new column also gets
# an idempotent ``ADD COLUMN IF NOT EXISTS`` that runs on every bootstrap.
# ``kind`` defaults to 'image' so every pre-existing row keeps its meaning.
ALTER_TABLE_SQL = [
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'image'",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS filename TEXT",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS storage_key TEXT",
    # Named mutable slots and immutable assets can contain identical bytes.
    # Preserve dedup uniqueness for immutable assets only, before dropping the old constraint.
    f"CREATE UNIQUE INDEX IF NOT EXISTS idx_media_assets_immutable_sha ON {TABLE_NAME}(sha256) WHERE storage_key IS NULL",
    f"ALTER TABLE {TABLE_NAME} DROP CONSTRAINT IF EXISTS media_assets_sha256_key",
    f"CREATE UNIQUE INDEX IF NOT EXISTS idx_media_assets_storage_key ON {TABLE_NAME}(storage_key)",
]

# Listing assets for a given user, newest-first, is the dominant access
# pattern (history rendering + the nightly GC job both scan by owner).
CREATE_INDEXES_SQL = [
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_owner_created "
    f"ON {TABLE_NAME}(owner_user_id, created_at DESC)",
    # expires_at lookups for soft-deletion / TTL cleanup.
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_expires_at "
    f"ON {TABLE_NAME}(expires_at) WHERE expires_at IS NOT NULL",
]


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

# Crockford's base32 alphabet — the ULID spec. Lexicographically ordered
# when encoded as a 26-char string, which means ``id`` sorts the same as
# ``created_at``. Useful for batch fetches that don't want to sort.
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _encode_crockford(value: int, length: int) -> str:
    chars: list[str] = []
    for _ in range(length):
        chars.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


def new_asset_id() -> str:
    """Return a fresh ``ma_<ulid>`` identifier.

    No external ulid library — we generate the canonical 26-char encoding
    from a 48-bit millisecond timestamp + 80 bits of cryptographic entropy.
    Roughly 1.2 * 10^24 ids/ms before a collision becomes likely; good
    enough for chat uploads and tool outputs.
    """
    ms = int(time.time() * 1000)
    rand = int.from_bytes(os.urandom(10), "big")
    return "ma_" + _encode_crockford(ms, 10) + _encode_crockford(rand, 16)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def _build_engine(config: Config):
    """Return the process-wide shared SQLAlchemy engine (TASK-202)."""
    from ..utils.db import get_shared_engine
    return get_shared_engine(config)


class MediaAssetStore:
    """CRUD operations for the ``media_assets`` table.

    Construct once per process; the underlying engine is shared across
    every Store via ``get_shared_engine`` so this is cheap.
    """

    _schema_guard = SchemaBootstrapGuard()

    def __init__(self, config: Config):
        self.engine = _build_engine(config)
        if self.engine is not None:
            self._ensure_table()
        else:
            logger.warning(
                "MediaAssetStore initialized with no engine — Postgres credentials missing"
            )

    # ------------------------------------------------------------------
    # Table initialisation
    # ------------------------------------------------------------------

    def _ensure_table(self) -> None:
        """Create the ``media_assets`` table + indexes if missing."""
        def bootstrap(conn) -> None:
            conn.execute(text(CREATE_TABLE_SQL))
            for alter_sql in ALTER_TABLE_SQL:
                conn.execute(text(alter_sql))
            for idx_sql in CREATE_INDEXES_SQL:
                conn.execute(text(idx_sql))

        self._schema_guard.run(self.engine, "media-assets-named-slots-v1", bootstrap)

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(
        self,
        *,
        sha256: str,
        mime_type: str,
        size_bytes: int,
        source: str,
        original_mime_type: str | None = None,
        width: int | None = None,
        height: int | None = None,
        owner_user_id: str | None = None,
        expires_at: datetime | None = None,
        asset_id: str | None = None,
        kind: str = "image",
        filename: str | None = None,
    ) -> dict[str, Any]:
        """Insert a new media asset row.

        ``kind`` is ``image`` (normalised WebP + variants) or ``file``
        (opaque bytes, TASK-847); ``filename`` is the caller-supplied
        display name, kept for ``Content-Disposition`` and UI chips.

        Returns the inserted row as a dict. If a row with the same
        ``sha256`` already exists, returns that existing row instead
        (dedup-by-content) — callers should rely on this for idempotent
        uploads rather than catching IntegrityError themselves.
        """
        if source not in ALLOWED_SOURCES:
            raise ValueError(
                f"source must be one of {ALLOWED_SOURCES!r}, got {source!r}"
            )
        if kind not in ALLOWED_KINDS:
            raise ValueError(f"kind must be one of {ALLOWED_KINDS!r}, got {kind!r}")

        existing = self.get_by_sha256(sha256)
        if existing is not None:
            logger.debug(
                "media_assets dedup hit: sha256=%s -> id=%s",
                sha256[:12],
                existing["id"],
            )
            return existing

        row_id = asset_id or new_asset_id()
        insert_sql = text(f"""
            INSERT INTO {TABLE_NAME}
                (id, sha256, mime_type, original_mime_type, size_bytes,
                 width, height, kind, filename, source, owner_user_id, expires_at)
            VALUES
                (:id, :sha256, :mime_type, :original_mime_type, :size_bytes,
                 :width, :height, :kind, :filename, :source, :owner_user_id, :expires_at)
            ON CONFLICT (sha256) WHERE storage_key IS NULL DO NOTHING
            RETURNING *
        """)
        params = {
            "id": row_id,
            "sha256": sha256,
            "mime_type": mime_type,
            "original_mime_type": original_mime_type,
            "size_bytes": size_bytes,
            "width": width,
            "height": height,
            "kind": kind,
            "filename": filename,
            "source": source,
            "owner_user_id": owner_user_id,
            "expires_at": expires_at,
        }
        with self.engine.connect() as conn:
            row = conn.execute(insert_sql, params).mappings().first()
            conn.commit()
        if row:
            return dict(row)
        # A concurrent upload of the same bytes inserted first; return its row.
        existing = self.get_by_sha256(sha256)
        return existing if existing is not None else {}

    def replace_named(self, *, storage_key: str, write_blob, **metadata) -> dict[str, Any]:
        """Serialize named writes and update one row, never immutable dedup rows.

        This short DB transaction spans the blob write intentionally: a slot's
        bytes and metadata must not be published by concurrent writers out of
        order. Failed writes roll back metadata; failed commits leave reads
        failing closed on checksum mismatch until a later replacement heals it.
        """
        with self.engine.begin() as conn:
            conn.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
                         {"key": "media-slot:" + storage_key})
            row = conn.execute(text(f"""
                INSERT INTO {TABLE_NAME}
                    (id, storage_key, sha256, mime_type, original_mime_type, size_bytes,
                     kind, filename, source, owner_user_id)
                VALUES (:id, :storage_key, :sha256, :mime_type, :original_mime_type,
                        :size_bytes, 'file', :filename, :source, :owner_user_id)
                ON CONFLICT (storage_key) DO UPDATE SET
                    sha256 = EXCLUDED.sha256, mime_type = EXCLUDED.mime_type,
                    original_mime_type = EXCLUDED.original_mime_type,
                    size_bytes = EXCLUDED.size_bytes, filename = EXCLUDED.filename,
                    source = EXCLUDED.source, owner_user_id = EXCLUDED.owner_user_id,
                    expires_at = NULL
                RETURNING *
            """), {"id": new_asset_id(), "storage_key": storage_key, **metadata}).mappings().one()
            write_blob()
            return dict(row)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_by_id(self, asset_id: str) -> dict[str, Any] | None:
        sql = text(f"SELECT * FROM {TABLE_NAME} WHERE id = :id")
        with self.engine.connect() as conn:
            row = conn.execute(sql, {"id": asset_id}).mappings().first()
            return dict(row) if row else None

    def get_many(self, asset_ids: list[str]) -> list[dict[str, Any]]:
        """Batch-fetch rows by id in a single round-trip (TASK-226).

        The ``/v1/history`` enrichment path collects every distinct
        ``asset_id`` referenced by a page of messages and resolves them
        in one query — paging fan-out per row would be O(messages) DB
        hits on a chatty history endpoint.

        Returns the rows in arbitrary order; callers should index by
        ``id`` if they need stable lookup. Missing ids are silently
        omitted (the history serializer treats those as orphans).
        """
        if not asset_ids:
            return []
        # ANY(:ids) lets us send the whole id list as a single SQL array
        # parameter — works with psycopg2's list-as-array codec.
        sql = text(f"SELECT * FROM {TABLE_NAME} WHERE id = ANY(:ids)")
        with self.engine.connect() as conn:
            rows = conn.execute(sql, {"ids": list(asset_ids)}).mappings().all()
            return [dict(r) for r in rows]

    def get_by_sha256(self, sha256: str) -> dict[str, Any] | None:
        sql = text(f"SELECT * FROM {TABLE_NAME} WHERE sha256 = :sha AND storage_key IS NULL")
        with self.engine.connect() as conn:
            row = conn.execute(sql, {"sha": sha256}).mappings().first()
            return dict(row) if row else None

    def list_by_owner(
        self,
        owner_user_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        sql = text(
            f"SELECT * FROM {TABLE_NAME} "
            f"WHERE owner_user_id = :owner "
            f"ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
        )
        params = {"owner": owner_user_id, "limit": limit, "offset": offset}
        with self.engine.connect() as conn:
            rows = conn.execute(sql, params).mappings().all()
            return [dict(r) for r in rows]

    def browse(
        self, *, q: str = "", kind: str | None = None,
        source: str | None = None, limit: int = 48, offset: int = 0,
    ) -> dict[str, Any]:
        """Browse the tenant's registry, not raw S3 keys or image variants.

        Bound parameters and literal substring matching keep filenames containing
        SQL LIKE wildcards searchable. Stable ordering breaks timestamp ties.
        """
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("Invalid pagination")
        if kind is not None and kind not in ALLOWED_KINDS:
            raise ValueError("Invalid asset kind")
        if source is not None and source not in ALLOWED_SOURCES:
            raise ValueError("Invalid asset source")
        clauses = []
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if q.strip():
            escaped = q.strip().lower().replace("!", "!!").replace("%", "!%").replace("_", "!_")
            params["query"] = f"%{escaped}%"
            clauses.append("(" + " OR ".join(
                f"LOWER(COALESCE({column}, '')) LIKE :query ESCAPE '!'"
                for column in ("id", "filename", "mime_type", "storage_key", "owner_user_id")
            ) + ")")
        if kind:
            clauses.append("kind = :kind")
            params["kind"] = kind
        if source:
            clauses.append("source = :source")
            params["source"] = source
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.engine.connect() as conn:
            totals = conn.execute(text(
                f"SELECT COUNT(*) AS total, COALESCE(SUM(size_bytes), 0) AS total_bytes "
                f"FROM {TABLE_NAME}{where}"
            ), params).mappings().one()
            rows = conn.execute(text(
                f"SELECT * FROM {TABLE_NAME}{where} "
                "ORDER BY created_at DESC, id DESC LIMIT :limit OFFSET :offset"
            ), params).mappings().all()
        return {"items": [dict(row) for row in rows], "total": totals["total"],
                "total_bytes": totals["total_bytes"], "limit": limit, "offset": offset}

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, asset_id: str) -> bool:
        """Hard-delete a row. Caller is responsible for removing the
        on-disk blob first (see :class:`MediaStorage`). Returns True if a
        row was removed."""
        sql = text(f"DELETE FROM {TABLE_NAME} WHERE id = :id")
        with self.engine.connect() as conn:
            result = conn.execute(sql, {"id": asset_id})
            conn.commit()
            return result.rowcount > 0


# ---------------------------------------------------------------------------
# SQLModel mirror
# ---------------------------------------------------------------------------


class MediaAsset(SQLModel, table=True):
    """Declarative mirror of the ``media_assets`` table.

    Provided so other Python code can ``from llm_bawt.media.assets import
    MediaAsset`` and use it with SQLModel / FastAPI response models. The
    actual table is created by :class:`MediaAssetStore._ensure_table`
    using raw DDL (the project pattern); this class is **not** used to
    issue ``CREATE TABLE`` -- ``__table_args__ = {"extend_existing": True}``
    lets it coexist with the raw definition without conflict.

    Field names mirror the SQL schema 1:1. ``created_at`` is timezone-aware
    (TIMESTAMPTZ); ``expires_at`` is nullable.
    """

    __tablename__ = "media_assets"
    __table_args__ = {"extend_existing": True}

    id: str = Field(default_factory=new_asset_id, primary_key=True)
    sha256: str = Field(index=True)
    storage_key: Optional[str] = Field(default=None, unique=True)
    mime_type: str
    original_mime_type: Optional[str] = Field(default=None)
    size_bytes: int = Field(sa_column=Column(Integer, nullable=False))
    width: Optional[int] = Field(default=None)
    height: Optional[int] = Field(default=None)
    kind: str = Field(default="image", description="image | file (TASK-847)")
    filename: Optional[str] = Field(default=None)
    source: str = Field(description="chat_upload | tool_generated | agent_attachment")
    owner_user_id: Optional[str] = Field(default=None, index=True)
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()")),
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
