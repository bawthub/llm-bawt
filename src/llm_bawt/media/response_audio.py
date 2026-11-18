"""Saved audio for spoken assistant replies (TASK-1115).

One ``response_audio`` row per spoken assistant turn links the turn (and,
once known, its assistant message) to a ``media_assets`` WAV. The asset
itself stays content-addressed and deduplicated; this table is the only
association, so deleting a reply never deletes bytes another row still uses.

Lifecycle
---------
``reserve`` (idempotent per ``turn_id``) hands the recorder a lease.
``finalize`` / ``fail`` succeed only with the live lease, and repeat calls
with the same result are no-ops. A row whose message is gone from both the
canonical and forgotten message tables is removed by :func:`reap_orphans`
from the existing media GC job; its asset then has no reference and the
ordinary media GC collects it after its grace period. Retention therefore
follows the message: no age-based expiry.

A finalize that arrives after its row was reaped returns ``None``; the
uploaded asset is left unreferenced for the GC, never deleted inline, so a
deduplicated blob shared with another reply survives.
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from ..utils.schema import SchemaBootstrapGuard
from .assets import new_asset_id

logger = logging.getLogger(__name__)

TABLE_NAME = "response_audio"

STATUS_RECORDING = "recording"
STATUS_FINALIZED = "finalized"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"
TERMINAL_STATUSES = (STATUS_FINALIZED, STATUS_PARTIAL, STATUS_FAILED)
ALL_STATUSES = (STATUS_RECORDING, *TERMINAL_STATUSES)

#: Recorder lease. Longer than the 10 minute audio cap plus upload time.
DEFAULT_LEASE_S = 20 * 60
#: A reservation whose turn never produced a message is dropped after this.
UNBOUND_RESERVATION_DAYS = 2
#: A bound row whose message is absent is kept this long: the canonical
#: message may still be committing. Reads hide it meanwhile, so a deleted
#: reply's audio is never served even before the reap.
PENDING_COMMIT_HOURS = 24

# Readers only see audio whose reply exists in canonical history. A reply
# being committed, forgotten (restorable) or deleted shows nothing.
_LIVE_MESSAGE = (
    "EXISTS (SELECT 1 FROM messages m WHERE m.bot_id = r.bot_id AND m.id = r.assistant_message_id)"
)
#: Hard cap matching the recorder: 10 minutes of 24 kHz mono PCM16 + header.
MAX_AUDIO_BYTES = 10 * 60 * 24_000 * 2 + 1024

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id                    TEXT PRIMARY KEY,
    turn_id               TEXT NOT NULL UNIQUE,
    assistant_message_id  TEXT UNIQUE,
    bot_id                TEXT NOT NULL,
    session_id            TEXT,
    owner_user_id         TEXT NOT NULL,
    status                TEXT NOT NULL,
    media_asset_id        TEXT,
    duration_ms           INTEGER,
    size_bytes            INTEGER,
    lease_token           TEXT,
    lease_expires_at      TIMESTAMPTZ,
    error                 TEXT,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT response_audio_status_check
        CHECK (status IN ('recording', 'finalized', 'partial', 'failed'))
)
"""

CREATE_INDEXES_SQL = [
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_owner ON {TABLE_NAME}(owner_user_id, created_at DESC)",
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_asset ON {TABLE_NAME}(media_asset_id)",
]

_PUBLIC_COLUMNS = (
    "id, turn_id, assistant_message_id, bot_id, session_id, owner_user_id, status, "
    "media_asset_id, duration_ms, size_bytes, error, created_at, updated_at"
)

# Reader projection: a recording whose upload lease ran out will never be
# finalized (the producer died or its upload failed), so readers see it as
# failed instead of "saving" forever. The row itself is left for the reaper.
_READ_COLUMNS = _PUBLIC_COLUMNS.replace(
    "status, ",
    "CASE WHEN status = 'recording' AND lease_expires_at < NOW() THEN 'failed' "
    "ELSE status END AS status, ",
)


class ResponseAudioConflict(RuntimeError):
    """The request contradicts the row's state (foreign lease, other message)."""


@dataclass(frozen=True)
class Reservation:
    row: dict[str, Any]
    lease_token: str | None  # None: someone else holds it, or it is terminal


def public_view(row: dict[str, Any]) -> dict[str, Any]:
    """Client descriptor; never includes the lease token."""
    return {
        "id": row["id"],
        "assistant_message_id": row.get("assistant_message_id"),
        "status": row["status"],
        "duration_ms": row.get("duration_ms"),
        "size_bytes": row.get("size_bytes"),
        "available": row["status"] in (STATUS_FINALIZED, STATUS_PARTIAL) and bool(row.get("media_asset_id")),
    }


class ResponseAudioStore:
    _schema_guard = SchemaBootstrapGuard()

    def __init__(self, engine):
        self.engine = engine
        self._schema_guard.run(engine, "response-audio-v1", self._bootstrap)

    @staticmethod
    def _bootstrap(conn) -> None:
        conn.execute(text(CREATE_TABLE_SQL))
        for sql in CREATE_INDEXES_SQL:
            conn.execute(text(sql))

    # -- reads ---------------------------------------------------------

    def _get(self, conn, *, turn_id: str, lock: bool = False) -> dict[str, Any] | None:
        row = conn.execute(
            text(
                f"SELECT {_PUBLIC_COLUMNS}, lease_token, lease_expires_at FROM {TABLE_NAME} WHERE turn_id = :t"
                + (" FOR UPDATE" if lock else "")
            ),
            {"t": turn_id},
        ).mappings().first()
        return dict(row) if row else None

    def get_for_message(self, assistant_message_id: str, owner_user_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    f"SELECT {_READ_COLUMNS} FROM {TABLE_NAME} r "
                    f"WHERE assistant_message_id = :m AND owner_user_id = :o AND {_LIVE_MESSAGE}"
                ),
                {"m": assistant_message_id, "o": owner_user_id},
            ).mappings().first()
        return dict(row) if row else None

    def get_many_for_messages(self, message_ids: list[str], owner_user_id: str) -> dict[str, dict[str, Any]]:
        if not message_ids:
            return {}
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT {_READ_COLUMNS} FROM {TABLE_NAME} r "
                    f"WHERE assistant_message_id = ANY(:ids) AND owner_user_id = :o AND {_LIVE_MESSAGE}"
                ),
                {"ids": list(message_ids), "o": owner_user_id},
            ).mappings().all()
        return {r["assistant_message_id"]: dict(r) for r in rows}

    # -- writes --------------------------------------------------------

    def reserve(
        self,
        *,
        turn_id: str,
        bot_id: str,
        owner_user_id: str,
        session_id: str | None = None,
        assistant_message_id: str | None = None,
        lease_s: int = DEFAULT_LEASE_S,
    ) -> Reservation:
        """Create or resume the reservation for ``turn_id``.

        Returns a fresh lease when the row is new or its previous lease has
        expired while still recording; otherwise the row without a lease.
        """
        token = secrets.token_urlsafe(24)
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    INSERT INTO {TABLE_NAME}
                        (id, turn_id, assistant_message_id, bot_id, session_id, owner_user_id,
                         status, lease_token, lease_expires_at)
                    VALUES (:id, :t, :m, :b, :s, :o, 'recording', :tok,
                            NOW() + make_interval(secs => :lease))
                    ON CONFLICT (turn_id) DO UPDATE
                       SET lease_token = EXCLUDED.lease_token,
                           lease_expires_at = EXCLUDED.lease_expires_at,
                           updated_at = NOW()
                     WHERE {TABLE_NAME}.status = 'recording'
                       AND {TABLE_NAME}.owner_user_id = EXCLUDED.owner_user_id
                       AND ({TABLE_NAME}.lease_expires_at IS NULL OR {TABLE_NAME}.lease_expires_at < NOW())
                    """
                ),
                {
                    "id": "ra_" + new_asset_id()[3:],
                    "t": turn_id, "m": assistant_message_id, "b": bot_id, "s": session_id,
                    "o": owner_user_id, "tok": token, "lease": int(lease_s),
                },
            )
            row = self._get(conn, turn_id=turn_id, lock=True)
            if row is None or row["owner_user_id"] != owner_user_id:
                raise ResponseAudioConflict("turn belongs to another owner")
            if assistant_message_id:
                self._bind(conn, row, assistant_message_id)
                row = self._get(conn, turn_id=turn_id)
        granted = row.pop("lease_token") == token
        row.pop("lease_expires_at", None)
        return Reservation(row=row, lease_token=token if granted else None)

    @staticmethod
    def _bind(conn, row: dict[str, Any], assistant_message_id: str) -> None:
        current = row.get("assistant_message_id")
        if current == assistant_message_id:
            return
        if current:
            raise ResponseAudioConflict("turn already bound to another message")
        updated = conn.execute(
            text(
                f"UPDATE {TABLE_NAME} SET assistant_message_id = :m, updated_at = NOW() "
                "WHERE turn_id = :t AND assistant_message_id IS NULL"
            ),
            {"m": assistant_message_id, "t": row["turn_id"]},
        ).rowcount
        if not updated:  # lost a race to another bind; callers hold the row lock
            raise ResponseAudioConflict("turn already bound to another message")

    def bind_message(self, *, turn_id: str, owner_user_id: str, assistant_message_id: str) -> dict[str, Any] | None:
        """Attach the canonical assistant message id once it is known. Idempotent."""
        with self.engine.begin() as conn:
            row = self._get(conn, turn_id=turn_id, lock=True)
            if row is None:
                return None
            if row["owner_user_id"] != owner_user_id:
                raise ResponseAudioConflict("turn belongs to another owner")
            self._bind(conn, row, assistant_message_id)
            row = self._get(conn, turn_id=turn_id)
        row.pop("lease_token", None)
        row.pop("lease_expires_at", None)
        return row

    def finalize(
        self,
        *,
        turn_id: str,
        lease_token: str,
        status: str,
        media_asset_id: str | None,
        duration_ms: int | None = None,
        size_bytes: int | None = None,
        error: str | None = None,
    ) -> dict[str, Any] | None:
        """Record the terminal state. Returns the row, or ``None`` if it was reaped.

        Repeating a finalize that already landed with the same asset and
        status is a no-op success; anything else without the live lease is a
        conflict.
        """
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"status must be one of {TERMINAL_STATUSES!r}")
        if status != STATUS_FAILED and not media_asset_id:
            raise ValueError("finalized and partial audio need a media asset")
        with self.engine.begin() as conn:
            updated = conn.execute(
                text(
                    f"""
                    UPDATE {TABLE_NAME}
                       SET status = :st, media_asset_id = :a, duration_ms = :d, size_bytes = :sz,
                           error = :e, lease_token = NULL, lease_expires_at = NULL, updated_at = NOW()
                     WHERE turn_id = :t AND status = 'recording' AND lease_token = :tok
                    """
                ),
                {"st": status, "a": media_asset_id, "d": duration_ms, "sz": size_bytes,
                 "e": (error or None) and error[:500], "t": turn_id, "tok": lease_token},
            ).rowcount
            row = self._get(conn, turn_id=turn_id)
        if row is None:
            return None
        row.pop("lease_token", None)
        row.pop("lease_expires_at", None)
        if updated:
            return row
        if row["status"] == status and row.get("media_asset_id") == media_asset_id:
            return row  # idempotent repeat
        raise ResponseAudioConflict("lease not held or reservation already finished")


def reap_orphans(conn, *, dry_run: bool = False) -> int:
    """Drop rows whose message no longer exists, or that never got one.

    Absence alone is not deletion: a bound row is reaped only once it is
    older than :data:`PENDING_COMMIT_HOURS`, so a reply still committing
    keeps its audio. Readers already hide audio of absent replies.

    Runs inside the media GC job before the reference sweep, so the freed
    assets lose their last reference in the same pass (and are collected
    once their grace period has passed). Forgotten messages keep their
    audio so a restore brings it back.
    """
    exists = conn.execute(
        text("SELECT to_regclass('public.response_audio') IS NOT NULL")
    ).scalar()
    if not exists:
        return 0
    where = f"""
        (r.assistant_message_id IS NOT NULL
         AND r.updated_at < NOW() - make_interval(hours => {PENDING_COMMIT_HOURS})
         AND NOT EXISTS (SELECT 1 FROM messages m WHERE m.bot_id = r.bot_id AND m.id = r.assistant_message_id)
         AND (to_regclass('public.forgotten_messages') IS NULL
              OR NOT EXISTS (SELECT 1 FROM forgotten_messages f WHERE f.bot_id = r.bot_id AND f.id = r.assistant_message_id)))
        OR (r.assistant_message_id IS NULL
            AND r.created_at < NOW() - make_interval(days => {UNBOUND_RESERVATION_DAYS}))
    """
    if dry_run:
        return int(conn.execute(text(f"SELECT count(*) FROM {TABLE_NAME} r WHERE {where}")).scalar_one())
    return conn.execute(text(f"DELETE FROM {TABLE_NAME} r WHERE {where}")).rowcount


_store: ResponseAudioStore | None = None


def get_response_audio_store(config=None) -> ResponseAudioStore | None:
    global _store
    if _store is None:
        from ..utils.config import config as default_config
        from ..utils.db import get_shared_engine

        engine = get_shared_engine(config or default_config)
        if engine is None:
            return None
        _store = ResponseAudioStore(engine)
    return _store
