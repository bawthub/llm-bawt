"""Garbage-collect orphan ``media_assets`` rows + their on-disk blobs (TASK-231).

An asset is an *orphan* if either:

* ``expires_at IS NOT NULL AND expires_at < now()`` — soft-deleted, GC'd
  immediately regardless of age. Producers use ``expires_at`` to mark
  things like failed/abandoned uploads.
* ``created_at < now() - interval 'N days'`` (default 7) **and** no
  ``{bot}_messages.attachments`` row anywhere references the asset by id.

The grace period gives clients a window to retry / paste cleanup / wire
the upload into a follow-up message without us yanking the blob out from
under them. Past the grace, an unreferenced asset is presumed permanently
unused.

Reference scan
--------------
Each per-bot table (``{bot}_messages``) carries an ``attachments`` JSONB
column shaped like ``[{"asset_id": "ma_...", "kind": "image"}, ...]``
(TASK-222). We enumerate every ``*_messages`` table from
``information_schema`` and ``UNION ALL`` their referenced asset ids into
one in-memory set. We do **not** hard-code the bot list — new bots that
land between releases are picked up automatically the next time the job
runs.

The reference set is bounded by the total count of message attachments,
which is small (tens of thousands at most for the foreseeable life of
this project). If that ever stops being true, switch to a per-asset
``EXISTS`` probe instead.

Deletion path
-------------
We delegate to :meth:`llm_bawt.media.store.MediaStore.delete` so blob
files (original / thumb / preview) and the DB row are removed together
with the same shard-prune logic that the upload path established. We
never ``os.unlink`` blobs directly — keeps the deletion path testable
and consistent.

Re-entrancy / crash safety
--------------------------
We compute the orphan list inside a single SELECT, then walk it row by
row calling ``MediaStore.delete``. Each delete is idempotent (deleting
an already-deleted asset returns cleanly). If we crash mid-loop, the
next nightly run picks up where we left off — the orphan list is
deterministic given the same DB state. No locks needed.

Result payload
--------------
``{"orphan_count": int, "deleted_count": int, "freed_bytes": int,
   "dry_run": bool, "errors": list[dict]}``
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text

from ..http_errors import safe_error_text

logger = logging.getLogger(__name__)

# Default grace period for unreferenced assets. Overridable per task via
# ``payload["grace_days"]`` (the scheduler reads this from ``config_json``).
DEFAULT_GRACE_DAYS = 7

# Discover every table that carries an ``attachments`` column at run time —
# column-driven so the table set the GC scans is BY CONSTRUCTION the set
# that can reference media assets, across every storage-layout era:
#
# * pre-TASK-571: the legacy per-bot ``<bot>_messages`` shards;
# * post-TASK-571: the partitioned ``messages`` parent (its ``messages_p_*``
#   partitions are excluded — scanning the parent covers them) plus, during
#   the soak window, the renamed ``zz_old_*`` shard copies. Including the
#   zz_old copies is deliberately conservative: an asset referenced only by
#   pre-cutover history stays alive until --finalize drops those tables.
#
# A name-pattern LIKE here would be a data-loss bug: after cutover nothing
# matches ``%_messages`` anymore, the referenced-set query would come back
# empty, and EVERY asset past the grace window would look orphaned.
_FIND_MESSAGES_TABLES_SQL = text(
    """
    SELECT c.table_name
    FROM information_schema.columns c
    JOIN information_schema.tables t
      ON t.table_schema = c.table_schema AND t.table_name = c.table_name
    WHERE c.table_schema = 'public'
      AND c.column_name = 'attachments'
      AND t.table_type = 'BASE TABLE'
      AND c.table_name NOT LIKE 'messages\\_p\\_%' ESCAPE '\\'
    ORDER BY c.table_name
    """
)


def _discover_message_tables(conn) -> list[str]:
    """Return every attachments-carrying table currently in ``public``."""
    rows = conn.execute(_FIND_MESSAGES_TABLES_SQL).fetchall()
    return [row.table_name for row in rows]


def _build_referenced_query(tables: list[str]) -> str:
    """Build a UNION ALL over every messages table's attachments.

    Each per-table fragment expands the JSONB array, pulls out the
    ``asset_id`` text, and filters NULLs (a malformed attachment dict
    without ``asset_id`` would otherwise show up as NULL and be wrong).

    ``information_schema`` only emits identifier-safe characters (lower
    snake_case ASCII), so it's safe to interpolate the names directly
    -- there's no user input here.
    """
    if not tables:
        return "SELECT NULL::text AS asset_id WHERE FALSE"

    fragments = [
        f"SELECT (a.elem->>'asset_id') AS asset_id "
        f"FROM {tname} m, LATERAL jsonb_array_elements(m.attachments) AS a(elem) "
        f"WHERE jsonb_typeof(m.attachments) = 'array' "
        f"  AND (a.elem->>'asset_id') IS NOT NULL"
        for tname in tables
    ]
    return " UNION ALL ".join(fragments)


def _find_orphan_assets(
    conn,
    tables: list[str],
    grace_days: int,
) -> list[dict[str, Any]]:
    """Return ``[{id, sha256, size_bytes}, ...]`` for every orphan.

    Single round-trip: builds the referenced-asset set inside Postgres
    and joins it against ``media_assets`` in one go. The
    ``NOT EXISTS`` correlated subquery is fast given there's an index
    on ``media_assets.sha256`` and the referenced set comes through a
    CTE that gets materialized once.
    """
    referenced_sql = _build_referenced_query(tables)
    sql = text(
        f"""
        WITH referenced AS (
            SELECT DISTINCT asset_id FROM ({referenced_sql}) r
        )
        SELECT a.id, a.sha256, a.size_bytes
        FROM media_assets a
        WHERE a.storage_key IS NULL AND (
            (a.expires_at IS NOT NULL AND a.expires_at < NOW())
         OR (
            a.created_at < NOW() - make_interval(days => :grace_days)
            AND NOT EXISTS (
                SELECT 1 FROM referenced r WHERE r.asset_id = a.id
            )
         ))
        ORDER BY a.created_at ASC
        """
    )
    rows = conn.execute(sql, {"grace_days": grace_days}).mappings().all()
    return [dict(r) for r in rows]


def _resolve_media_store(config: Any):
    """Return a process-wide :class:`MediaStore`.

    Imports late so unit tests that monkeypatch :mod:`llm_bawt.media.store`
    pick up the replacement. Wires the Store against the active config
    (same engine via ``get_shared_engine``) -- never construct a
    fresh engine here.
    """
    from llm_bawt.media.assets import MediaAssetStore
    from llm_bawt.media.store import MediaStore

    db = MediaAssetStore(config)
    return MediaStore(db=db)


def run_media_gc(
    config: Any,
    *,
    grace_days: int = DEFAULT_GRACE_DAYS,
    dry_run: bool = False,
    media_store=None,
) -> dict[str, Any]:
    """Execute one GC sweep. Returns the counts payload.

    Parameters
    ----------
    config:
        Any object exposing the shared SQLAlchemy engine via
        :func:`llm_bawt.utils.db.get_shared_engine`. In production this
        is the live :class:`llm_bawt.utils.config.Config` from the
        running service.
    grace_days:
        Unreferenced assets younger than this are kept (default 7).
    dry_run:
        Compute the orphan list and return counts without deleting
        anything. Useful for ops verification.
    media_store:
        Optional pre-built :class:`MediaStore`. Defaults to a fresh
        Store wired to ``config``; tests inject a fake.

    Returns
    -------
    dict
        ``{"orphan_count", "deleted_count", "freed_bytes", "dry_run",
        "scanned_tables", "errors"}``. ``errors`` is a list of
        ``{"asset_id", "error"}`` entries -- a non-empty list means a
        per-asset delete blew up but the sweep continued.
    """
    from llm_bawt.utils.db import get_shared_engine

    engine = get_shared_engine(config)
    if engine is None:
        raise RuntimeError(
            "media_gc requires a Postgres engine; config has no DB credentials"
        )

    store = media_store if media_store is not None else _resolve_media_store(config)

    with engine.connect() as conn:
        tables = _discover_message_tables(conn)
        orphans = _find_orphan_assets(conn, tables, grace_days)

    orphan_count = len(orphans)
    logger.info(
        "media_gc: scanned %d messages table(s), found %d orphan(s), grace_days=%d, dry_run=%s",
        len(tables),
        orphan_count,
        grace_days,
        dry_run,
    )

    if dry_run or orphan_count == 0:
        freed = sum(int(r.get("size_bytes") or 0) for r in orphans) if dry_run else 0
        return {
            "orphan_count": orphan_count,
            "deleted_count": 0,
            "freed_bytes": freed,
            "dry_run": dry_run,
            "scanned_tables": len(tables),
            "errors": [],
        }

    # Lazy import — keeps the GC module free of an object_store dep at
    # load time, which matters for the scheduler's import graph.
    from llm_bawt.media.object_store import BlobBackendUnavailable

    deleted_count = 0
    freed_bytes = 0
    errors: list[dict[str, Any]] = []
    aborted_reason: str | None = None

    # If the storage backend is wedged we'd hammer it for every orphan,
    # each delete costing the full boto3 timeout. Bail out after three
    # consecutive backend-unavailable errors and let the next nightly
    # sweep retry — preserves "never crash the scheduler" without
    # turning a backend outage into a thousand-line error log.
    consecutive_unavailable = 0
    UNAVAILABLE_ABORT_THRESHOLD = 3

    for row in orphans:
        asset_id = row["id"]
        size = int(row.get("size_bytes") or 0)
        try:
            store.delete(asset_id)
            deleted_count += 1
            freed_bytes += size
            consecutive_unavailable = 0  # any success resets the streak
        except BlobBackendUnavailable as e:
            consecutive_unavailable += 1
            logger.warning(
                "media_gc: backend unavailable for asset %s (streak=%d): %s",
                asset_id, consecutive_unavailable, e,
            )
            errors.append({"asset_id": asset_id, "error": f"backend unavailable: {safe_error_text(e)}"})
            if consecutive_unavailable >= UNAVAILABLE_ABORT_THRESHOLD:
                aborted_reason = (
                    f"backend unavailable for {consecutive_unavailable} consecutive "
                    "deletes — aborting pass, next sweep will retry"
                )
                logger.warning("media_gc: %s", aborted_reason)
                break
        except Exception as e:
            consecutive_unavailable = 0
            logger.warning(
                "media_gc: failed to delete asset %s: %s", asset_id, e
            )
            errors.append({"asset_id": asset_id, "error": safe_error_text(e)})

    logger.info(
        "media_gc: deleted %d/%d orphan(s), freed %d byte(s), errors=%d, aborted=%s",
        deleted_count,
        orphan_count,
        freed_bytes,
        len(errors),
        bool(aborted_reason),
    )

    result: dict[str, Any] = {
        "orphan_count": orphan_count,
        "deleted_count": deleted_count,
        "freed_bytes": freed_bytes,
        "dry_run": False,
        "scanned_tables": len(tables),
        "errors": errors,
    }
    if aborted_reason:
        result["aborted_reason"] = aborted_reason
    return result
