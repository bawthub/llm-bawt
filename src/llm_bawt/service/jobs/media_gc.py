"""Garbage-collect orphan ``media_assets`` rows + their blobs (TASK-231, TASK-2224).

An asset is an *orphan* when nothing references it and either:

* ``expires_at IS NOT NULL AND expires_at < now()`` (soft-deleted), or
* ``created_at < now() - interval 'N days'`` (default 7).

A referenced asset is never an orphan, expired or not.

Reachability (TASK-2224)
------------------------
An asset is *referenced* when its id (``ma_`` + 26 ULID characters) appears
anywhere in a text, JSON or array column of either database:

* llm-bawt's own database, swept here, and
* BawtHub's database (task cards, steps, reviews, ...), which llm-bawt cannot
  read directly; BawtHub runs the same sweep and serves the result at
  ``GET /internal/media/referenced-assets`` (internal API).

The columns are discovered from ``information_schema`` at run time instead of
being listed by hand. The previous version only read columns named
``attachments``; card images, tool cards and inline images in message text
were invisible to it and got deleted after the grace period. Discovering every
text-bearing column makes a new holder covered by construction. Logs that
mention an id (turn logs, deliveries) keep that asset alive until their own
retention prunes them: extra disk, never lost data.

Fail closed
-----------
The sweep deletes nothing, and returns ``aborted_reason``, when:

* any column query fails or times out, or the whole sweep exceeds its budget;
* the BawtHub reference list is unreachable, not JSON, malformed, or not
  marked ``complete`` with at least one scanned column;
* the combined referenced set is empty while assets exist;
* the candidate list is larger than the breaker allows
  (``max(BREAKER_MIN, BREAKER_FRACTION * eligible assets)``), unless the
  caller passes an explicit ``max_candidates`` for an intended purge.

Known limit (pre-existing, not closed here): references are read before
deletion, so an id that becomes referenced after its source was swept, for
example a re-upload that deduplicates to an old unreferenced row, can still be
deleted in the same run.

Deletion path
-------------
Deletion delegates to :meth:`llm_bawt.media.store.MediaStore.delete`, which
removes the blob variants and the row together.

Result payload
--------------
``{"orphan_count", "deleted_count", "freed_bytes", "dry_run",
"scanned_columns", "bawthub_columns", "referenced_count", "errors"}`` plus
``aborted_reason`` when the run refused to delete. ``include_candidates=True``
(direct calls only) adds ``candidates``; scheduled runs keep ids out of
``job_runs`` because that table is itself swept for references.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from ..http_errors import safe_error_text

logger = logging.getLogger(__name__)

# Default grace period for unreferenced assets. Overridable per task via
# ``payload["grace_days"]`` (the scheduler reads this from ``config_json``).
DEFAULT_GRACE_DAYS = 7

ASSET_ID_PATTERN = r"ma_[0-9A-Z]{26}"
_ASSET_ID_RE = re.compile(ASSET_ID_PATTERN)

#: Per-statement limit for one column's sweep.
STATEMENT_TIMEOUT_MS = 120_000
#: Whole-sweep budget across every column of both databases.
DEFAULT_SWEEP_BUDGET_S = 900.0
#: HTTP timeout for BawtHub's reference list (it runs the same sweep).
BAWTHUB_TIMEOUT_S = 600.0
BAWTHUB_REFERENCES_PATH = "/internal/media/referenced-assets"

#: Breaker: refuse a run that would delete more than this many assets ...
BREAKER_MIN = 25
#: ... or this fraction of all collectable assets, whichever is larger.
BREAKER_FRACTION = 0.05

# Every text-bearing column of a public base table. Partitions are covered by
# their parent; media_assets itself is the thing being collected.
_FIND_TEXT_COLUMNS_SQL = text(
    """
    SELECT c.table_name, c.column_name
    FROM information_schema.columns c
    JOIN information_schema.tables t
      ON t.table_schema = c.table_schema AND t.table_name = c.table_name
    WHERE c.table_schema = 'public'
      AND t.table_type = 'BASE TABLE'
      AND c.table_name NOT LIKE 'messages\\_p\\_%' ESCAPE '\\'
      AND c.table_name <> 'media_assets'
      AND (
        c.data_type IN ('text', 'character varying', 'character', 'json', 'jsonb')
        OR (c.data_type = 'ARRAY'
            AND c.udt_name IN ('_text', '_varchar', '_bpchar', '_json', '_jsonb'))
      )
    ORDER BY c.table_name, c.column_name
    """
)


class ReferenceScanIncomplete(RuntimeError):
    """A reference source could not be read in full; nothing may be deleted."""


@dataclass(frozen=True)
class ReferenceSet:
    ids: frozenset[str]
    local_columns: int
    bawthub_columns: int


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _discover_text_columns(conn) -> list[tuple[str, str]]:
    rows = conn.execute(_FIND_TEXT_COLUMNS_SQL).fetchall()
    return [(row.table_name, row.column_name) for row in rows]


def _column_ids_sql(table: str, column: str) -> str:
    """Every asset id in one column, all matches per value (``'g'``)."""
    t, c = _quote_ident(table), _quote_ident(column)
    return (
        f"SELECT DISTINCT m[1] AS asset_id "
        f"FROM {t} x, LATERAL regexp_matches(x.{c}::text, '({ASSET_ID_PATTERN})', 'g') AS m "
        f"WHERE strpos(x.{c}::text, 'ma_') > 0"
    )


def collect_local_references(
    conn,
    *,
    deadline: float,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[set[str], int]:
    """Sweep every text-bearing column of this database for asset ids.

    Runs inside one read-only transaction with a per-statement timeout. Any
    failure raises :class:`ReferenceScanIncomplete`.
    """
    try:
        conn.exec_driver_sql("SET TRANSACTION READ ONLY")
        conn.exec_driver_sql(f"SET LOCAL statement_timeout = {int(STATEMENT_TIMEOUT_MS)}")
        columns = _discover_text_columns(conn)
    except Exception as e:
        raise ReferenceScanIncomplete(f"column discovery failed: {safe_error_text(e)}") from e
    if not columns:
        raise ReferenceScanIncomplete("no text columns discovered")

    ids: set[str] = set()
    for table, column in columns:
        if clock() > deadline:
            raise ReferenceScanIncomplete(f"sweep budget exceeded before {table}.{column}")
        try:
            rows = conn.exec_driver_sql(_column_ids_sql(table, column)).fetchall()
        except Exception as e:
            raise ReferenceScanIncomplete(f"{table}.{column}: {safe_error_text(e)}") from e
        ids.update(row[0] for row in rows if row[0])
    return ids, len(columns)


def _bawthub_url() -> str:
    # BawtHub's internal API (service token), as task associations use.
    base = os.getenv("BAWTHUB_TASK_ASSOCIATION_INTERNAL_URL", "http://frontend-prod:3002").rstrip("/")
    return base + BAWTHUB_REFERENCES_PATH


def parse_bawthub_references(payload: Any) -> tuple[set[str], int]:
    """Validate BawtHub's reference list strictly; raise on anything off."""
    if not isinstance(payload, dict):
        raise ReferenceScanIncomplete("BawtHub references: body is not an object")
    if payload.get("complete") is not True:
        raise ReferenceScanIncomplete("BawtHub references: not marked complete")
    columns = payload.get("columns_scanned")
    if type(columns) is not int or columns <= 0:
        raise ReferenceScanIncomplete("BawtHub references: no columns scanned")
    ids = payload.get("ids")
    if not isinstance(ids, list):
        raise ReferenceScanIncomplete("BawtHub references: ids is not a list")
    for value in ids:
        if not isinstance(value, str) or not _ASSET_ID_RE.fullmatch(value):
            raise ReferenceScanIncomplete("BawtHub references: malformed asset id")
    return set(ids), columns


def fetch_bawthub_references(timeout: float = BAWTHUB_TIMEOUT_S) -> tuple[set[str], int]:
    """BawtHub's referenced asset ids (task cards, steps, reviews, ...)."""
    import httpx

    from llm_bawt.bawthub_auth import service_auth

    try:
        with httpx.Client(timeout=timeout, trust_env=False, auth=service_auth()) as client:
            resp = client.get(_bawthub_url())
            resp.raise_for_status()
            payload = resp.json()
    except Exception as e:
        raise ReferenceScanIncomplete(f"BawtHub references unavailable: {safe_error_text(e)}") from e
    return parse_bawthub_references(payload)


def _find_orphan_assets(conn, referenced: frozenset[str], grace_days: int) -> list[dict[str, Any]]:
    """``[{id, sha256, size_bytes}, ...]`` for unreferenced, collectable assets."""
    sql = text(
        """
        SELECT a.id, a.sha256, a.size_bytes
        FROM media_assets a
        WHERE a.storage_key IS NULL
          AND NOT (a.id = ANY(:referenced))
          AND (
            (a.expires_at IS NOT NULL AND a.expires_at < NOW())
            OR a.created_at < NOW() - make_interval(days => :grace_days)
          )
        ORDER BY a.created_at ASC
        """
    )
    rows = conn.execute(sql, {"referenced": sorted(referenced), "grace_days": grace_days}).mappings().all()
    return [dict(r) for r in rows]


def _bound_statements(conn, *, deadline: float, clock: Callable[[], float]) -> None:
    """Cap this transaction's statements at the remaining sweep budget."""
    remaining_ms = int((deadline - clock()) * 1000)
    if remaining_ms <= 0:
        raise ReferenceScanIncomplete("sweep budget exceeded before candidate lookup")
    timeout_ms = min(int(STATEMENT_TIMEOUT_MS), remaining_ms)
    conn.exec_driver_sql(f"SET LOCAL statement_timeout = {timeout_ms}")


def _count_collectable(conn) -> int:
    return int(conn.execute(text("SELECT count(*) FROM media_assets WHERE storage_key IS NULL")).scalar_one())


def _resolve_media_store(config: Any):
    """Return a :class:`MediaStore` wired to ``config``'s shared engine."""
    from llm_bawt.media.assets import MediaAssetStore
    from llm_bawt.media.store import MediaStore

    db = MediaAssetStore(config)
    return MediaStore(db=db)


def breaker_limit(collectable: int) -> int:
    return max(BREAKER_MIN, int(collectable * BREAKER_FRACTION))


def _aborted(reason: str, *, dry_run: bool, refs: ReferenceSet | None = None, orphan_count: int = 0) -> dict[str, Any]:
    logger.warning("media_gc: deleting nothing: %s", reason)
    return {
        "orphan_count": orphan_count,
        "deleted_count": 0,
        "freed_bytes": 0,
        "dry_run": dry_run,
        "scanned_columns": refs.local_columns if refs else 0,
        "bawthub_columns": refs.bawthub_columns if refs else 0,
        "referenced_count": len(refs.ids) if refs else 0,
        "errors": [],
        "aborted_reason": reason,
    }


def run_media_gc(
    config: Any,
    *,
    grace_days: int = DEFAULT_GRACE_DAYS,
    dry_run: bool = False,
    media_store=None,
    max_candidates: int | None = None,
    include_candidates: bool = False,
    sweep_budget_s: float = DEFAULT_SWEEP_BUDGET_S,
    fetch_remote: Callable[[], tuple[set[str], int]] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Execute one GC sweep and return the counts payload.

    ``max_candidates`` overrides the breaker for an intended large purge.
    ``fetch_remote`` replaces the BawtHub call in tests.
    """
    from llm_bawt.utils.db import get_shared_engine

    engine = get_shared_engine(config)
    if engine is None:
        raise RuntimeError("media_gc requires a Postgres engine; config has no DB credentials")

    deadline = clock() + sweep_budget_s
    try:
        with engine.connect() as conn, conn.begin():
            local_ids, local_columns = collect_local_references(conn, deadline=deadline, clock=clock)
        remote_ids, remote_columns = (fetch_remote or fetch_bawthub_references)()
        if clock() > deadline:
            raise ReferenceScanIncomplete("sweep budget exceeded")
    except ReferenceScanIncomplete as e:
        return _aborted(str(e), dry_run=dry_run)

    refs = ReferenceSet(frozenset(local_ids | remote_ids), local_columns, remote_columns)

    try:
        with engine.connect() as conn, conn.begin():
            _bound_statements(conn, deadline=deadline, clock=clock)
            collectable = _count_collectable(conn)
            # Each source is checked on its own: references from one database
            # must not mask an empty (mispointed, freshly reset) other one.
            if collectable:
                if not local_ids:
                    return _aborted("local referenced set is empty while assets exist", dry_run=dry_run, refs=refs)
                if not remote_ids:
                    return _aborted("BawtHub referenced set is empty while assets exist", dry_run=dry_run, refs=refs)
            orphans = _find_orphan_assets(conn, refs.ids, grace_days)
        if clock() > deadline:
            raise ReferenceScanIncomplete("sweep budget exceeded during candidate lookup")
    except ReferenceScanIncomplete as e:
        return _aborted(str(e), dry_run=dry_run, refs=refs)
    except DBAPIError as e:
        return _aborted(f"candidate lookup failed: {safe_error_text(e)}", dry_run=dry_run, refs=refs)

    orphan_count = len(orphans)
    limit = max_candidates if max_candidates is not None else breaker_limit(collectable)
    logger.info(
        "media_gc: %d local + %d BawtHub column(s), %d referenced id(s), %d orphan(s), limit=%d, grace_days=%d, dry_run=%s",
        refs.local_columns, refs.bawthub_columns, len(refs.ids), orphan_count, limit, grace_days, dry_run,
    )

    result: dict[str, Any] = {
        "orphan_count": orphan_count,
        "deleted_count": 0,
        "freed_bytes": 0,
        "dry_run": dry_run,
        "scanned_columns": refs.local_columns,
        "bawthub_columns": refs.bawthub_columns,
        "referenced_count": len(refs.ids),
        "errors": [],
    }
    if include_candidates:
        result["candidates"] = [
            {"id": r["id"], "size_bytes": int(r.get("size_bytes") or 0)} for r in orphans
        ]

    if orphan_count > limit:
        result["aborted_reason"] = (
            f"{orphan_count} candidates exceed the breaker limit of {limit}; "
            "rerun with max_candidates to purge deliberately"
        )
        logger.warning("media_gc: deleting nothing: %s", result["aborted_reason"])
        return result

    if dry_run or orphan_count == 0:
        if dry_run:
            result["freed_bytes"] = sum(int(r.get("size_bytes") or 0) for r in orphans)
        return result

    store = media_store if media_store is not None else _resolve_media_store(config)

    # Lazy import keeps the scheduler's import graph free of the object store.
    from llm_bawt.media.object_store import BlobBackendUnavailable

    deleted_count = 0
    freed_bytes = 0
    errors: list[dict[str, Any]] = []
    aborted_reason: str | None = None

    # A wedged backend would cost a full timeout per orphan; stop after three
    # consecutive backend-unavailable errors and let the next sweep retry.
    consecutive_unavailable = 0
    UNAVAILABLE_ABORT_THRESHOLD = 3

    for row in orphans:
        asset_id = row["id"]
        size = int(row.get("size_bytes") or 0)
        try:
            store.delete(asset_id)
            deleted_count += 1
            freed_bytes += size
            consecutive_unavailable = 0
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
            logger.warning("media_gc: failed to delete asset %s: %s", asset_id, e)
            errors.append({"asset_id": asset_id, "error": safe_error_text(e)})

    logger.info(
        "media_gc: deleted %d/%d orphan(s), freed %d byte(s), errors=%d, aborted=%s",
        deleted_count, orphan_count, freed_bytes, len(errors), bool(aborted_reason),
    )
    result.update({"deleted_count": deleted_count, "freed_bytes": freed_bytes, "errors": errors})
    if aborted_reason:
        result["aborted_reason"] = aborted_reason
    return result
