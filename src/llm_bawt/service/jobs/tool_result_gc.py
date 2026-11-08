"""Prune transient overflow tool-result blobs + legacy payload rows (TASK-594).

Large tool results are content-addressed (``toolblobs/<sha256>``) in the object
store and are transient by nature — read on a rare "download full result" click,
never part of the durable conversation. This nightly sweep expires them so the
object store (and the pre-migration Postgres fallback table) stay lean.

What it prunes past ``retention_days``:

* **Object-store blobs.** A blob is a *candidate* if some ``tool_call_records``
  row older than the cutoff points at it. Because keys are content-addressed,
  the SAME blob can be shared by a NEWER record — so we subtract the *keep-set*
  (keys referenced by any row within retention) before deleting. Only keys that
  no surviving record needs are removed. Records whose blob we actually deleted
  get ``result_blob_key`` cleared and ``result_payload_available`` set false, so
  the UI stops offering a dead download link (the inline preview stays).
* **Legacy Postgres overflow rows.** ``tool_call_result_payloads`` rows attached
  to records older than the cutoff are deleted outright (this is the fallback
  table; the record keeps its preview).

The ``tool_call_records`` rows THEMSELVES are never deleted — they are the tool
timeline / audit. Only the heavy result body is reclaimed.

Re-entrancy: everything runs in one transaction computed from deterministic
SQL; a crash mid-run is safe because the next sweep recomputes the same set.
Blob deletes are idempotent (missing keys are not an error).

Result payload
--------------
``{"retention_days", "dry_run", "blob_candidates", "blobs_deleted",
   "records_cleared", "legacy_rows_pruned", "errors"}``
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import bindparam, text

from ..http_errors import safe_error_text

logger = logging.getLogger(__name__)

#: Default retention for overflow tool results. Overridable per task via
#: ``payload["retention_days"]`` (scheduler reads it from ``config_json``).
DEFAULT_RETENTION_DAYS = 14


def _distinct_keys(conn, *, older_than: bool, retention_days: int) -> set[str]:
    """DISTINCT non-null result_blob_key for records on one side of the cutoff."""
    comparison = "<" if older_than else ">="
    sql = text(
        f"""
        SELECT DISTINCT result_blob_key
        FROM tool_call_records
        WHERE result_blob_key IS NOT NULL
          AND created_at {comparison} NOW() - make_interval(days => :days)
        """
    )
    rows = conn.execute(sql, {"days": retention_days}).fetchall()
    return {row[0] for row in rows if row[0]}


def run_tool_result_gc(
    config: Any,
    *,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    dry_run: bool = False,
    blob_backend=None,
) -> dict[str, Any]:
    """Execute one tool-result GC sweep. Returns the counts payload.

    Parameters
    ----------
    config:
        Object exposing the shared engine via
        :func:`llm_bawt.utils.db.get_shared_engine`.
    retention_days:
        Overflow results older than this are eligible for pruning (default 14).
    dry_run:
        Compute the counts without deleting anything.
    blob_backend:
        Optional pre-built backend; defaults to the shared tool-results backend.
        ``None`` (no object store) skips blob deletion but still prunes legacy
        Postgres rows.
    """
    from llm_bawt.utils.db import get_shared_engine

    from ..tool_call_store import _get_tool_blob_backend

    engine = get_shared_engine(config)
    if engine is None:
        raise RuntimeError(
            "tool_result_gc requires a Postgres engine; config has no DB credentials"
        )

    backend = blob_backend if blob_backend is not None else _get_tool_blob_backend()

    result: dict[str, Any] = {
        "retention_days": retention_days,
        "dry_run": dry_run,
        "blob_candidates": 0,
        "blobs_deleted": 0,
        "records_cleared": 0,
        "legacy_rows_pruned": 0,
        "errors": [],
    }

    with engine.begin() as conn:
        expired = _distinct_keys(conn, older_than=True, retention_days=retention_days)
        keep = _distinct_keys(conn, older_than=False, retention_days=retention_days)
        to_delete = sorted(expired - keep)  # content-addressed keep-set guard
        result["blob_candidates"] = len(to_delete)

        legacy_count_sql = text(
            """
            SELECT COUNT(*)
            FROM tool_call_result_payloads p
            JOIN tool_call_records r ON r.id = p.tool_call_record_id
            WHERE r.created_at < NOW() - make_interval(days => :days)
            """
        )

        if dry_run:
            result["legacy_rows_pruned"] = int(
                conn.execute(legacy_count_sql, {"days": retention_days}).scalar() or 0
            )
            logger.info(
                "tool_result_gc DRY RUN: %d blob(s) eligible, %d legacy row(s) eligible (retention=%dd)",
                result["blob_candidates"],
                result["legacy_rows_pruned"],
                retention_days,
            )
            return result

        deleted: list[str] = []
        if backend is not None:
            for key in to_delete:
                try:
                    backend.delete(key)  # idempotent
                    deleted.append(key)
                except Exception as exc:  # pragma: no cover - backend transport
                    logger.warning("tool_result_gc: blob delete failed key=%s: %s", key, exc)
                    result["errors"].append({"key": key, "error": safe_error_text(exc)})
            result["blobs_deleted"] = len(deleted)
        elif to_delete:
            logger.warning(
                "tool_result_gc: %d blob(s) eligible but no object store available; "
                "leaving keys in place",
                len(to_delete),
            )

        # Clear record pointers only for blobs we actually removed. Records that
        # share a surviving key keep their download link.
        if deleted:
            clear_sql = text(
                """
                UPDATE tool_call_records
                SET result_blob_key = NULL, result_payload_available = FALSE
                WHERE result_blob_key IN :keys
                """
            ).bindparams(bindparam("keys", expanding=True))
            res = conn.execute(clear_sql, {"keys": deleted})
            result["records_cleared"] = res.rowcount or 0

        # Prune legacy Postgres overflow rows past retention (record keeps preview).
        prune_sql = text(
            """
            DELETE FROM tool_call_result_payloads
            WHERE tool_call_record_id IN (
                SELECT id FROM tool_call_records
                WHERE created_at < NOW() - make_interval(days => :days)
            )
            """
        )
        res2 = conn.execute(prune_sql, {"days": retention_days})
        result["legacy_rows_pruned"] = res2.rowcount or 0

    logger.info(
        "tool_result_gc: deleted %d blob(s), cleared %d record(s), pruned %d legacy row(s) "
        "(retention=%dd, errors=%d)",
        result["blobs_deleted"],
        result["records_cleared"],
        result["legacy_rows_pruned"],
        retention_days,
        len(result["errors"]),
    )
    return result
