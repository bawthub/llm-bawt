"""Operator-only forward read cursor over one bot/user pair's user-role history.

Each cursor covers exactly one canonical bot slug and one user. The bot is
validated through the classifier's partition check, so protected bots and
shared or ambiguous storage partitions are refused before any SQL runs, and a
scan touches exactly one ``messages`` partition. Rows belong to the user only
through their ``sessions`` row (``sessions.bot_id``/``user_id``), never through
``author_entity_id``.

Within one partition ``(timestamp, id)`` is a total order, so a strict keyset
``> cursor`` neither skips nor repeats a row that was present when it was read.
The cursor is a forward keyset, not a lossless ingestion log: rows inserted
late with older timestamps, restored from ignored, or whose provenance changed
behind the cursor are outside observed coverage. Recovery is
:func:`reset_cursor` and a rescan; classification is idempotent, so a rescan
only reprocesses and recounts.

Dormant: no bot tool, MCP route or scheduler calls this module.
"""
from __future__ import annotations

import json
import time
from typing import Any, Mapping, Sequence

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB

from .history_classification import _partitions, classify, report

CURSOR_TABLE = "history_read_cursors"

_DDL = f"""
CREATE TABLE IF NOT EXISTS {CURSOR_TABLE} (
  bot_id               VARCHAR(128)     NOT NULL,
  user_id              VARCHAR(128)     NOT NULL,
  cursor_ts            DOUBLE PRECISION NULL,
  cursor_message_id    TEXT             NULL,
  processed_rows_total BIGINT           NOT NULL DEFAULT 0,
  last_run_at          TIMESTAMPTZ      NULL,
  last_coverage        JSONB            NULL,
  updated_at           TIMESTAMPTZ      NOT NULL DEFAULT now(),
  PRIMARY KEY (bot_id, user_id),
  CHECK ((cursor_ts IS NULL) = (cursor_message_id IS NULL))
)
"""


def ensure_schema(conn) -> None:
    """Additive and idempotent, like the other module-owned tables."""
    conn.execute(text(_DDL))


# The LIMIT applies to owned message rows before the delivery join, so a
# message with several deliveries cannot push owned rows off the page.
_PAGE = text("""
    SELECT m.id, m.bot_id, m.content, m.timestamp,
           m.author_entity_type, m.author_entity_id,
           d.id AS delivery_id, d.sender_bot_id, d.target_bot_id,
           d.metadata_json AS delivery_metadata
    FROM (
        SELECT m.id, m.bot_id, m.content, m.timestamp,
               m.author_entity_type, m.author_entity_id
        FROM messages m
        JOIN sessions s ON s.id = m.session_id
         AND s.bot_id = :slug AND s.user_id = :user_id
        WHERE m.role = 'user' AND m.bot_id = :part
          AND (m.timestamp, m.id) > (:after_ts, :after_id)
          AND m.timestamp < :until
        ORDER BY m.timestamp, m.id
        LIMIT :limit
    ) m
    LEFT JOIN inter_bot_deliveries d
      ON d.user_message_id = m.id AND d.target_bot_id = :slug
    ORDER BY m.timestamp, m.id, d.id
""")

# Partition rows in the same window with no session of this bot: their owner
# is unknown, so they are counted, never attributed to this user.
_UNATTRIBUTED_SQL = """
    SELECT count(*) FROM messages m
    WHERE m.role = 'user' AND m.bot_id = :part
      AND (m.timestamp, m.id) > (:after_ts, :after_id)
      AND {upper}
      AND NOT EXISTS (SELECT 1 FROM sessions s
                      WHERE s.id = m.session_id AND s.bot_id = :slug)
"""
_UNATTRIBUTED_PAGE = text(_UNATTRIBUTED_SQL.format(upper="(m.timestamp, m.id) <= (:upto_ts, :upto_id)"))
_UNATTRIBUTED_WINDOW = text(_UNATTRIBUTED_SQL.format(upper="m.timestamp < :until"))

_READ = text(f"""SELECT cursor_ts, cursor_message_id, processed_rows_total
                 FROM {CURSOR_TABLE} WHERE bot_id = :slug AND user_id = :user_id""")

_INSERT = text(f"""
    INSERT INTO {CURSOR_TABLE} (bot_id, user_id, cursor_ts, cursor_message_id,
                                processed_rows_total, last_run_at, last_coverage, updated_at)
    VALUES (:slug, :user_id, :ts, :mid, :n, now(), :cov, now())
    ON CONFLICT (bot_id, user_id) DO NOTHING
    RETURNING bot_id
""").bindparams(bindparam("cov", type_=JSONB))

_CAS = text(f"""
    UPDATE {CURSOR_TABLE}
    SET cursor_ts = :ts, cursor_message_id = :mid,
        processed_rows_total = processed_rows_total + :n,
        last_run_at = now(), last_coverage = :cov, updated_at = now()
    WHERE bot_id = :slug AND user_id = :user_id
      AND cursor_ts IS NOT DISTINCT FROM :prev_ts
      AND cursor_message_id IS NOT DISTINCT FROM :prev_mid
    RETURNING bot_id
""").bindparams(bindparam("cov", type_=JSONB))

_NEG_INF = float("-inf")


def _pair(bot_id: str, excluded_bot_ids: Sequence[str]) -> tuple[str, str]:
    """(partition, canonical slug) for exactly one authorized bot; refuses before SQL."""
    if not isinstance(bot_id, str):
        raise ValueError("bot_id must be one canonical slug")
    parts = _partitions([bot_id], excluded_bot_ids)
    (part, slug), = parts.items()
    return part, slug


def _user(user_id: str) -> str:
    if not isinstance(user_id, str) or not user_id.strip() or user_id == "system":
        raise ValueError("user_id must name one real user")
    return user_id


def fetch_after(conn: Any, *, bot_id: str, user_id: str, excluded_bot_ids: Sequence[str],
                after: tuple[float, str] | None, until: float,
                limit: int = 1000) -> dict[str, Any]:
    """One page of this pair's user-role rows strictly after ``after``.

    Returns ``rows`` (deduped by (bot, id), canonical bot slug), ``last`` (the
    (ts, id) of the last fully processed row, or None), ``truncated`` (a full
    page: more rows may remain before ``until``) and
    ``unattributed_partition_rows`` (rows of this bot's partition in the same
    window with no session of this bot; not this user's rows).
    """
    part, slug = _pair(bot_id, excluded_bot_ids)
    user_id = _user(user_id)
    if limit < 1:
        raise ValueError("limit must be positive")
    after_ts, after_id = after if after is not None else (_NEG_INF, "")
    params = {"part": part, "slug": slug, "user_id": user_id,
              "after_ts": after_ts, "after_id": after_id}
    rows: dict[str, dict[str, Any]] = {}
    for r in conn.execute(_PAGE, {**params, "until": until, "limit": limit}).mappings():
        row = dict(r, bot_id=slug)
        rows.setdefault(row["id"], row)  # (bot, id) is unique: one partition
    ordered = list(rows.values())
    truncated = len(ordered) >= limit
    last = (ordered[-1]["timestamp"], ordered[-1]["id"]) if ordered else None
    # A truncated page covers only up to its last row; otherwise up to until.
    if truncated:
        unattributed = conn.execute(_UNATTRIBUTED_PAGE, {
            **params, "upto_ts": last[0], "upto_id": last[1]}).scalar_one()
    else:
        unattributed = conn.execute(_UNATTRIBUTED_WINDOW, {**params, "until": until}).scalar_one()
    return {"rows": ordered, "last": last, "truncated": truncated,
            "unattributed_partition_rows": int(unattributed)}


def _parse_meta(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for r in rows:
        if isinstance(r.get("delivery_metadata"), str):
            r["delivery_metadata"] = json.loads(r["delivery_metadata"])
    return rows


def _low_and_unknown(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    low = unknown = 0
    for r in rows:
        c = classify(r)
        low += c.confidence == "low"
        unknown += c.cls == "UNKNOWN"
    return low, unknown


class _Conflict(Exception):
    pass


def consume(engine: Any, *, bot_id: str, user_id: str, excluded_bot_ids: Sequence[str],
            until: float | None = None, limit: int = 1000,
            dry_run: bool = True, run: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Classify the next page for one pair and, unless ``dry_run``, advance its cursor.

    Read, classification and advance share one transaction. The advance is a
    compare-and-set on the cursor that was read (null-safe for a fresh or reset
    cursor; first creation is insert-on-conflict). If another run moved the
    cursor first, nothing is committed and ``status`` is ``"conflict"``.
    Dry-run (the default) writes nothing.

    ``processed_rows_total`` counts rows processed by committed runs, rescans
    included; it is a processing count, not a count of unique history rows.

    ``run`` is a :func:`guidance.begin_run` handle for this pair. Its receipt
    is finished in the same transaction as the cursor advance (committed), or
    marked failed on a conflict; ``last_coverage`` only mirrors the latest pin.
    """
    part, slug = _pair(bot_id, excluded_bot_ids)  # refuse before any SQL
    user_id = _user(user_id)
    if run is not None and (run["bot_id"], run["user_id"]) != (slug, user_id):
        raise ValueError("run handle belongs to another bot/user pair")
    until = time.time() if until is None else until
    with engine.connect() as conn:
        try:
            cur = conn.execute(_READ, {"slug": slug, "user_id": user_id}).mappings().first()
            prev = (cur["cursor_ts"], cur["cursor_message_id"]) if cur else (None, None)
            after = prev if prev[0] is not None else None
            page = fetch_after(conn, bot_id=slug, user_id=user_id,
                               excluded_bot_ids=excluded_bot_ids,
                               after=after, until=until, limit=limit)
            rows = _parse_meta(page["rows"])
            low, unknown = _low_and_unknown(rows)
            summary = report(rows)
            coverage = {
                "window": {"after": list(after) if after else None, "until": until,
                           "covered_to": list(page["last"]) if page["truncated"] else until},
                "rows_read": summary["rows_read"],
                "classes": {k: v["count"] for k, v in summary["classes"].items()},
                "unknown_rows": unknown, "low_confidence_rows": low,
                "truncated": page["truncated"],
                "unattributed_partition_rows": page["unattributed_partition_rows"],
                "excluded_bots": sorted({b.lower() for b in excluded_bot_ids}),
            }
            if run is not None:
                coverage["guidance"] = {"receipt_id": run["receipt_id"],
                                        "version": run["guidance_version"],
                                        "version_row_id": run["guidance_version_row_id"]}
            new = page["last"] or prev
            result = {"status": "dry_run" if dry_run else "advanced", "bot_id": slug,
                      "user_id": user_id, "cursor_before": list(prev) if prev[0] is not None else None,
                      "cursor_after": list(new) if new[0] is not None else None,
                      "coverage": coverage, "report": summary,
                      # Classified rows of this page, for an in-process caller
                      # (the guidance run); never written to the cursor row.
                      "rows": [{"bot_id": slug, "message_id": r["id"], "ts": r["timestamp"],
                                "content": r["content"], "classification": classify(r)}
                               for r in rows]}
            if dry_run:
                conn.rollback()
                return result
            vals = {"slug": slug, "user_id": user_id, "ts": new[0], "mid": new[1],
                    "n": summary["rows_read"], "cov": coverage}
            if cur is None:
                done = conn.execute(_INSERT, vals).first()
            else:
                done = conn.execute(_CAS, {**vals, "prev_ts": prev[0], "prev_mid": prev[1]}).first()
            if done is None:
                raise _Conflict
            if run is not None:
                from .guidance import finish_run
                finish_run(conn, handle=run, status="committed",
                           before=after, after=new if new[0] is not None else None,
                           consumed=summary["rows_read"])
            conn.commit()
            return result
        except _Conflict:
            conn.rollback()
            if run is not None:
                from .guidance import finish_run
                finish_run(conn, handle=run, status="failed", before=after, after=None,
                           consumed=None)
                conn.commit()
            return {"status": "conflict", "bot_id": slug, "user_id": user_id,
                    "cursor_before": list(prev) if prev[0] is not None else None}
        except BaseException:
            conn.rollback()
            raise


def reset_cursor(engine: Any, *, bot_id: str, user_id: str, excluded_bot_ids: Sequence[str],
                 to: tuple[float, str] | None = None) -> None:
    """Operator recovery: move the pair's cursor back (None = start) for a rescan.

    ``processed_rows_total`` is kept; a rescan adds to it as processing.
    """
    _part, slug = _pair(bot_id, excluded_bot_ids)
    user_id = _user(user_id)
    ts, mid = to if to is not None else (None, None)
    with engine.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {CURSOR_TABLE} (bot_id, user_id, cursor_ts, cursor_message_id, updated_at)
            VALUES (:slug, :user_id, :ts, :mid, now())
            ON CONFLICT (bot_id, user_id) DO UPDATE
            SET cursor_ts = EXCLUDED.cursor_ts, cursor_message_id = EXCLUDED.cursor_message_id,
                updated_at = now()
        """), {"slug": slug, "user_id": user_id, "ts": ts, "mid": mid})
