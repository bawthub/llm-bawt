"""TASK-1323: stored ``content_tsv`` for message full-text search.

Word and identifier search used to rank with ``ts_rank(to_tsvector(content))``,
re-tokenizing every matched row on every query (~1.3 s on prod). A stored
tsvector, filled by a row trigger and indexed with GIN, makes that ~60 ms.

Same contract as :mod:`session_unique_index` (TASK-1149, TASK-739):

* A freshly created ``messages`` parent gets the column, trigger and index at
  bootstrap (instant: no rows, no leaves yet).
* An existing parent is never altered at startup. Missing pieces are logged as
  an error and startup continues: writes keep working, FTS fails visibly until
  the reviewed online upgrade (:func:`upgrade_message_tsv`, run by
  ``scripts/upgrade_message_tsv.py``) has run.

The upgrade is lock-bounded and resumable: column, trigger, keyset-batched
backfill per leaf, then ``CREATE INDEX ON ONLY`` on the parent with one
``CREATE INDEX CONCURRENTLY`` per leaf, each attached to the parent index.
Objects are found structurally (catalog shape), never by name alone.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import OperationalError

logger = logging.getLogger(__name__)

PARENT = "messages"
COLUMN = "content_tsv"
VECTOR_FUNCTION = "messages_content_tsv"
FUNCTION = "messages_content_tsv_fill"
TRIGGER = "messages_content_tsv_fill"
INDEX = "messages_content_tsv_idx"
UPGRADE_COMMAND = "scripts/upgrade_message_tsv.py --phase upgrade"

# A tsvector is capped at 1 MB. Content whose vector would exceed it is
# indexed from its first FALLBACK_CHARS characters instead of failing the
# write. Prod's largest message is ~600 KB and fits whole.
FALLBACK_CHARS = 100_000

# One vector function for the trigger and the backfill. Queries compare with
# to_tsquery('english', ...), so both sides use the same configuration.
VECTOR_BODY = f"""
BEGIN
    RETURN to_tsvector('english'::regconfig, content);
EXCEPTION WHEN program_limit_exceeded THEN
    RETURN to_tsvector('english'::regconfig, left(content, {FALLBACK_CHARS}));
END
"""
FUNCTION_BODY = f"""
BEGIN
    NEW.{COLUMN} := {VECTOR_FUNCTION}(NEW.content);
    RETURN NEW;
END
"""
VECTOR_FUNCTION_SQL = (
    f"CREATE OR REPLACE FUNCTION {VECTOR_FUNCTION}(content text) RETURNS tsvector "
    f"LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE AS $fn${VECTOR_BODY}$fn$"
)
FUNCTION_SQL = (
    f"CREATE OR REPLACE FUNCTION {FUNCTION}() RETURNS trigger "
    f"LANGUAGE plpgsql AS $fn${FUNCTION_BODY}$fn$"
)
TRIGGER_SQL = (
    f"CREATE OR REPLACE TRIGGER {TRIGGER} BEFORE INSERT OR UPDATE OF content "
    f"ON {PARENT} FOR EACH ROW EXECUTE FUNCTION {FUNCTION}()"
)
COLUMN_SQL = f"ALTER TABLE {PARENT} ADD COLUMN IF NOT EXISTS {COLUMN} tsvector"

LOCK_NOT_AVAILABLE = "55P03"


@dataclass(frozen=True)
class MessageTsvState:
    """What the catalog holds for the stored-tsvector contract."""

    parent_exists: bool
    column: bool
    function: bool
    trigger: bool
    # Name of the structurally matching parent index, if any.
    index_name: str | None
    # The parent index and every attached leaf index are valid and ready.
    index_valid: bool
    # Pre-1323 expression indexes on the parent, found by structure, as
    # regclass text (schema-qualified when not on the search_path).
    legacy_indexes: tuple[str, ...] = ()
    # Attached leaf indexes that are not valid and ready.
    invalid_leaf_indexes: tuple[str, ...] = ()

    @property
    def legacy_index(self) -> bool:
        return bool(self.legacy_indexes)

    @property
    def complete(self) -> bool:
        return (self.column and self.function and self.trigger
                and self.index_name is not None and self.index_valid)

    def missing(self) -> list[str]:
        gaps = []
        if not self.column:
            gaps.append(f"column {COLUMN}")
        if not self.function:
            gaps.append(f"functions {VECTOR_FUNCTION}/{FUNCTION}")
        if not self.trigger:
            gaps.append(f"trigger {TRIGGER}")
        if self.index_name is None:
            gaps.append(f"GIN index on {COLUMN}")
        elif self.invalid_leaf_indexes:
            gaps.append(f"valid leaf indexes ({', '.join(self.invalid_leaf_indexes)} invalid)")
        elif not self.index_valid:
            gaps.append(f"valid index {self.index_name} (leaves not all attached)")
        return gaps


def _parent_oid(conn: Connection) -> int | None:
    return conn.execute(text("SELECT to_regclass(:name)::oid"), {"name": PARENT}).scalar_one()


def _tsv_index_rows(conn: Connection, table_oid: int):
    """Indexes on ``table_oid`` that are exactly ``USING gin (content_tsv)``."""
    return conn.execute(text("""
        SELECT ic.relname AS name, ic.oid AS oid, x.indisvalid AS valid,
               x.indisready AS ready
        FROM pg_index x
        JOIN pg_class ic ON ic.oid = x.indexrelid
        JOIN pg_am am ON am.oid = ic.relam
        JOIN pg_attribute a ON a.attrelid = x.indrelid AND a.attnum = x.indkey[0]
        WHERE x.indrelid = :table
          AND am.amname = 'gin'
          AND x.indnatts = 1
          AND x.indexprs IS NULL
          AND x.indpred IS NULL
          AND a.attname = :column
        ORDER BY ic.relname
    """), {"table": table_oid, "column": COLUMN}).mappings().all()


# pg_get_expr deparse of the pre-1323 index expression (48837b5).
LEGACY_EXPRESSION = "to_tsvector('english'::regconfig, content)"


def _legacy_indexes(conn: Connection, parent_oid: int) -> tuple[str, ...]:
    """Parent indexes that are exactly ``USING gin (to_tsvector('english', content))``.

    Matched on the owning table and the expression, never on the name, so an
    unrelated index that happens to share the old name is never touched.
    """
    return tuple(conn.execute(text("""
        SELECT x.indexrelid::regclass::text
        FROM pg_index x
        JOIN pg_class ic ON ic.oid = x.indexrelid
        JOIN pg_am am ON am.oid = ic.relam
        WHERE x.indrelid = :parent
          AND am.amname = 'gin'
          AND x.indnatts = 1
          AND x.indkey[0] = 0
          AND x.indpred IS NULL
          AND pg_get_expr(x.indexprs, x.indrelid) = :expression
        ORDER BY 1
    """), {"parent": parent_oid, "expression": LEGACY_EXPRESSION}).scalars().all())


def _invalid_leaf_indexes(conn: Connection, parent_index_oid: int) -> tuple[str, ...]:
    """Leaf indexes attached under the parent index that are not valid and ready."""
    return tuple(conn.execute(text("""
        SELECT ic.relname
        FROM pg_partition_tree(CAST(:index AS regclass)) t
        JOIN pg_index x ON x.indexrelid = t.relid
        JOIN pg_class ic ON ic.oid = t.relid
        WHERE t.isleaf AND NOT (x.indisvalid AND x.indisready)
        ORDER BY 1
    """), {"index": parent_index_oid}).scalars().all())


def _function_body(conn: Connection, signature: str) -> str | None:
    body = conn.execute(text(
        "SELECT prosrc FROM pg_proc WHERE oid = to_regprocedure(:signature)"
    ), {"signature": signature}).scalar_one_or_none()
    return body.strip() if body is not None else None


def message_tsv_state(conn: Connection) -> MessageTsvState:
    """Read-only catalog check of the column, function, trigger and index."""
    parent = _parent_oid(conn)
    if parent is None:
        return MessageTsvState(False, False, False, False, None, False)
    column = conn.execute(text("""
        SELECT format_type(atttypid, atttypmod) FROM pg_attribute
        WHERE attrelid = :parent AND attname = :column AND NOT attisdropped
    """), {"parent": parent, "column": COLUMN}).scalar_one_or_none()
    if column not in (None, "tsvector"):
        raise RuntimeError(f"{PARENT}.{COLUMN} exists with type {column}, not tsvector")
    function = (_function_body(conn, f"{VECTOR_FUNCTION}(text)") == VECTOR_BODY.strip()
                and _function_body(conn, f"{FUNCTION}()") == FUNCTION_BODY.strip())
    trigger = conn.execute(text("""
        SELECT t.tgenabled <> 'D' AND p.proname = :function
               AND (t.tgtype & 1) = 1   -- FOR EACH ROW
               AND (t.tgtype & 2) = 2   -- BEFORE
               AND (t.tgtype & 4) = 4   -- INSERT
               AND (t.tgtype & 16) = 16 -- UPDATE
               -- UPDATE OF content only. int2vector is 0-based: index, don't
               -- compare it with a 1-based array.
               AND cardinality(t.tgattr::int2[]) = 1
               AND t.tgattr[0] = (SELECT attnum FROM pg_attribute
                                  WHERE attrelid = :parent AND attname = 'content')
        FROM pg_trigger t JOIN pg_proc p ON p.oid = t.tgfoid
        WHERE t.tgrelid = :parent AND t.tgname = :trigger AND NOT t.tgisinternal
    """), {"parent": parent, "function": FUNCTION, "trigger": TRIGGER}).scalar_one_or_none()
    indexes = _tsv_index_rows(conn, parent) if column else []
    index = indexes[0] if indexes else None
    invalid_leaves = _invalid_leaf_indexes(conn, index["oid"]) if index else ()
    return MessageTsvState(
        parent_exists=True,
        column=column is not None,
        function=function,
        trigger=bool(trigger),
        index_name=index["name"] if index else None,
        index_valid=bool(index and index["valid"] and index["ready"] and not invalid_leaves),
        legacy_indexes=_legacy_indexes(conn, parent),
        invalid_leaf_indexes=invalid_leaves,
    )


def ensure_message_tsv(conn: Connection, *, table_was_new: bool) -> MessageTsvState:
    """Bootstrap a new parent; only report on an existing one.

    Startup never runs blocking DDL on an existing ``messages`` (TASK-739).
    It does not raise either: message writes must keep working; FTS queries
    fail until the operator runs the reviewed upgrade.
    """
    state = message_tsv_state(conn)
    if state.complete:
        return state
    if not table_was_new:
        logger.error(
            "%s lacks stored-tsvector search (%s); full-text message search "
            "fails until `%s` runs (TASK-1323)",
            PARENT, ", ".join(state.missing()), UPGRADE_COMMAND,
        )
        return state
    conn.execute(text(COLUMN_SQL))
    conn.execute(text(VECTOR_FUNCTION_SQL))
    conn.execute(text(FUNCTION_SQL))
    conn.execute(text(TRIGGER_SQL))
    if state.index_name is None:
        conn.execute(text(f"CREATE INDEX {INDEX} ON {PARENT} USING gin ({COLUMN})"))
    state = message_tsv_state(conn)
    if not state.complete:
        raise RuntimeError(f"fresh {PARENT} stored tsvector did not validate: {state.missing()}")
    return state


@dataclass
class MessageTsvUpgradeReport:
    column_added: bool = False
    trigger_installed: bool = False
    rows_backfilled: dict[str, int] = field(default_factory=dict)
    parent_index_created: bool = False
    leaf_indexes_built: list[str] = field(default_factory=list)
    leaf_indexes_attached: list[str] = field(default_factory=list)
    # Attached leaf indexes found invalid and rebuilt in place.
    leaf_indexes_reindexed: list[str] = field(default_factory=list)
    invalid_indexes_dropped: list[str] = field(default_factory=list)


def leaf_index_name(leaf: str) -> str:
    """Stable per-leaf name within the 63-byte identifier limit."""
    return "mtsv_" + hashlib.sha256(leaf.encode()).hexdigest()[:16]


def _quote(conn: Connection, name: str) -> str:
    return conn.execute(text("SELECT quote_ident(:name)"), {"name": name}).scalar_one()


class _BoundedSession:
    """AUTOCOMMIT connection with temporary lock/statement timeouts.

    Restores the caller's settings rather than RESET, which would drop the
    app's on-connect limits; a connection that can't be restored is
    invalidated so it never returns to the pool unbounded.
    """

    def __init__(self, engine: Engine, *, lock_timeout: str, statement_timeout: str):
        self.engine = engine
        self.lock_timeout = lock_timeout
        self.statement_timeout = statement_timeout

    def __enter__(self) -> Connection:
        self.conn = self.engine.connect().execution_options(isolation_level="AUTOCOMMIT")
        self.previous = self.conn.execute(text(
            "SELECT current_setting('lock_timeout'), current_setting('statement_timeout')"
        )).one()
        self.conn.execute(text("SELECT set_config('lock_timeout', :v, false)"),
                          {"v": self.lock_timeout})
        self.conn.execute(text("SELECT set_config('statement_timeout', :v, false)"),
                          {"v": self.statement_timeout})
        return self.conn

    def __exit__(self, exc_type, exc, _tb) -> None:
        try:
            self.conn.execute(text("SELECT set_config('lock_timeout', :v, false)"),
                              {"v": self.previous[0]})
            self.conn.execute(text("SELECT set_config('statement_timeout', :v, false)"),
                              {"v": self.previous[1]})
        except BaseException as restore_error:
            self.conn.invalidate(restore_error)
            if exc is not None:
                exc.add_note(f"Session timeout restoration failed; connection invalidated: {restore_error}")
            else:
                raise
        finally:
            self.conn.close()


def _with_lock_retry(conn: Connection, sql: str, *, attempts: int, pause: float) -> None:
    """Run lock-taking DDL; on lock_timeout, back off and retry."""
    for attempt in range(1, attempts + 1):
        try:
            conn.execute(text(sql))
            return
        except OperationalError as exc:
            if getattr(exc.orig, "pgcode", None) != LOCK_NOT_AVAILABLE or attempt == attempts:
                raise
            logger.warning("lock timeout on %r (attempt %d/%d); retrying", sql[:60], attempt, attempts)
            time.sleep(pause * attempt)


def _leaves(conn: Connection) -> list[tuple[int, str]]:
    return [tuple(row) for row in conn.execute(text("""
        SELECT c.oid, c.relname FROM pg_partition_tree(CAST(:parent AS regclass)) t
        JOIN pg_class c ON c.oid = t.relid
        WHERE t.isleaf ORDER BY c.relname
    """), {"parent": PARENT}).all()]


def _backfill_leaf(conn: Connection, leaf: str, batch_size: int) -> int:
    """Fill NULL vectors in primary-key order, one short statement per batch.

    ``content`` is not in the SET list, so the trigger does not re-fire.
    Concurrent content edits are safe: the trigger fills them, and a racing
    batch re-evaluates the newest row version.
    """
    table = _quote(conn, leaf)
    after: tuple[str, str] | None = None
    total = 0
    while True:
        lower = "WHERE (bot_id, id) > (:after_bot, :after_id)" if after else ""
        bound = dict(zip(("after_bot", "after_id"), after)) if after else {}
        upper = conn.execute(text(f"""
            SELECT bot_id, id FROM {table} {lower}
            ORDER BY bot_id, id OFFSET :skip LIMIT 1
        """), {**bound, "skip": batch_size - 1}).one_or_none()
        clauses = [f"{COLUMN} IS NULL"]
        if after:
            clauses.append("(bot_id, id) > (:after_bot, :after_id)")
        if upper:
            clauses.append("(bot_id, id) <= (:upper_bot, :upper_id)")
            bound.update(upper_bot=upper[0], upper_id=upper[1])
        result = conn.execute(text(
            f"UPDATE {table} SET {COLUMN} = {VECTOR_FUNCTION}(content) WHERE {' AND '.join(clauses)}"
        ), bound)
        total += result.rowcount
        if upper is None:
            return total
        after = (upper[0], upper[1])


def _attached_children(conn: Connection, parent_index_oid: int) -> set[int]:
    return set(conn.execute(text(
        "SELECT inhrelid FROM pg_inherits WHERE inhparent = :parent"
    ), {"parent": parent_index_oid}).scalars().all())


def upgrade_message_tsv(
    engine: Engine,
    *,
    batch_size: int = 1000,
    lock_timeout: str = "2s",
    statement_timeout: str = "10min",
    lock_attempts: int = 10,
    lock_pause: float = 1.0,
) -> MessageTsvUpgradeReport:
    """Online upgrade of an existing ``messages`` parent. Never run at startup.

    Idempotent and resumable: each step checks the catalog first, so a rerun
    after any failure continues where it stopped. An INVALID leaf index left
    by a failed concurrent build is dropped concurrently and rebuilt; an
    attached leaf index that is not valid is rebuilt in place with
    ``REINDEX INDEX CONCURRENTLY``, which keeps its attachment.

    Every leaf is covered, private bots' included (Nick, 2026-10-24: fine as
    long as the data never reaches a model; the column is only read inside
    SQL for matching and ranking).
    """
    report = MessageTsvUpgradeReport()
    with _BoundedSession(engine, lock_timeout=lock_timeout,
                         statement_timeout=statement_timeout) as conn:
        state = message_tsv_state(conn)
        if not state.parent_exists:
            raise RuntimeError(f"{PARENT} does not exist; nothing to upgrade")
        if not state.column:
            _with_lock_retry(conn, COLUMN_SQL, attempts=lock_attempts, pause=lock_pause)
            report.column_added = True
        if not state.function or not state.trigger:
            conn.execute(text(VECTOR_FUNCTION_SQL))
            conn.execute(text(FUNCTION_SQL))
            _with_lock_retry(conn, TRIGGER_SQL, attempts=lock_attempts, pause=lock_pause)
            report.trigger_installed = True
        # Trigger first, then backfill: every row is either written after the
        # trigger exists or still NULL for the backfill to find.
        for _oid, leaf in _leaves(conn):
            report.rows_backfilled[leaf] = _backfill_leaf(conn, leaf, batch_size)

        parent = _parent_oid(conn)
        parent_indexes = _tsv_index_rows(conn, parent)
        if parent_indexes:
            parent_index = parent_indexes[0]
        else:
            _with_lock_retry(
                conn, f"CREATE INDEX {INDEX} ON ONLY {PARENT} USING gin ({COLUMN})",
                attempts=lock_attempts, pause=lock_pause)
            report.parent_index_created = True
            parent_index = _tsv_index_rows(conn, parent)[0]
        quoted_parent_index = _quote(conn, parent_index["name"])

        for leaf_oid, leaf in _leaves(conn):
            attached = _attached_children(conn, parent_index["oid"])
            candidates = _tsv_index_rows(conn, leaf_oid)
            # Unattached INVALID leftovers (a failed CIC or REINDEX's _ccnew).
            for row in candidates:
                if row["oid"] not in attached and not (row["valid"] and row["ready"]):
                    conn.execute(text(f"DROP INDEX CONCURRENTLY IF EXISTS {_quote(conn, row['name'])}"))
                    report.invalid_indexes_dropped.append(row["name"])
            current = [row for row in candidates if row["oid"] in attached]
            if current:
                if not (current[0]["valid"] and current[0]["ready"]):
                    conn.execute(text(f"REINDEX INDEX CONCURRENTLY {_quote(conn, current[0]['name'])}"))
                    report.leaf_indexes_reindexed.append(current[0]["name"])
                continue
            usable = [row for row in candidates
                      if row["oid"] not in attached and row["valid"] and row["ready"]]
            if usable:
                name = usable[0]["name"]
            else:
                name = leaf_index_name(leaf)
                conn.execute(text(
                    f"CREATE INDEX CONCURRENTLY {_quote(conn, name)} "
                    f"ON {_quote(conn, leaf)} USING gin ({COLUMN})"))
                report.leaf_indexes_built.append(name)
            _with_lock_retry(
                conn, f"ALTER INDEX {quoted_parent_index} ATTACH PARTITION {_quote(conn, name)}",
                attempts=lock_attempts, pause=lock_pause)
            report.leaf_indexes_attached.append(name)

        nulls = conn.execute(text(
            f"SELECT count(*) FROM {PARENT} WHERE {COLUMN} IS NULL")).scalar_one()
        state = message_tsv_state(conn)
        if nulls or not state.complete:
            raise RuntimeError(
                f"stored tsvector upgrade did not validate: {nulls} NULL rows, "
                f"missing {state.missing()}")
    return report


def drop_legacy_fts_index(
    engine: Engine,
    *,
    lock_timeout: str = "2s",
    lock_attempts: int = 10,
    lock_pause: float = 1.0,
) -> bool:
    """Drop the pre-1323 expression index(es) once nothing queries them.

    Run only after the app restart that loads the ``content_tsv`` queries. A
    partitioned index cannot be dropped concurrently; the drop is a short
    catalog change, bounded by ``lock_timeout`` and retried.
    """
    with _BoundedSession(engine, lock_timeout=lock_timeout,
                         statement_timeout="1min") as conn:
        state = message_tsv_state(conn)
        if not state.complete:
            raise RuntimeError("refusing to drop the legacy FTS index: stored tsvector "
                               f"incomplete ({state.missing()})")
        if not state.legacy_indexes:
            return False
        parent = _parent_oid(conn)
        dropped = False
        for name in state.legacy_indexes:
            # Names are regclass text from the structural match (owning table
            # and expression); recheck right before each drop so a rename or
            # replacement since the state read is never dropped by mistake.
            if name not in _legacy_indexes(conn, parent):
                continue
            _with_lock_retry(conn, f"DROP INDEX IF EXISTS {name}",
                             attempts=lock_attempts, pause=lock_pause)
            dropped = True
        return dropped
