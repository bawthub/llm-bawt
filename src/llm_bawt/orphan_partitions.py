"""Bounds-based, read-only orphan planning and digest-confirmed partition execution.

This module never selects candidates from a table-name suffix. A manifest is a
confirmation token, not authorization or a backup receipt. Operational backup
and run approval remain an operator responsibility outside this API.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from .bot_tags.vocabulary import BehaviorTag
from .memory.postgresql import PARENT_TABLES, _sanitize_table_name, partition_name

PLAN_VERSION = 2
POLICY_VERSION = "partition-orphans-v3"
_BOUND = re.compile(r"^FOR VALUES IN \('((?:[^']|'')*)'\)$")


# Bots tagged private, plus the legacy column fallback (BOT_TAGS.md §6). The
# first branch deliberately skips bot_profiles: a private assignment whose
# profile was deleted is exactly the storage being protected.
_PROTECTED_SQL = text("""
    SELECT bot_slug FROM bot_profile_tags WHERE tag_slug = :private
    UNION
    SELECT lower(btrim(slug)) FROM bot_profiles WHERE NOT include_in_global_search
""")


def read_protected_slugs(conn) -> frozenset[str]:
    """Private identities; a missing table or failed read raises (fail closed)."""
    return frozenset(row[0] for row in conn.execute(
        _PROTECTED_SQL, {"private": str(BehaviorTag.PRIVATE)}))


def protected_storage_ids(conn) -> frozenset[str]:
    return frozenset(_sanitize_table_name(slug) for slug in read_protected_slugs(conn))


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _qualified(schema: str, name: str) -> str:
    """Quote catalog-sourced identifiers, never manifest/request identifiers."""
    return '.'.join('"' + part.replace('"', '""') + '"' for part in (schema, name))


def _storage_bound(raw: str) -> str:
    match = _BOUND.fullmatch(raw)
    if match is None:
        raise ValueError(f"Ambiguous or unsupported LIST partition bound: {raw!r}")
    value = match.group(1).replace("''", "'")
    if not value or _sanitize_table_name(value) != value:
        raise ValueError(f"Partition bound is not a sanitized storage identity: {raw!r}")
    return value


@dataclass(frozen=True)
class PartitionChild:
    parent_oid: int
    parent: str
    child_oid: int
    child: str
    bound: str
    storage_id: str
    relkind: str
    persistence: str
    row_count: int | None
    incoming_fks: int
    disposition: str
    reason: str | None
    action: str | None


@dataclass(frozen=True)
class PartitionPlan:
    schema_version: int
    policy_version: str
    source_revision: str
    database: str
    database_oid: int
    cluster_id: str
    server_version: int
    max_identifier_length: int
    parent_oids: dict[str, int]
    registry_digest: str
    catalog_digest: str
    protected_ids: tuple[str, ...]
    children: tuple[PartitionChild, ...]

    def content(self) -> dict[str, Any]:
        # Normalize tuple values to lists, just as JSON on the wire does.
        return json.loads(canonical_bytes(asdict(self)))

    @property
    def content_digest(self) -> str:
        return digest(self.content())

    def manifest(self) -> dict[str, Any]:
        children = self.content()["children"]
        return {
            "content": self.content(), "sha256": self.content_digest,
            "counts": {
                "proposed": sum(c["disposition"] == "proposed" for c in children),
                "held": sum(c["disposition"] == "held" for c in children),
                "nonempty": sum(c["disposition"] == "proposed" and c["row_count"] > 0 for c in children),
            },
            "lock_impact": "NOWAIT SHARE on bot_profiles and bot_profile_tags, then ACCESS EXCLUSIVE on ONLY one parent and one child per transaction",
        }


class OrphanPartitionPlanner:
    """Capture one consistent catalog and registered-bot view without mutations."""

    def __init__(self, engine: Engine):
        self.engine = engine

    @staticmethod
    def _identity(conn) -> dict[str, Any]:
        return dict(conn.execute(text("""
            SELECT current_database() AS db,
                   (SELECT oid FROM pg_database WHERE datname=current_database()) AS db_oid,
                   (pg_control_system()).system_identifier::text AS cluster_id,
                   current_setting('server_version_num')::int AS server_version,
                   current_setting('max_identifier_length')::int AS max_name
        """)).mappings().one())

    @staticmethod
    def _parents(conn) -> dict[str, dict[str, Any]]:
        rows = conn.execute(text("""
            SELECT c.oid::bigint AS oid, c.relname, c.relkind, n.nspname
            FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
            WHERE n.nspname=current_schema() AND c.relname = ANY(:names)
        """), {"names": list(PARENT_TABLES)}).mappings().all()
        if (len(rows) != len(PARENT_TABLES) or
                {row["relname"] for row in rows} != set(PARENT_TABLES) or
                any(row["relkind"] != "p" for row in rows)):
            raise ValueError("Expected three partitioned parents in the current schema")
        return {row["relname"]: dict(row) for row in rows}

    @staticmethod
    def _registry(conn) -> dict[str, str]:
        registry: dict[str, str] = {}
        for (raw_slug,) in conn.execute(text("SELECT slug FROM bot_profiles ORDER BY slug")):
            slug = raw_slug.strip().lower()
            if not slug:
                raise ValueError("Blank registered bot slug")
            storage = _sanitize_table_name(slug)
            if storage in registry and registry[storage] != slug:
                raise ValueError(f"Registered bot storage identity collision: {storage}")
            registry[storage] = slug
        return registry

    @staticmethod
    def _children(conn, parents: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        rows = conn.execute(text("""
            SELECT p.oid::bigint AS parent_oid, pn.nspname AS parent_schema,
                   p.relname AS parent_name, c.oid::bigint AS child_oid,
                   cn.nspname AS child_schema, c.relname AS child_name,
                   c.relkind, c.relpersistence,
                   pg_get_expr(c.relpartbound, c.oid) AS bound,
                   (SELECT count(*) FROM pg_constraint k WHERE k.confrelid=c.oid) AS incoming_fks
            FROM pg_inherits i JOIN pg_class p ON p.oid=i.inhparent
            JOIN pg_namespace pn ON pn.oid=p.relnamespace
            JOIN pg_class c ON c.oid=i.inhrelid
            JOIN pg_namespace cn ON cn.oid=c.relnamespace
            WHERE p.oid = ANY(:parents)
            ORDER BY p.relname, c.relname
        """), {"parents": [row["oid"] for row in parents.values()]}).mappings().all()
        if any(row["child_schema"] != row["parent_schema"] for row in rows):
            raise ValueError("Child partition belongs to a different schema")
        unattached = conn.execute(text("""
            SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
            WHERE n.nspname=current_schema() AND c.relkind IN ('r','p')
              AND (c.relname LIKE 'messages_p_%' OR c.relname LIKE 'memories_p_%'
                   OR c.relname LIKE 'forgotten_messages_p_%')
              AND NOT EXISTS (SELECT 1 FROM pg_inherits i WHERE i.inhrelid=c.oid)
            LIMIT 1
        """)).first()
        if unattached:
            raise ValueError(f"Detached or unrecognized partition-like table: {unattached[0]}")
        return [dict(row) for row in rows]

    @staticmethod
    def _validate_child(row: dict[str, Any], parents: dict[str, dict[str, Any]],
                        max_name: int) -> str:
        parent = parents.get(row["parent_name"])
        if (parent is None or row["parent_oid"] != parent["oid"] or
                row["parent_schema"] != parent["nspname"] or
                row["child_schema"] != parent["nspname"] or
                row["relkind"] != "r" or row["relpersistence"] != "p"):
            raise ValueError("Wrong attachment, relation type or schema for partition")
        storage = _storage_bound(row["bound"])
        if row["child_name"] != partition_name(row["parent_name"], storage)[:max_name]:
            raise ValueError("Partition name/bound identity drift")
        return storage

    def plan(self) -> PartitionPlan:
        with self.engine.connect() as conn:
            conn.exec_driver_sql("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            try:
                identity = self._identity(conn)
                parents = self._parents(conn)
                registry = self._registry(conn)
                protected_ids = protected_storage_ids(conn)
                raw = self._children(conn, parents)
                max_name = int(identity["max_name"])
                seen: dict[tuple[str, str], str] = {}
                # Include registered slugs as well as the observed bounds: truncation
                # can make two distinct storage IDs look like the same child name.
                for parent in PARENT_TABLES:
                    for bound in [*registry, *(_storage_bound(r["bound"]) for r in raw
                                                if r["parent_name"] == parent)]:
                        name = partition_name(parent, bound)[:max_name]
                        key = (parent, name)
                        if key in seen and seen[key] != bound:
                            raise ValueError(f"Truncated partition-name collision: {name}")
                        seen[key] = bound
                children: list[PartitionChild] = []
                for row in raw:
                    storage = self._validate_child(row, parents, max_name)
                    parent = row["parent_name"]
                    # Inspect both identities BEFORE accessing child rows. A protected
                    # name paired with any other bound is ambiguous and stops planning.
                    protected_name = any(row["child_name"] == partition_name(parent, p)[:max_name]
                                         for p in protected_ids)
                    if protected_name != (storage in protected_ids):
                        raise ValueError("Protected partition name/bound mismatch")
                    protected = protected_name
                    child = _qualified(row["child_schema"], row["child_name"])
                    count = None if protected else int(conn.execute(text(
                        f"SELECT count(*) FROM ONLY {child}"
                    )).scalar_one())
                    incoming = int(row["incoming_fks"])
                    reason = ("protected identity" if protected else
                              "registered bot" if storage in registry else
                              "incoming foreign keys" if incoming else None)
                    children.append(PartitionChild(
                        parent_oid=int(row["parent_oid"]),
                        parent=_qualified(row["parent_schema"], parent),
                        child_oid=int(row["child_oid"]), child=child,
                        bound=row["bound"], storage_id=storage,
                        relkind=row["relkind"], persistence=row["relpersistence"],
                        row_count=count, incoming_fks=incoming,
                        disposition="held" if reason else "proposed", reason=reason,
                        action=None if reason else ("truncate-then-drop-RESTRICT" if count else
                                                    "drop-empty-RESTRICT"),
                    ))
                catalog = [tuple(sorted(row.items())) for row in raw]
                return PartitionPlan(
                    schema_version=PLAN_VERSION, policy_version=POLICY_VERSION,
                    source_revision=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    database=identity["db"], database_oid=int(identity["db_oid"]),
                    cluster_id=identity["cluster_id"],
                    server_version=int(identity["server_version"]),
                    max_identifier_length=max_name,
                    parent_oids={name: int(row["oid"]) for name, row in parents.items()},
                    registry_digest=digest(registry), catalog_digest=digest(catalog),
                    protected_ids=tuple(sorted(protected_ids)),
                    children=tuple(children),
                )
            finally:
                conn.rollback()


class PartitionExecutionHalted(RuntimeError):
    """An execution stopped; earlier child transactions may have committed."""

    def __init__(self, outcomes: list[dict[str, Any]], journal_path: Path, *,
                 expected_children: int, journal_error: str | None = None):
        self.outcomes = outcomes
        self.journal_path = journal_path
        self.expected_children = expected_children
        self.journal_error = journal_error
        super().__init__("Partition cleanup halted; inspect the journal before re-previewing")

    def receipt(self) -> dict[str, Any]:
        committed = [item for item in self.outcomes if item["status"] == "committed"]
        unresolved = [item for item in self.outcomes if item["status"] != "committed"]
        status = ("unknown" if any(item["status"] == "unknown" for item in unresolved) else
                  "completed" if len(committed) == self.expected_children else
                  "partial" if committed else "halted")
        receipt = {
            "dry_run": False,
            "status": status,
            "outcomes": self.outcomes,
            "committed_count": len(committed),
            "journal": str(self.journal_path),
            "journal_incomplete": self.journal_error is not None,
            "journal_error": self.journal_error,
        }
        if unresolved:
            receipt["failed_child"] = unresolved[-1]["child"]
            receipt["error"] = unresolved[-1].get("error")
        if self.journal_error:
            receipt["warning"] = "Journal incomplete; reconcile the receipt with the catalog before retrying."
        return receipt


def _append_journal(fd: int, record: dict[str, Any]) -> None:
    """Write and sync one JSONL record without a close-time userspace buffer."""
    line = json.dumps(record, sort_keys=True).encode("utf-8") + b"\n"
    view = memoryview(line)
    while view:
        written = os.write(fd, view)
        if written == 0:
            raise OSError(errno.EIO, "Journal write made no progress")
        view = view[written:]
    os.fsync(fd)


class PartitionManifestExecutor:
    """Execute a freshly planned digest, journaling each short transaction."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def execute(self, plan: PartitionPlan, confirmed_digest: str, *,
                journal_path: Path) -> list[dict[str, Any]]:
        # Re-plan before creating a journal or mutating any child. Timestamp-free
        # canonical content ensures this digest is a meaningful compare token.
        fresh = OrphanPartitionPlanner(self.engine).plan()
        if (confirmed_digest != plan.content_digest or fresh.content_digest != confirmed_digest):
            raise ValueError("Manifest digest mismatch; preview again")
        entries = [child for child in fresh.children if child.disposition == "proposed"]
        if not entries:
            return []
        journal_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        outcomes: list[dict[str, Any]] = []
        fd = os.open(journal_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_APPEND, 0o600)
        primary_error: BaseException | None = None
        journal_error: str | None = None
        try:
            dir_fd = os.open(journal_path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
            for child_plan in entries:
                commit_pending = False
                committed = False
                try:
                    # Durable intent precedes DDL. An interrupted commit leaves
                    # a pending entry requiring reconciliation, never replay.
                    intent = {"digest": confirmed_digest, "child_oid": child_plan.child_oid,
                              "child": child_plan.child, "status": "pending",
                              "at": datetime.now(timezone.utc).isoformat()}
                    try:
                        _append_journal(fd, intent)
                    except OSError as exc:
                        journal_error = str(exc)
                        raise
                    with self.engine.begin() as conn:
                        conn.exec_driver_sql("SET LOCAL lock_timeout = '1s'")
                        conn.exec_driver_sql("SET LOCAL statement_timeout = '5s'")
                        identity = OrphanPartitionPlanner._identity(conn)
                        if (identity["db"] != fresh.database or
                                int(identity["db_oid"]) != fresh.database_oid or
                                identity["cluster_id"] != fresh.cluster_id or
                                int(identity["server_version"]) != fresh.server_version or
                                int(identity["max_name"]) != fresh.max_identifier_length):
                            raise ValueError("Database or cluster identity drift")
                        parents = OrphanPartitionPlanner._parents(conn)
                        if {name: int(row["oid"]) for name, row in parents.items()} != fresh.parent_oids:
                            raise ValueError("Parent identity drift")
                        # BotProfileStore.upsert commits the registry row before
                        # partition provisioning takes a parent lock. Mirror that
                        # order and fence all registry writes through this commit.
                        registry_sql = _qualified(next(iter(parents.values()))["nspname"], "bot_profiles")
                        conn.exec_driver_sql(f"LOCK TABLE {registry_sql} IN SHARE MODE NOWAIT")
                        # Same fence on private assignments, then re-read the
                        # protected set under the lock: any change is drift.
                        tags_sql = _qualified(next(iter(parents.values()))["nspname"], "bot_profile_tags")
                        conn.exec_driver_sql(f"LOCK TABLE {tags_sql} IN SHARE MODE NOWAIT")
                        protected_now = protected_storage_ids(conn)
                        if protected_now != frozenset(fresh.protected_ids):
                            raise ValueError("Protected identity set changed after preview")
                        catalog = OrphanPartitionPlanner._children(conn, parents)
                        matches = [r for r in catalog if int(r["child_oid"]) == child_plan.child_oid]
                        if len(matches) != 1:
                            raise ValueError("Child is missing, detached or ambiguous")
                        current = matches[0]
                        # Validate catalog data BEFORE locking. Never interpolate
                        # child_plan.parent/child from a manifest into SQL.
                        bound = OrphanPartitionPlanner._validate_child(
                            current, parents, fresh.max_identifier_length)
                        parent_sql = _qualified(current["parent_schema"], current["parent_name"])
                        child_sql = _qualified(current["child_schema"], current["child_name"])
                        if (parent_sql != child_plan.parent or child_sql != child_plan.child or
                                int(current["parent_oid"]) != child_plan.parent_oid or
                                current["bound"] != child_plan.bound or bound != child_plan.storage_id or
                                int(current["incoming_fks"]) != child_plan.incoming_fks or
                                bound in protected_now):
                            raise ValueError("Child partition identity or dependency drift")
                        # ONLY avoids locking all descendants of the parent.
                        conn.exec_driver_sql(f"LOCK TABLE ONLY {parent_sql} IN ACCESS EXCLUSIVE MODE NOWAIT")
                        conn.exec_driver_sql(f"LOCK TABLE {child_sql} IN ACCESS EXCLUSIVE MODE NOWAIT")
                        # Re-read catalog and registry after obtaining both locks.
                        locked = OrphanPartitionPlanner._children(conn, parents)
                        selected = [r for r in locked if int(r["child_oid"]) == child_plan.child_oid]
                        if len(selected) != 1 or selected[0] != current or bound in OrphanPartitionPlanner._registry(conn):
                            raise ValueError("Child changed or bot registered after preview")
                        count = int(conn.execute(text(f"SELECT count(*) FROM ONLY {child_sql}")).scalar_one())
                        if count != child_plan.row_count:
                            raise ValueError("Child row count changed after preview")
                        if child_plan.action == "truncate-then-drop-RESTRICT" and count > 0:
                            conn.exec_driver_sql(f"TRUNCATE TABLE ONLY {child_sql} RESTRICT")
                        elif child_plan.action != "drop-empty-RESTRICT" or count != 0:
                            raise ValueError("Manifest action does not match child rows")
                        conn.exec_driver_sql(f"DROP TABLE {child_sql} RESTRICT")
                        # A failure from context exit can mean the COMMIT reply
                        # was lost. A DDL error before this point rolls back.
                        commit_pending = True
                    committed = True
                    outcome = {"digest": confirmed_digest, "child_oid": child_plan.child_oid,
                               "child": child_plan.child, "row_count": count,
                               "action": child_plan.action, "status": "committed",
                               "at": datetime.now(timezone.utc).isoformat()}
                    # The transaction committed even if its receipt cannot be
                    # synced. Keep that fact in the outward partial receipt.
                    outcomes.append(outcome)
                    try:
                        _append_journal(fd, outcome)
                    except OSError as exc:
                        journal_error = str(exc)
                        outcome["journal_error"] = journal_error
                        raise
                except Exception as exc:
                    if not committed:
                        # Only an exception while leaving the successful DDL
                        # block has an unknown COMMIT result. An earlier SQL
                        # error rolls back inside engine.begin().
                        outcome = {"digest": confirmed_digest, "child_oid": child_plan.child_oid,
                                   "child": child_plan.child,
                                   "status": "unknown" if commit_pending else "halted",
                                   "error": str(exc),
                                   "at": datetime.now(timezone.utc).isoformat()}
                        if journal_error is None:
                            try:
                                _append_journal(fd, outcome)
                            except OSError as journal_exc:
                                journal_error = str(journal_exc)
                                outcome["journal_error"] = journal_error
                        outcomes.append(outcome)
                    raise PartitionExecutionHalted(
                        outcomes, journal_path, expected_children=len(entries),
                        journal_error=journal_error,
                    ) from exc
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                os.close(fd)
            except OSError as exc:
                close_error = f"Journal close failed: {exc}"
                if primary_error is not None:
                    primary_error.add_note(close_error)
                    if isinstance(primary_error, PartitionExecutionHalted):
                        primary_error.journal_error = (
                            f"{primary_error.journal_error}; {close_error}"
                            if primary_error.journal_error else close_error
                        )
                else:
                    raise PartitionExecutionHalted(
                        outcomes, journal_path, expected_children=len(entries),
                        journal_error=close_error,
                    ) from exc
        return outcomes
