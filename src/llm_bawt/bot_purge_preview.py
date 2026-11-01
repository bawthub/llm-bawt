"""Read-only, advisory impact snapshot for the current standalone bot purge.

Never call the executor or a store constructor here: many stores bootstrap DDL.
The digest describes observed scope/counts, NOT row preimages, a confirmation
capability, writer admission, or a promise that a later purge can commit.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from .bot_purge_inventory import DELETED, OWNER_SOURCES, RETAINED, SCOPE_VERSION, UNKNOWN
from .utils.bot_partition import sanitize_table_name


class PreviewUnavailable(RuntimeError):
    """Local snapshot cannot be read; no writes were attempted."""


def normalize_slug(slug: str) -> str:
    normalized = slug.strip().lower()
    if not normalized or len(normalized) > 128 or not normalized.isprintable():
        raise ValueError("A nonblank printable bot slug of at most 128 characters is required")
    return normalized


def _json_value(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"Unsupported fingerprint type: {type(value).__name__}")


def fingerprint(value) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=True, allow_nan=False, default=_json_value)
    return hashlib.sha256(serialized.encode()).hexdigest()


def collect_preview(conn, slug: str) -> dict:
    """Collect on a caller-owned snapshot; SELECT/reflection only, no commit.

    Production callers use preview_bot_purge() for enforced PostgreSQL RR/RO.
    Connection injection keeps fixture collection independent of application
    startup, profile existence and service configuration.
    """
    slug = normalize_slug(slug)
    storage = sanitize_table_name(slug)
    params = {"slug": slug, "storage": storage}
    inspector = inspect(conn)
    tables = set(inspector.get_table_names())
    columns = {}

    def available(table, required):
        if table not in tables:
            return False
        if table not in columns:
            columns[table] = {c["name"] for c in inspector.get_columns(table)}
        return set(required) <= columns[table]

    def count(table, predicate):
        # Both identifiers and predicates come only from the static inventory.
        return int(conn.execute(text(f'SELECT count(*) FROM "{table}" WHERE {predicate}'), params).scalar_one())

    if available("bot_profiles", ("slug",)):
        profile = conn.execute(text("SELECT * FROM bot_profiles WHERE slug=:slug"), params).mappings().first()
        profile_version = ({"state": "present", "fingerprint": fingerprint(dict(profile))}
                           if profile is not None else {"state": "absent", "fingerprint": None})
    else:
        profile_version = {"state": "unknown", "fingerprint": None}

    owners = set()
    missing_owner_sources = []
    for table, required, column, predicate in OWNER_SOURCES:
        if not available(table, required):
            missing_owner_sources.append(table)
            continue
        for raw in conn.execute(text(f'SELECT DISTINCT "{column}" FROM "{table}" WHERE {predicate}')).scalars():
            if isinstance(raw, str):
                identity = raw.strip().lower()
                if identity and sanitize_table_name(identity) == storage:
                    owners.add(identity)
    aliases = owners - {slug}
    ownership = {
        "status": ("ambiguous" if aliases else "incomplete" if missing_owner_sources
                   else "unproven_profileless" if profile_version["state"] != "present" else "observed_unique"),
        "candidate_count": len(owners),
        "candidate_fingerprint": fingerprint(sorted(owners)),
        "missing_sources": sorted(missing_owner_sources),
        "reason": "Raw-owner evidence is not an exhaustive historical ownership registry; partition rows store only the sanitized ID.",
    }
    entries = []
    for spec in (*DELETED, *RETAINED):
        requirements = ((spec.table, spec.columns), *spec.dependencies)
        missing = sorted(t for t, cols in requirements if not available(t, cols))
        entry = {"key": spec.key, "store": "local_postgresql", "table": spec.table,
                 "disposition": spec.disposition, "count": None, "coverage": "observed",
                 "reason": spec.reason}
        if missing:
            entry.update(disposition="unknown", coverage="missing_table_or_columns",
                         reason=f"Cannot count current schema requirements: {', '.join(missing)}. Absence is not a measured zero.")
        else:
            n = count(spec.table, spec.predicate)
            if spec.key in {"messages", "memories", "forgotten_messages"} and ownership["status"] != "observed_unique":
                entry.update(disposition="unknown", coverage="ambiguous_ownership",
                             observed_storage_count=n,
                             reason="Storage-key rows cannot be attributed exclusively to this raw slug; no target-owned deletion count claimed.")
            else:
                entry["count"] = n
        entries.append(entry)
    for key, reason in UNKNOWN.items():
        entries.append({"key": key, "store": "outside_counted_scope", "table": None,
                        "disposition": "unknown", "count": None, "coverage": "not_inspected", "reason": reason})

    payload = {
        "schema_version": 1, "scope_version": SCOPE_VERSION,
        "scope": "current_standalone_purge_data", "execution_enabled": False,
        "scope_fingerprint": fingerprint([asdict(s) for s in (*DELETED, *RETAINED)]),
        "slug": slug, "storage_id": storage, "profile_version": profile_version,
        "storage_ownership": ownership,
        "snapshot": "local_database_observation_only",
        "limitations": [
            "Deleted means conditional selection by the existing executor, not completed deletion or approval to execute.",
            "Retained counts can overlap; do not sum entries. Operational retained rows are not necessarily historical or inactive.",
            "Digest covers observed scope/counts/profile, not every row identity/value; equal-count changes can preserve it.",
            "No future freshness, alias-reuse safety, active/protected-bot clearance, FK success, writer fence or cross-store atomicity is established.",
            "Profile-only delete and legacy destructive endpoints are unchanged; this preview is not accepted by an execution endpoint.",
        ],
        "inventory": entries,
    }
    payload["manifest_digest"] = fingerprint(payload)
    return payload


def preview_bot_purge(engine, slug: str) -> dict:
    """Enforce a coherent, read-only local transaction; never dispose shared pool."""
    slug = normalize_slug(slug)
    if engine is None or engine.dialect.name != "postgresql":
        raise PreviewUnavailable("PostgreSQL preview unavailable")
    try:
        with engine.connect() as conn:
            conn = conn.execution_options(isolation_level="REPEATABLE READ", postgresql_readonly=True)
            with conn.begin():
                return collect_preview(conn, slug)
    except SQLAlchemyError as exc:
        # Driver errors/SQL parameters may contain private data; API is generic.
        raise PreviewUnavailable("PostgreSQL preview unavailable") from exc
