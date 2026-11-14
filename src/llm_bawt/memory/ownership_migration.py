"""Explicit, one-bot memory ownership activation for a maintenance window.

Schema expansion never infers an owner or activates existing bots. Activation
uses writer-excluding table locks: the memory dedup advisory lock alone cannot
exclude a concurrent session for a second user. Callers own the transaction.
"""
from __future__ import annotations

import json
from uuid import uuid4

from sqlalchemy import text

from llm_bawt.memory.message_store import FORGOTTEN_PARENT
from llm_bawt.utils.bot_partition import sanitize_table_name as _sanitize_table_name

ACTIVATIONS = "memory_ownership_activations"
OWNER_INDEX = "idx_memories_owner_user"


def ensure_schema(conn) -> None:
    """Nullable metadata-only expansion; no data scan, index build or backfill."""
    present = conn.execute(text("""
        SELECT 1 FROM pg_attribute
        WHERE attrelid = 'memories'::regclass AND attname = 'owner_user_id'
          AND NOT attisdropped
    """)).first()
    if not present:
        conn.execute(text("ALTER TABLE memories ADD COLUMN owner_user_id TEXT"))
    ensure_activation_schema(conn)


def ensure_activation_schema(conn) -> None:
    """Profile creation can precede provisioning the memory parent."""
    conn.execute(text(f"""
        CREATE TABLE IF NOT EXISTS {ACTIVATIONS} (
            bot_id TEXT PRIMARY KEY,
            mode TEXT NOT NULL CHECK (mode IN ('new', 'inferred', 'attested', 'mark_only')),
            owner_user_id TEXT,
            actor_id TEXT NOT NULL,
            activated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            counts JSONB NOT NULL DEFAULT '{{}}'::jsonb
        )
    """))


def mark_new_bot(conn, bot_id: str) -> None:
    """Mark new profiles, never rediscovered/recreated historical identities.

    Seed/import may recreate a missing profile without the HTTP create checks.
    Keep those old rows unmarked; only explicit migration may interpret them.
    """
    from llm_bawt.bot_slug_aliases import storage_id_reserved
    from llm_bawt.utils.bot_partition import lock_bot_storage

    partition = _sanitize_table_name(bot_id)
    lock_bot_storage(conn, partition)
    if storage_id_reserved(conn, partition):
        raise ValueError("bot storage identity is reserved by a retired alias")
    for table, column in (("bot_profiles", "slug"), ("sessions", "bot_id"),
                          ("bot_profile_changes", "bot_slug")):
        if conn.execute(text("SELECT to_regclass(:table)"), {"table": table}).scalar():
            collision = conn.execute(text(f"""
                SELECT 1 FROM {table} WHERE {column}<>:bot AND
                    COALESCE(NULLIF(regexp_replace(lower({column}), '[^a-z0-9_]', '', 'g'), ''), 'default')=:partition
                LIMIT 1
            """), {"bot": bot_id, "partition": partition}).first()
            if collision:
                return  # A new alias/profile must not activate an old partition.
    for table, column, identity in (
        ("memories", "bot_id", partition),
        ("messages", "bot_id", partition),
        (FORGOTTEN_PARENT, "bot_id", partition),
        ("sessions", "bot_id", bot_id),
        ("bot_profile_changes", "bot_slug", bot_id),
    ):
        if conn.execute(text("SELECT to_regclass(:table)"), {"table": table}).scalar():
            if conn.execute(text(f"SELECT 1 FROM {table} WHERE {column}=:identity LIMIT 1"),
                            {"identity": identity}).first():
                return
    conn.execute(text(f"""
        INSERT INTO {ACTIVATIONS} (bot_id, mode, actor_id)
        VALUES (:bot, 'new', 'profile-create') ON CONFLICT (bot_id) DO NOTHING
    """), {"bot": partition})


def _identity(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip() or "*" in value:
        raise ValueError(f"{label} must be one explicit nonblank identity")
    return value


def _restricted_counts(conn, params: dict) -> dict:
    """Disjoint reasons NULL rows cannot prove one human owner; no fact content.

    Priority is unsourced, legacy, service/unknown (including malformed or missing
    sources), then mixed human owners. Counts describe rows before any stamping.
    """
    return dict(conn.execute(text(f"""
        WITH all_messages AS (
            SELECT id, session_id FROM messages WHERE bot_id=:partition
            UNION ALL
            SELECT id, session_id FROM {FORGOTTEN_PARENT} WHERE bot_id=:partition
        ), reasons AS (
            SELECT CASE
                WHEN mem.source_message_ids IS NULL OR mem.source_message_ids='[]'::jsonb
                    THEN 'unsourced'
                WHEN proof.legacy THEN 'legacy'
                WHEN NOT COALESCE(proof.valid, false) THEN 'service_or_unknown'
                WHEN proof.owners<>1 THEN 'mixed_human'
                ELSE NULL END AS reason
            FROM memories mem
            LEFT JOIN LATERAL (
                SELECT bool_or(s.session_metadata->>'legacy_backfill'='true') AS legacy,
                    bool_and(COALESCE(jsonb_typeof(src.value)='string'
                        AND s.bot_id=:bot AND s.status IN ('active', 'archived')
                        AND EXISTS (SELECT 1 FROM entity_profiles p
                            WHERE lower(p.entity_type::text)='user' AND p.entity_id=s.user_id
                              AND NOT p.is_service)
                        AND NOT EXISTS (SELECT 1 FROM entity_profiles p
                            WHERE lower(p.entity_type::text)='user' AND p.entity_id=s.user_id
                              AND p.is_service), false)) AS valid,
                    count(DISTINCT s.user_id) AS owners
                FROM jsonb_array_elements(CASE WHEN jsonb_typeof(mem.source_message_ids)='array'
                    THEN mem.source_message_ids ELSE '[]'::jsonb END) src(value)
                LEFT JOIN all_messages m ON m.id=(src.value #>> '{{}}')
                LEFT JOIN sessions s ON s.id=m.session_id
            ) proof ON true
            WHERE mem.bot_id=:partition AND mem.owner_user_id IS NULL
        )
        SELECT count(*) FILTER (WHERE reason IS NOT NULL) AS restricted_memories,
               count(*) FILTER (WHERE reason='unsourced') AS restricted_unsourced,
               count(*) FILTER (WHERE reason='legacy') AS restricted_legacy,
               count(*) FILTER (WHERE reason='service_or_unknown') AS restricted_service_or_unknown,
               count(*) FILTER (WHERE reason='mixed_human') AS restricted_mixed_human
        FROM reasons
    """), params).mappings().one())


def activate(conn, bot_id: str, *, actor_id: str, attest_user: str | None = None,
             mark_only: bool = False, dry_run: bool = True) -> dict:
    """Inspect or activate one bot. Results contain counts/reasons, never facts.

    Inference marks an evidenced multi-human bot without stamping ambiguous
    rows. Legacy or uncertain evidence refuses inference. Explicit mark-only
    mode preserves every owner/NULL row even with uncertain evidence; it never
    silently replaces failed inference. Attestation cannot override another
    proven human owner and is mutually exclusive with mark-only.
    Refusal does not mutate any row. Success is idempotent, including after a
    second user joins: an existing marker is never interpreted again.
    """
    bot_id = _identity(bot_id, "bot_id")
    actor_id = _identity(actor_id, "actor_id")
    if mark_only and attest_user is not None:
        raise ValueError("mark-only and attestation are mutually exclusive")
    if attest_user is not None:
        attest_user = _identity(attest_user, "attest_user")
    partition = _sanitize_table_name(bot_id)
    params = {"bot": bot_id, "partition": partition}
    if conn.execute(text("SHOW transaction_isolation")).scalar_one() != "read committed":
        raise ValueError("activation requires READ COMMITTED for a fresh post-lock evidence snapshot")
    # LOCK without ONLY covers children, including forgotten messages. Holding
    # profiles also prevents an is_service reclassification during inference.
    conn.execute(text(f"""
        LOCK TABLE bot_profiles, entity_profiles, sessions, messages,
                   {FORGOTTEN_PARENT}, memories, {ACTIVATIONS}
        IN SHARE ROW EXCLUSIVE MODE
    """))
    if not conn.execute(text("SELECT 1 FROM bot_profiles WHERE slug=:bot"), params).first():
        return {"status": "refused", "reason": "unknown_bot", "dry_run": dry_run}
    # A partition can outlive a deleted profile; check session identities as
    # well as current profiles before assigning every row in that partition.
    aliases = conn.execute(text("""
        SELECT count(*) FROM (
            SELECT slug AS bot FROM bot_profiles UNION SELECT bot_id AS bot FROM sessions
        ) identities WHERE bot<>:bot AND
          COALESCE(NULLIF(regexp_replace(lower(bot), '[^a-z0-9_]', '', 'g'), ''), 'default')=:partition
    """), params).scalar_one()
    if aliases:
        return {"status": "refused", "reason": "storage_identity_collision", "dry_run": dry_run}
    previous = conn.execute(text(f"""
        SELECT mode, counts FROM {ACTIVATIONS} WHERE bot_id=:partition
    """), params).mappings().first()
    if previous:
        return {"status": "already_active", "mode": previous["mode"],
                "counts": previous["counts"], "dry_run": dry_run}

    sessions = conn.execute(text("""
        SELECT s.id, s.user_id,
               COALESCE(s.session_metadata->>'legacy_backfill', 'false') = 'true' AS legacy,
               EXISTS (SELECT 1 FROM entity_profiles p WHERE lower(p.entity_type::text)='user'
                       AND p.entity_id=s.user_id AND NOT p.is_service) AS human,
               EXISTS (SELECT 1 FROM entity_profiles p WHERE lower(p.entity_type::text)='user'
                       AND p.entity_id=s.user_id AND p.is_service) AS service
        FROM sessions s WHERE s.bot_id=:bot
    """), params).mappings().all()
    humans = {s["user_id"] for s in sessions if s["human"] and not s["service"] and not s["legacy"]}
    evidence = conn.execute(text(f"""
        WITH all_messages AS (
            SELECT session_id, role, author_entity_type, author_entity_id
            FROM messages WHERE bot_id=:partition
            UNION ALL
            SELECT session_id, role, author_entity_type, author_entity_id
            FROM {FORGOTTEN_PARENT} WHERE bot_id=:partition
        )
        SELECT count(*) AS messages,
               count(*) FILTER (WHERE s.session_metadata->>'legacy_backfill'='true') AS legacy_messages,
               count(*) FILTER (WHERE s.id IS NULL OR s.bot_id<>:bot) AS unbound_messages,
               count(*) FILTER (WHERE m.role='user' AND
                   (m.author_entity_type IS DISTINCT FROM 'user' OR
                    m.author_entity_id IS DISTINCT FROM s.user_id)) AS unproven_authors
        FROM all_messages m LEFT JOIN sessions s ON s.id=m.session_id
    """), params).mappings().one()
    counts = {key: int(value) for key, value in evidence.items()}
    counts.update(sessions=len(sessions), human_owners=len(humans),
                  service_sessions=sum(bool(s["service"]) for s in sessions),
                  unknown_sessions=sum(not s["human"] and not s["service"] for s in sessions))
    row_counts = conn.execute(text("""
        SELECT count(*) AS memories,
               count(*) FILTER (WHERE owner_user_id IS NULL) AS unowned_memories
        FROM memories WHERE bot_id=:partition
    """), params).mappings().one()
    counts.update({key: int(value) for key, value in row_counts.items()})
    counts.update(_restricted_counts(conn, params))
    owner = attest_user
    reason = None
    if attest_user is not None:
        human = conn.execute(text("""
            SELECT count(*) = 1 AND bool_and(NOT is_service)
            FROM entity_profiles WHERE lower(entity_type::text)='user' AND entity_id=:user
        """), {"user": attest_user}).scalar()
        if not human:
            reason = "attestation_requires_human_profile"
        elif humans - {attest_user}:
            reason = "other_human_owner"
        elif conn.execute(text("""
            SELECT 1 FROM memories WHERE bot_id=:partition
            AND owner_user_id IS NOT NULL AND owner_user_id<>:user LIMIT 1
        """), {**params, "user": attest_user}).first():
            reason = "other_explicit_owner"
    elif not mark_only:
        if counts["legacy_messages"]:
            reason = "legacy_evidence_requires_attestation"
        elif counts["service_sessions"] or counts["unknown_sessions"] or counts["unbound_messages"] or counts["unproven_authors"]:
            reason = "unproven_evidence_requires_attestation"
        elif not humans:
            reason = "no_human_evidence_requires_attestation"
        elif len(humans) == 1:
            owner = next(iter(humans))
        if reason is None and owner:
            # A single human session is insufficient when a row claims sources
            # that cannot be attributed to that human. Manual NULL/[] sources
            # are the only intentionally unsourced case.
            unproven = conn.execute(text(f"""
                SELECT count(*) FROM memories mem WHERE bot_id=:partition AND (
                    (owner_user_id IS NOT NULL AND owner_user_id<>:owner)
                    OR (source_message_ids IS NOT NULL AND
                        jsonb_typeof(source_message_ids)<>'array')
                    OR EXISTS (
                        SELECT 1 FROM jsonb_array_elements(CASE
                            WHEN jsonb_typeof(mem.source_message_ids)='array'
                            THEN mem.source_message_ids ELSE '[]'::jsonb END) src(value)
                        WHERE jsonb_typeof(src.value)<>'string' OR NOT EXISTS (
                            SELECT 1 FROM (
                                SELECT id, session_id FROM messages WHERE bot_id=:partition
                                UNION ALL
                                SELECT id, session_id FROM {FORGOTTEN_PARENT} WHERE bot_id=:partition
                            ) m JOIN sessions s ON s.id=m.session_id
                            WHERE m.id=src.value #>> '{{}}' AND s.bot_id=:bot
                              AND s.user_id=:owner
                              AND COALESCE(s.session_metadata->>'legacy_backfill', 'false')<>'true'
                        )
                    )
                )
            """), {**params, "owner": owner}).scalar_one()
            counts["unproven_memories"] = unproven
            if unproven:
                reason = "unproven_row_provenance_requires_attestation"
    if reason:
        return {"status": "refused", "reason": reason, "counts": counts, "dry_run": dry_run}
    counts["stamp_memories"] = counts["unowned_memories"] if owner else 0
    mode = "mark_only" if mark_only else ("attested" if attest_user else "inferred")
    if not dry_run:
        if owner:
            conn.execute(text("""
                UPDATE memories SET owner_user_id=:owner
                WHERE bot_id=:partition AND owner_user_id IS NULL
            """), {**params, "owner": owner})
        conn.execute(text(f"""
            INSERT INTO {ACTIVATIONS} (bot_id, mode, owner_user_id, actor_id, counts)
            VALUES (:partition, :mode, :owner, :actor, CAST(:counts AS jsonb))
        """), {**params, "mode": mode, "owner": owner, "actor": actor_id,
               "counts": json.dumps(counts)})
    return {"status": "would_activate" if dry_run else "activated",
            "mode": mode, "counts": counts, "dry_run": dry_run}


def status(conn) -> dict:
    """Report rollout coverage without dumping bot or user names."""
    return dict(conn.execute(text(f"""
        SELECT count(*) AS bots, count(*) FILTER (WHERE a.bot_id IS NULL) AS unmarked_bots
        FROM bot_profiles b LEFT JOIN {ACTIVATIONS} a
          ON a.bot_id=COALESCE(NULLIF(regexp_replace(lower(b.slug), '[^a-z0-9_]', '', 'g'), ''), 'default')
    """)).mappings().one())


def build_owner_index(engine) -> None:
    """Separate online index phase; never scan children under a parent lock.

    Run once outside the activation transaction, with partition creation paused.
    Rerunning resumes valid completed children; an interrupted invalid concurrent
    index is dropped/rebuilt. Future children inherit the attached parent index.
    """
    quote = engine.dialect.identifier_preparer.quote
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS {OWNER_INDEX} ON ONLY memories (owner_user_id)"))
        children = conn.execute(text("""
            SELECT c.relname, c.oid FROM pg_inherits i JOIN pg_class c ON c.oid=i.inhrelid
            WHERE i.inhparent='memories'::regclass ORDER BY c.relname
        """)).all()
        for child, oid in children:
            # Partitions created after the parent became valid already inherit
            # an attached index (with PostgreSQL's name, not our build name).
            if conn.execute(text("""
                SELECT 1 FROM pg_inherits i JOIN pg_index x ON x.indexrelid=i.inhrelid
                WHERE i.inhparent=to_regclass(:parent) AND x.indrelid=:table_oid
            """), {"parent": OWNER_INDEX, "table_oid": oid}).first():
                continue
            index = f"idx_memory_owner_{oid}"
            existing = conn.execute(text("""
                SELECT indisvalid FROM pg_index WHERE indexrelid=to_regclass(:index)
            """), {"index": index}).scalar()
            if existing is False:
                conn.execute(text(f"DROP INDEX CONCURRENTLY {quote(index)}"))
            conn.execute(text(f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {quote(index)} ON {quote(child)} (owner_user_id)"))
            attached = conn.execute(text("""
                SELECT 1 FROM pg_inherits WHERE inhparent=to_regclass(:parent)
                AND inhrelid=to_regclass(:child)
            """), {"parent": OWNER_INDEX, "child": index}).first()
            if not attached:
                conn.execute(text(f"ALTER INDEX {OWNER_INDEX} ATTACH PARTITION {quote(index)}"))


def downgrade(conn) -> str | None:
    """After code rollback, archive ownership/receipts before removing schema.

    The archive stays in PostgreSQL for an operator-led recovery; no ownership
    data is thrown away, and a failure rolls the whole downgrade back.
    """
    marker = conn.execute(text("SELECT to_regclass(:table)"), {"table": ACTIVATIONS}).scalar()
    column = conn.execute(text("""
        SELECT 1 FROM pg_attribute WHERE attrelid='memories'::regclass
        AND attname='owner_user_id' AND NOT attisdropped
    """)).first()
    if not marker and not column:
        return None
    if not marker or not column:
        raise ValueError("incomplete ownership schema; preserve it for manual recovery")
    archive_id = str(uuid4())
    conn.execute(text(f"LOCK TABLE memories, {ACTIVATIONS} IN ACCESS EXCLUSIVE MODE"))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS memory_ownership_rollback (
            archive_id UUID NOT NULL, kind TEXT NOT NULL, data JSONB NOT NULL,
            archived_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """))
    conn.execute(text("""
        INSERT INTO memory_ownership_rollback (archive_id, kind, data)
        SELECT :archive, 'owner', jsonb_build_object('bot_id', bot_id, 'id', id,
                                                    'owner_user_id', owner_user_id)
        FROM memories WHERE owner_user_id IS NOT NULL
    """), {"archive": archive_id})
    conn.execute(text(f"""
        INSERT INTO memory_ownership_rollback (archive_id, kind, data)
        SELECT :archive, 'activation', to_jsonb(a) FROM {ACTIVATIONS} a
    """), {"archive": archive_id})
    conn.execute(text(f"DROP INDEX IF EXISTS {OWNER_INDEX}"))
    conn.execute(text("ALTER TABLE memories DROP COLUMN owner_user_id"))
    conn.execute(text(f"DROP TABLE {ACTIVATIONS}"))
    return archive_id
