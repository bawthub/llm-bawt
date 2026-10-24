"""PostgreSQL memory backend for llm-bawt with pgvector support.

This backend uses PostgreSQL with the pgvector extension for semantic similarity search.

Storage layout (TASK-571): three LIST-partitioned parent tables, one
partition per bot:

  - ``messages``            PARTITION BY LIST (bot_id) — permanent message storage
  - ``memories``            PARTITION BY LIST (bot_id) — distilled, importance-weighted
  - ``forgotten_messages``  PARTITION BY LIST (bot_id) — soft-deleted messages

Each bot's partition is named ``<parent>_p_<sanitized_bot_id>`` (e.g.
``messages_p_snark``) and the backend addresses its own bot's partition
DIRECTLY — a query against ``messages_p_snark`` physically cannot return
another bot's rows, preserving the hard per-bot isolation the legacy
``<bot>_messages`` shard tables provided. Cross-bot features (Spotlight
search, source listing) query the parent with ``bot_id`` in the select
list instead of UNION-ALL-ing shards.

Each partition carries ``ALTER COLUMN bot_id SET DEFAULT '<bot>'`` so the
existing INSERT statements (which don't mention bot_id) keep working
unchanged; the partition bound guarantees correctness.

The separation allows:
  - Messages: Complete conversation history, never deleted
  - Memories: Curated facts extracted from conversations with importance scores
"""

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Column, DateTime, Float, Integer, JSON, MetaData, String, Table, Text, text

from .base import MemoryBackend
from .postgresql_memories import PostgreSQLSemanticMemoryMixin
from .postgresql_messages import PostgreSQLMessageMixin
from .session_unique_index import ensure_fresh_sessions_index
from .message_store import (
    FORGOTTEN_PARENT,
    MESSAGES_PARENT,
    MessageRowStore,
    ensure_message_parent_tables,
    get_forgotten_message_table,
    get_message_table,
    message_parent_indexes,
)
from ..utils.schema import SchemaBootstrapGuard
# Compatibility export; policy and storage must use exactly the same mapping.
from ..utils.bot_partition import sanitize_table_name as _sanitize_table_name

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Partitioned parent tables (TASK-571)
# ---------------------------------------------------------------------------

# Parent table names. Message parents are owned by message_store and re-exported
# here for compatibility with existing migrations/routes.
MEMORIES_PARENT = "memories"
PARENT_TABLES = (MESSAGES_PARENT, MEMORIES_PARENT, FORGOTTEN_PARENT)


def partition_name(base: str, bot_id: str) -> str:
    """Single authority for a bot's partition name: ``<base>_p_<sanitized>``.

    ``base`` is one of the parent table names (``messages`` / ``memories`` /
    ``forgotten_messages``); ``bot_id`` may be raw — it is sanitized here.
    """
    return f"{base}_p_{_sanitize_table_name(bot_id)}"


# Set to True when hnsw-on-partitioned-parent turned out to be unsupported by
# the installed pgvector build; ensure_bot_partitions then creates the vector
# indexes per-partition instead (functionally identical — a partitioned index
# is just a template that materializes per-partition anyway).
_hnsw_parent_unsupported = False


def ensure_parent_tables(conn, embedding_dim: int = 384) -> None:
    """Create the three LIST-partitioned parent tables + parent indexes.

    Idempotent (CREATE ... IF NOT EXISTS throughout). Runs against a live
    DB safely: new names, no collision with any legacy ``<bot>_*`` shard
    tables. Indexes created on the parent are templates — every partition
    (existing and future) gets its own physical index automatically.

    The composite PRIMARY KEY (bot_id, id) is required on partitioned
    tables (the partition key must be part of the PK). Within a partition
    ``id`` remains effectively unique (UUIDs), so partition-direct
    ``WHERE id = :id`` queries are unaffected.
    """
    global _hnsw_parent_unsupported

    ensure_message_parent_tables(conn)

    memories_sql = text(f"""
        CREATE TABLE IF NOT EXISTS {MEMORIES_PARENT} (
            bot_id VARCHAR(64) NOT NULL,
            id VARCHAR(36) NOT NULL,
            content TEXT NOT NULL,
            tags JSONB NOT NULL DEFAULT '["misc"]'::jsonb,
            importance REAL NOT NULL DEFAULT 0.5,
            source_message_ids JSONB,
            access_count INTEGER DEFAULT 0,
            last_accessed TIMESTAMP,
            intent TEXT,
            stakes TEXT,
            emotional_charge REAL,
            recurrence_keywords JSONB,
            meaning_updated_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            embedding vector({embedding_dim}),
            meaning_embedding vector({embedding_dim}),
            PRIMARY KEY (bot_id, id)
        ) PARTITION BY LIST (bot_id)
    """)

    conn.execute(memories_sql)

    # Parent-level btree/gin indexes — template to every partition.
    parent_indexes = [
        *message_parent_indexes(),
        f"CREATE INDEX IF NOT EXISTS idx_{MEMORIES_PARENT}_importance ON {MEMORIES_PARENT}(importance)",
        f"CREATE INDEX IF NOT EXISTS idx_{MEMORIES_PARENT}_accessed ON {MEMORIES_PARENT}(last_accessed)",
        f"CREATE INDEX IF NOT EXISTS idx_{MEMORIES_PARENT}_tags_gin ON {MEMORIES_PARENT} USING gin (tags)",
    ]
    # Each index runs under a SAVEPOINT (begin_nested) so a single failure
    # doesn't abort the enclosing transaction — load-bearing for the hnsw
    # fallback below, and for the migration script's rolled-back dry-run.
    for idx_sql in parent_indexes:
        try:
            with conn.begin_nested():
                conn.execute(text(idx_sql))
        except Exception as e:
            logger.debug(f"Parent index creation (may already exist): {e}")

    # HNSW vector indexes on the parent. If the installed pgvector build
    # rejects hnsw on a partitioned table, fall back to per-partition
    # creation inside ensure_bot_partitions (same physical result).
    for col in ("embedding", "meaning_embedding"):
        hnsw_sql = f"""
            CREATE INDEX IF NOT EXISTS idx_{MEMORIES_PARENT}_{col}
            ON {MEMORIES_PARENT}
            USING hnsw ({col} vector_cosine_ops)
        """
        try:
            with conn.begin_nested():
                conn.execute(text(hnsw_sql))
        except Exception as e:
            _hnsw_parent_unsupported = True
            logger.warning(
                f"hnsw index on partitioned parent unsupported ({e}); "
                "falling back to per-partition vector indexes"
            )


def ensure_bot_partitions(conn, bot_id: str) -> None:
    """Create (idempotently) one partition per parent table for ``bot_id``.

    The partition NAME uses the sanitized identifier; the partition VALUE is
    the sanitized bot id too — deliberately: the sanitized form IS the bot's
    storage identity today (it's what the legacy shard-table prefix was), so
    every consumer (scheduler aggregates, global-search exclusion sets,
    Spotlight ``source`` attribution) keeps exactly the identity it already
    used. A per-partition column DEFAULT makes bot_id transparent to the
    existing INSERTs, which never mention it.

    No DEFAULT (catch-all) partition on purpose: an unprovisioned bot_id
    write fails loudly instead of silently pooling — and this function makes
    that unreachable in practice (it runs at every backend init).

    Concurrent init of the same new bot races CREATE TABLE IF NOT EXISTS —
    PG serializes on the parent lock; duplicate errors are swallowed.
    """
    sanitized = _sanitize_table_name(bot_id)
    for base in PARENT_TABLES:
        part = f"{base}_p_{sanitized}"
        # This catalog probe is intentionally ahead of every DDL statement.
        # ``IF NOT EXISTS`` still takes heavyweight relation locks, so issuing
        # CREATE/ALTER on every cold process start made a read path capable of
        # blocking behind its own open SELECT transaction (TASK-739).
        exists = conn.execute(
            text("SELECT to_regclass(:partition_name)"),
            {"partition_name": part},
        ).scalar()
        if exists:
            continue
        try:
            # SAVEPOINT so a create race doesn't abort the enclosing txn
            # before the existence re-check below.
            with conn.begin_nested():
                conn.execute(text(
                    f"CREATE TABLE {part} "
                    f"PARTITION OF {base} FOR VALUES IN ('{sanitized}')"
                ))
                conn.execute(text(
                    f"ALTER TABLE {part} ALTER COLUMN bot_id SET DEFAULT '{sanitized}'"
                ))
        except Exception as e:
            # Duplicate-table race from a concurrent backend init, or the
            # partition already exists with the same bound. Idempotent-ish:
            # verify existence before treating as fatal.
            exists = conn.execute(
                text("SELECT to_regclass(:partition_name)"),
                {"partition_name": part},
            ).scalar()
            if not exists:
                raise
            logger.debug(f"Partition {part} creation race (already exists): {e}")

    # Per-partition vector indexes when the parent-level hnsw template was
    # rejected by the installed pgvector build.
    if _hnsw_parent_unsupported:
        mem_part = f"{MEMORIES_PARENT}_p_{sanitized}"
        for col in ("embedding", "meaning_embedding"):
            try:
                with conn.begin_nested():
                    conn.execute(text(f"""
                        CREATE INDEX IF NOT EXISTS idx_{mem_part}_{col}
                        ON {mem_part}
                        USING hnsw ({col} vector_cosine_ops)
                    """))
            except Exception as e:
                logger.debug(f"Per-partition hnsw creation: {e}")


# Memory backends share the process-wide engine (TASK-202): see
# llm_bawt.utils.db.get_shared_engine. All bots talk to the same Postgres
# database — only table names differ — so they all share one pool.


# Shared metadata for table definitions
metadata = MetaData()

# Cache for non-message table objects. Message table caches live in
# message_store alongside their definitions.
_memory_table_cache: dict[str, Table] = {}

# Shared sessions table (not per-bot — sessions are first-class entities,
# scoped by bot_id column rather than a separate table per bot).
# Schema:
#   id          UUID primary key (matches messages.session_id)
#   bot_id      Which bot the session belongs to
#   started_at  When the session opened
#   ended_at    When the session was closed (NULL while active)
#   status      'active' | 'archived' | 'deleted' (TASK-250; legacy rows may
#               still read 'completed' — normalized to 'archived' on read)
#   archived_at When the session left the active state (NULL while active)
#   metadata    JSONB grab-bag for future extensibility
sessions_table = Table(
    "sessions",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("bot_id", String(64), nullable=False),
    # TASK-284: raw history is namespaced (bot_id, user_id), so the session —
    # the thread boundary — must carry the user dimension too. Nullable for
    # back-compat with legacy rows (migrated to a per-(bot,user) legacy session).
    Column("user_id", String(64), nullable=True),
    Column("started_at", DateTime, nullable=False, default=_utcnow),
    Column("ended_at", DateTime, nullable=True),
    Column("status", String(16), nullable=False, default="active"),
    # TASK-250: when the session left the active state (archive timestamp).
    Column("archived_at", DateTime, nullable=True),
    Column("session_metadata", JSON, nullable=True),
    extend_existing=True,
)


def get_message_table_pg(bot_id: str) -> Table:
    """Compatibility wrapper for one bot's canonical message partition."""
    return get_message_table(
        metadata,
        partition_name(MESSAGES_PARENT, bot_id),
        _utcnow,
    )


def get_forgotten_table_pg(bot_id: str) -> Table:
    """Compatibility wrapper for one bot's forgotten message partition."""
    return get_forgotten_message_table(
        metadata,
        partition_name(FORGOTTEN_PARENT, bot_id),
        _utcnow,
    )


def get_memory_table_pg(bot_id: str) -> Table:
    """Get or create a memory Table for a specific bot (PostgreSQL version).

    Points at the bot's PARTITION of the ``memories`` parent (TASK-571).
    """
    table_name = partition_name(MEMORIES_PARENT, bot_id)
    
    if table_name in _memory_table_cache:
        return _memory_table_cache[table_name]
    
    table = Table(
        table_name,
        metadata,
        Column("id", String(36), primary_key=True),  # UUID
        Column("content", Text, nullable=False),
        Column("tags", JSON, nullable=False, default=["misc"]),
        Column("importance", Float, nullable=False, default=0.5),
        Column("source_message_ids", JSON, nullable=True),  # Array of message UUIDs
        Column("access_count", Integer, default=0),  # For reinforcement
        Column("last_accessed", DateTime, nullable=True),
        Column("intent", Text, nullable=True),
            Column("stakes", Text, nullable=True),
            Column("emotional_charge", Float, nullable=True),
            Column("recurrence_keywords", JSON, nullable=True),
            Column("meaning_updated_at", DateTime, nullable=True),
        Column("created_at", DateTime, default=_utcnow),
        Column("updated_at", DateTime, default=_utcnow, onupdate=_utcnow),
        # embedding column added via raw SQL for pgvector
        extend_existing=True
    )
    
    _memory_table_cache[table_name] = table
    return table


class PostgreSQLMemoryBackend(
    PostgreSQLMessageMixin, PostgreSQLSemanticMemoryMixin, MemoryBackend
):
    """PostgreSQL-based memory backend with pgvector for semantic search.
    
    This is the new memory backend designed for:
    - Permanent message storage (all messages preserved)
    - Distilled memories with importance weighting
    - Semantic similarity search via pgvector
    - Source linking back to original messages
    
    Configuration (via environment variables or .env):
        LLM_BAWT_POSTGRES_HOST: Database host (default: localhost)
        LLM_BAWT_POSTGRES_PORT: Database port (default: 5432)
        LLM_BAWT_POSTGRES_USER: Database user
        LLM_BAWT_POSTGRES_PASSWORD: Database password
        LLM_BAWT_POSTGRES_DATABASE: Database name (default: llm_bawt)
    """
    
    # Embedding dimension (matches sentence-transformers all-MiniLM-L6-v2)
    EMBEDDING_DIM = 384
    _schema_guard = SchemaBootstrapGuard()
    
    def __init__(
        self,
        config: Any,
        bot_id: str = "nova",
        embedding_dim: int | None = None,
        *,
        provision_schema: bool = True,
    ):
        super().__init__(config, bot_id=bot_id)
        
        # Get embedding settings from config
        self.embedding_model = getattr(config, 'MEMORY_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        if embedding_dim is None:
            embedding_dim = getattr(config, 'MEMORY_EMBEDDING_DIM', 384)
        
        # Get PostgreSQL connection settings from config
        host = getattr(config, 'POSTGRES_HOST', 'localhost')
        port = int(getattr(config, 'POSTGRES_PORT', 5432))
        user = getattr(config, 'POSTGRES_USER', 'llm_bawt')
        password = getattr(config, 'POSTGRES_PASSWORD', '')
        database = getattr(config, 'POSTGRES_DATABASE', 'llm_bawt')
        
        self.database = database
        self.bot_id_sanitized = _sanitize_table_name(bot_id)
        # Partition-direct access (TASK-571): all bot-scoped SQL targets the
        # bot's partition, so every interpolated query below (and in
        # maintenance/consolidation/summarization, which ride these attrs)
        # keeps its exact shape — only the name changed.
        self._messages_table_name = partition_name(MESSAGES_PARENT, bot_id)
        self._memories_table_name = partition_name(MEMORIES_PARENT, bot_id)
        self._forgotten_table_name = partition_name(FORGOTTEN_PARENT, bot_id)
        self.embedding_dim = embedding_dim
        
        # Get table definitions
        self.messages_table = get_message_table_pg(bot_id)
        self.memories_table = get_memory_table_pg(bot_id)
        self.forgotten_table = get_forgotten_table_pg(bot_id)
        
        # Use the process-wide shared engine (TASK-202). Every Store +
        # every bot's memory backend share one connection pool.
        from ..utils.db import get_shared_engine
        self.engine = get_shared_engine(config)
        if self.engine is None:
            raise RuntimeError(
                "PostgreSQLMemoryBackend requires Postgres credentials"
            )
        self._message_rows = MessageRowStore(
            self.engine,
            self.messages_table,
            self._messages_table_name,
        )

        self._schema_ready = False
        if provision_schema:
            self.ensure_schema()
        logger.debug(f"Connected to PostgreSQL at {host}:{port}/{database} (bot: {bot_id})")

    def ensure_schema(self) -> None:
        """Provision shared schema and this bot's partitions before a write.

        Read-only callers construct the backend with ``provision_schema=False``
        so an HTTP GET can never acquire DDL locks.  A cached read backend can
        later be promoted safely when a mutation arrives.
        """
        if self._schema_ready:
            return
        self._ensure_tables_exist()
        self._schema_ready = True
    
    def _ensure_tables_exist(self) -> None:
        """Ensure the partitioned parents + this bot's partitions exist.

        Shared parents/extensions/sessions are guarded per engine and embedding
        dimension. Bot partitions are guarded separately by sanitized storage
        identity so distinct bots still provision independently.

        The parents are created with the full current schema; runtime
        bootstrap does not migrate legacy shard tables.
        """
        def bootstrap_shared(conn) -> None:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            messages_was_new = (
                conn.dialect.name == "postgresql" and
                conn.execute(text("SELECT to_regclass('public.messages') IS NULL")).scalar_one()
            )
            ensure_parent_tables(conn, self.embedding_dim)
            sessions_was_new = (
                conn.dialect.name == "postgresql" and
                conn.execute(text("SELECT to_regclass('public.sessions') IS NULL")).scalar_one()
            )

            # Shared sessions table (TASK-183). One row per session across
            # all bots; promotes session_id from a bare UUID to a first-class
            # entity with start/end timestamps and a status.
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id VARCHAR(36) PRIMARY KEY,
                    bot_id VARCHAR(64) NOT NULL,
                    user_id VARCHAR(64),
                    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    ended_at TIMESTAMP,
                    archived_at TIMESTAMP,
                    status VARCHAR(16) NOT NULL DEFAULT 'active',
                    session_metadata JSONB
                )
            """))
            # TASK-284: add the user dimension to pre-existing sessions tables.
            conn.execute(text("""
                ALTER TABLE sessions
                ADD COLUMN IF NOT EXISTS user_id VARCHAR(64)
            """))
            # TASK-250 gap (caught in review): pre-existing tables bootstrapped
            # before the status-lifecycle migration lack archived_at.
            conn.execute(text("""
                ALTER TABLE sessions
                ADD COLUMN IF NOT EXISTS archived_at TIMESTAMP
            """))
            # Idempotent legacy-data migration: retire the pre-TASK-250
            # 'completed' status and stamp archived_at on any archived row
            # still missing it (backfilled from ended_at). Zero rows touched
            # on an already-migrated deployment.
            conn.execute(text("""
                UPDATE sessions
                SET status = 'archived',
                    archived_at = COALESCE(archived_at, ended_at)
                WHERE status = 'completed'
            """))
            conn.execute(text("""
                UPDATE sessions
                SET archived_at = ended_at
                WHERE status = 'archived'
                  AND archived_at IS NULL
                  AND ended_at IS NOT NULL
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_sessions_bot_started
                ON sessions(bot_id, started_at)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_sessions_status
                ON sessions(status)
            """))
            # TASK-284: the active-session lookup is keyed (bot_id, user_id) and
            # filtered to the live thread — index that access path.
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_sessions_bot_user_active
                ON sessions(bot_id, user_id, status, started_at)
            """))
            # Enforce the cold-start race invariant on NEW databases. Existing
            # tables without it require the explicit concurrent upgrade path;
            # never build a potentially blocking index during app startup.
            if conn.dialect.name == "postgresql":
                ensure_fresh_sessions_index(conn, table_was_new=sessions_was_new)
                # The parent exists before sessions, so its outgoing FK can
                # only be installed after sessions. Never retrofit an existing
                # messages parent at startup; that belongs to an explicit plan.
                if messages_was_new:
                    conn.execute(text("""
                        ALTER TABLE public.messages ADD CONSTRAINT fk_messages_session
                        FOREIGN KEY (session_id) REFERENCES public.sessions(id)
                        ON DELETE NO ACTION NOT DEFERRABLE
                    """))

            from ..bot_session_state import BotSessionState
            BotSessionState.ensure_schema(conn)

        self._schema_guard.run(
            self.engine,
            ("memory-shared", self.embedding_dim),
            bootstrap_shared,
        )

        def bootstrap_partitions(conn) -> None:
            ensure_bot_partitions(conn, self.bot_id_sanitized)

        self._schema_guard.run(
            self.engine,
            ("memory-partitions", self.bot_id_sanitized),
            bootstrap_partitions,
        )

    @classmethod
    def get_short_term_manager(cls, config: Any, bot_id: str = "nova") -> "PostgreSQLShortTermManager":
        """Factory method to get a short-term memory manager."""
        return PostgreSQLShortTermManager(config, bot_id)


# Compatibility export: existing callers import this class from postgresql.py.
from .postgresql_short_term import PostgreSQLShortTermManager
