"""Runtime settings storage and resolution.

Supports layered runtime tuning with precedence:
request overrides > bot DB settings > global DB settings > Config fallback.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from typing import Any
from urllib.parse import quote_plus

from sqlalchemy import Column, DateTime, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Session, SQLModel, create_engine, select

from .utils.config import Config, has_database_credentials
from .utils.schema import SchemaBootstrapGuard
from .bot_types import normalize_bot_type

if TYPE_CHECKING:
    from .bots import Bot

logger = logging.getLogger(__name__)


class RuntimeSetting(SQLModel, table=True):
    """DB-backed runtime setting."""

    __tablename__ = "runtime_settings"

    id: int | None = Field(default=None, primary_key=True)
    scope_type: str = Field(
        sa_column=Column(String(16), nullable=False, index=True),
        description="global or bot",
    )
    scope_id: str = Field(
        sa_column=Column(String(128), nullable=False, index=True),
        description="* for global, bot slug for bot scope",
    )
    key: str = Field(sa_column=Column(String(128), nullable=False, index=True))
    value_json: str = Field(sa_column=Column(Text, nullable=False))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class BotProfile(SQLModel, table=True):
    """DB-backed bot personality profile."""

    __tablename__ = "bot_profiles"

    id: int | None = Field(default=None, primary_key=True)
    slug: str = Field(sa_column=Column(String(128), nullable=False, index=True, unique=True))
    name: str = Field(sa_column=Column(String(255), nullable=False))
    description: str = Field(sa_column=Column(Text, nullable=False, default=""))
    system_prompt: str = Field(sa_column=Column(Text, nullable=False))
    # Active persona override. NULL => use ``system_prompt`` (the default persona).
    # When set, points at a ``prompt_templates`` row (category='persona',
    # scope_type='global') whose body REPLACES ``system_prompt`` at prompt-build
    # time. The base ``system_prompt`` is never mutated by switching, so clearing
    # this back to NULL restores the original prompt. See TASK-477.
    prompt_override_id: int | None = Field(
        default=None, sa_column=Column(Integer, nullable=True, index=True)
    )
    requires_memory: bool = Field(default=True)
    voice_optimized: bool = Field(default=False)
    tts_mode: bool = Field(default=False)
    include_summaries: bool = Field(default=True)
    include_in_global_search: bool = Field(default=True)
    uses_tools: bool = Field(default=False)
    uses_search: bool = Field(default=False)
    uses_home_assistant: bool = Field(default=False)
    default_model: str | None = Field(default=None, sa_column=Column(String(255), nullable=True))
    harness: str | None = Field(default=None, sa_column=Column(String(32), nullable=True))
    endpoint_id: int | None = Field(default=None, sa_column=Column(Integer, nullable=True, index=True))
    color: str | None = Field(default=None, sa_column=Column(String(64), nullable=True))
    avatar: str | None = Field(default=None, sa_column=Column(String(512), nullable=True))
    # Self-hosted, ready-to-embed render of ``avatar`` as a ``data:`` URL
    # (Twemoji SVG for emoji, normalized small WebP for images). Resolved
    # ONCE at write time so the chat UI never fetches an avatar from an
    # external CDN at runtime. NULL => frontend renders the native emoji
    # glyph (still zero external hosts).
    avatar_render: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    default_voice: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))
    nextcloud_config: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB, nullable=True))
    bot_type: str | None = Field(default=None, sa_column=Column(String(32), nullable=True))
    agent_backend: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))
    agent_backend_config: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB, nullable=True))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class BotProfileStore:
    """DB access for bot personality profiles."""

    _schema_guard = SchemaBootstrapGuard()

    def __init__(self, config: Config):
        self.config = config
        self.engine = None
        if not has_database_credentials(config):
            return

        try:
            from .utils.db import get_shared_engine
            self.engine = get_shared_engine(config)  # TASK-202: shared pool
            if self.engine is None:
                return
            self._ensure_tables_exist()
        except Exception as e:
            self.engine = None
            logger.warning("Bot profiles DB unavailable: %s", e)

    def _ensure_tables_exist(self) -> None:
        if self.engine is None:
            return
        def bootstrap(conn) -> None:
            SQLModel.metadata.create_all(bind=conn, tables=[BotProfile.__table__])
            self._migrate_add_columns(conn)

        self._schema_guard.run(self.engine, "bot-profile-store", bootstrap)

    def _migrate_add_columns(self, conn) -> None:
        """Add columns introduced after initial schema creation."""
        migrations = [
            "ALTER TABLE bot_profiles ADD COLUMN IF NOT EXISTS color VARCHAR(64)",
            "ALTER TABLE bot_profiles ADD COLUMN IF NOT EXISTS bot_type VARCHAR(32)",
            "ALTER TABLE bot_profiles ADD COLUMN IF NOT EXISTS agent_backend VARCHAR(128)",
            "ALTER TABLE bot_profiles ADD COLUMN IF NOT EXISTS agent_backend_config JSONB",
            "ALTER TABLE bot_profiles ADD COLUMN IF NOT EXISTS tts_mode BOOLEAN DEFAULT FALSE",
            "ALTER TABLE bot_profiles ADD COLUMN IF NOT EXISTS include_summaries BOOLEAN DEFAULT TRUE",
            "ALTER TABLE bot_profiles ADD COLUMN IF NOT EXISTS include_in_global_search BOOLEAN DEFAULT TRUE",
            "ALTER TABLE bot_profiles ADD COLUMN IF NOT EXISTS default_voice VARCHAR(128)",
            "ALTER TABLE bot_profiles ADD COLUMN IF NOT EXISTS avatar VARCHAR(512)",
            "ALTER TABLE bot_profiles ADD COLUMN IF NOT EXISTS avatar_render TEXT",
            "ALTER TABLE bot_profiles ADD COLUMN IF NOT EXISTS prompt_override_id INTEGER",
        ]
        for stmt in migrations:
            conn.execute(text(stmt))

    def get(self, slug: str) -> BotProfile | None:
        if self.engine is None:
            return None
        normalized_slug = (slug or "").strip().lower()
        if not normalized_slug:
            return None
        with Session(self.engine) as session:
            return session.exec(
                select(BotProfile).where(BotProfile.slug == normalized_slug)
            ).first()

    def list_all(self) -> list[BotProfile]:
        if self.engine is None:
            return []
        with Session(self.engine) as session:
            return list(session.exec(select(BotProfile).order_by(BotProfile.slug)).all())

    def upsert(self, profile: BotProfile | dict[str, Any]) -> BotProfile:
        if self.engine is None:
            raise RuntimeError("Bot profiles DB unavailable")

        if isinstance(profile, BotProfile):
            payload = profile.model_dump()
        else:
            payload = dict(profile)

        slug = str(payload.get("slug", "")).strip().lower()
        if not slug:
            raise ValueError("Bot profile slug is required")

        now = datetime.now(timezone.utc)
        with Session(self.engine) as session:
            row = session.exec(select(BotProfile).where(BotProfile.slug == slug)).first()
            payload_agent_backend = payload.get("agent_backend") if "agent_backend" in payload else None
            if isinstance(payload_agent_backend, str):
                payload_agent_backend = payload_agent_backend.strip().lower() or None
            if row is None:
                row = BotProfile(
                    slug=slug,
                    name=str(payload.get("name") or slug.title()),
                    description=str(payload.get("description") or ""),
                    system_prompt=str(payload.get("system_prompt") or "You are a helpful assistant."),
                    prompt_override_id=payload.get("prompt_override_id"),
                    requires_memory=bool(payload.get("requires_memory", True)),
                    voice_optimized=bool(payload.get("voice_optimized", False)),
                    tts_mode=bool(payload.get("tts_mode", False)),
                    include_summaries=bool(payload.get("include_summaries", True)),
                    include_in_global_search=bool(payload.get("include_in_global_search", True)),
                    uses_tools=bool(payload.get("uses_tools", False)),
                    uses_search=bool(payload.get("uses_search", False)),
                    uses_home_assistant=bool(payload.get("uses_home_assistant", False)),
                    default_model=payload.get("default_model"),
                    harness=payload.get("harness"),
                    endpoint_id=payload.get("endpoint_id"),
                    color=payload.get("color"),
                    avatar=payload.get("avatar"),
                    avatar_render=payload.get("avatar_render"),
                    default_voice=payload.get("default_voice"),
                    nextcloud_config=payload.get("nextcloud_config"),
                    bot_type=normalize_bot_type(payload.get("bot_type"), payload_agent_backend),
                    agent_backend=payload_agent_backend,
                    agent_backend_config=payload.get("agent_backend_config"),
                    created_at=now,
                    updated_at=now,
                )
            else:
                row.name = str(payload.get("name", row.name) or row.name)
                row.description = str(payload.get("description", row.description) or "")
                row.system_prompt = str(payload.get("system_prompt", row.system_prompt) or row.system_prompt)
                if "prompt_override_id" in payload:
                    row.prompt_override_id = payload.get("prompt_override_id")
                row.requires_memory = bool(payload.get("requires_memory", row.requires_memory))
                row.voice_optimized = bool(payload.get("voice_optimized", row.voice_optimized))
                row.tts_mode = bool(payload.get("tts_mode", row.tts_mode))
                row.include_summaries = bool(payload.get("include_summaries", row.include_summaries))
                row.include_in_global_search = bool(payload.get("include_in_global_search", row.include_in_global_search))
                row.uses_tools = bool(payload.get("uses_tools", row.uses_tools))
                row.uses_search = bool(payload.get("uses_search", row.uses_search))
                row.uses_home_assistant = bool(payload.get("uses_home_assistant", row.uses_home_assistant))
                if payload.get("default_model") is not None:
                    row.default_model = payload["default_model"]
                if "harness" in payload:
                    row.harness = payload.get("harness")
                if "endpoint_id" in payload:
                    row.endpoint_id = payload.get("endpoint_id")
                if payload.get("color") is not None:
                    row.color = payload["color"]
                if "avatar" in payload:
                    row.avatar = payload.get("avatar")
                if "avatar_render" in payload:
                    row.avatar_render = payload.get("avatar_render")
                if "default_voice" in payload:
                    row.default_voice = payload.get("default_voice")
                if payload.get("nextcloud_config") is not None:
                    row.nextcloud_config = payload["nextcloud_config"]
                if "agent_backend" in payload:
                    row.agent_backend = payload_agent_backend
                    if payload_agent_backend is None and "agent_backend_config" not in payload:
                        row.agent_backend_config = {}
                if "agent_backend_config" in payload:
                    row.agent_backend_config = payload["agent_backend_config"]
                if "bot_type" in payload or "agent_backend" in payload or not row.bot_type:
                    # Derive bot_type from a SINGLE source of truth so the two
                    # fields can never drift (TASK-516):
                    #   - explicit bot_type in payload wins (caller intent),
                    #   - else if agent_backend was explicitly set/cleared,
                    #     derive from the NEW backend ALONE. Feeding the stale
                    #     row.bot_type back in made normalize_bot_type sticky
                    #     toward "agent", so clearing the backend (agent→chat)
                    #     left the row as agent-typed with no backend — an
                    #     invalid state that then 400s every later edit.
                    #   - else backfill from the existing row.
                    if "bot_type" in payload:
                        derived_type = payload.get("bot_type")
                    elif "agent_backend" in payload:
                        derived_type = None
                    else:
                        derived_type = row.bot_type
                    row.bot_type = normalize_bot_type(derived_type, row.agent_backend)
                row.updated_at = now

            # Single-source-of-truth invariant: a bot is an agent iff it has a
            # backend. Refuse to persist an agent-typed row with no backend so
            # the drift that stranded mira (TASK-516) can never land, no matter
            # which caller (route, CLI, seed, migration) reaches this store.
            if row.bot_type == "agent" and not row.agent_backend:
                # Message follows the DB-trigger convention ("bot <slug>: ...")
                # so routes/settings._humanize_bot_constraint_error lifts it
                # into a clean 400. Keep the "bot_type=" token — it's the marker
                # that humanizer matches on.
                raise ValueError(
                    f"bot {slug}: agent_backend is required when bot_type=agent "
                    "(a bot is an agent only if it has a backend)"
                )

            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    def delete(self, slug: str) -> bool:
        if self.engine is None:
            raise RuntimeError("Bot profiles DB unavailable")
        normalized_slug = (slug or "").strip().lower()
        if not normalized_slug:
            return False
        with Session(self.engine) as session:
            row = session.exec(
                select(BotProfile).where(BotProfile.slug == normalized_slug)
            ).first()
            if not row:
                return False
            session.delete(row)
            session.commit()
            return True

    def seed_from_yaml(self, bots_dict: dict[str, Any]) -> int:
        if self.engine is None:
            raise RuntimeError("Bot profiles DB unavailable")
        if not isinstance(bots_dict, dict):
            return 0

        seeded = 0
        for slug, bot_data in bots_dict.items():
            if not isinstance(bot_data, dict):
                continue
            payload = {
                "slug": slug,
                "name": bot_data.get("name", str(slug).title()),
                "description": bot_data.get("description", ""),
                "system_prompt": bot_data.get("system_prompt", "You are a helpful assistant."),
                "requires_memory": bot_data.get("requires_memory", True),
                "voice_optimized": bot_data.get("voice_optimized", False),
                "tts_mode": bot_data.get("tts_mode", False),
                "include_summaries": bot_data.get("include_summaries", True),
                "include_in_global_search": bot_data.get("include_in_global_search", True),
                "uses_tools": bot_data.get("uses_tools", False),
                "uses_search": bot_data.get("uses_search", False),
                "uses_home_assistant": bot_data.get("uses_home_assistant", False),
                "default_model": bot_data.get("default_model"),
                "harness": bot_data.get("harness"),
                "endpoint_id": bot_data.get("endpoint_id"),
                "nextcloud_config": bot_data.get("nextcloud"),
                "bot_type": normalize_bot_type(
                    bot_data.get("bot_type"),
                    bot_data.get("agent_backend"),
                ),
                "agent_backend": bot_data.get("agent_backend"),
                "agent_backend_config": bot_data.get("agent_backend_config"),
            }
            self.upsert(payload)
            seeded += 1
        return seeded


def purge_bot_data(config: Config, slug: str) -> dict[str, Any]:
    """Delete all data for a bot from the database.

    Drops the per-bot dynamic tables (_messages, _memories, _forgotten_messages)
    and removes rows from every shared table that references this bot.
    Does NOT delete the bot_profiles row — call BotProfileStore.delete() for that.
    """
    import re as _re

    from sqlalchemy import create_engine as _create_engine, text as _text

    normalized = (slug or "").strip().lower()
    if not normalized:
        return {"error": "Invalid slug"}
    if not has_database_credentials(config):
        return {"error": "No database credentials configured"}

    sanitized = _re.sub(r"[^a-z0-9_]", "", normalized) or "default"

    host = getattr(config, "POSTGRES_HOST", "localhost")
    port = int(getattr(config, "POSTGRES_PORT", 5432))
    user = getattr(config, "POSTGRES_USER", "llm_bawt")
    password = getattr(config, "POSTGRES_PASSWORD", "")
    database = getattr(config, "POSTGRES_DATABASE", "llm_bawt")
    from urllib.parse import quote_plus as _qp
    url = f"postgresql+psycopg2://{user}:{_qp(password)}@{host}:{port}/{database}"
    engine = _create_engine(url, echo=False)

    result: dict[str, Any] = {"slug": normalized, "dropped_tables": [], "deleted_rows": {}}

    try:
        with engine.connect() as conn:
            from .prompt_purge import PromptPurgeGuard
            prompt_purge = PromptPurgeGuard(conn)
            prompt_purge.guard({normalized})
            # 1. Delete per-bot rows from the LIST-partitioned parents (TASK-571).
            #    The old ``<bot>_messages`` shard tables no longer exist — dropping
            #    them was a silent no-op that left every message/memory/forgotten
            #    row alive while reporting success. The bot definition is preserved
            #    here, so DELETE the rows rather than DROP the partition (a drop
            #    would race the auto-provision on the bot's next write). ``bot_id``
            #    in the parents is the sanitized partition value.
            for parent in ("messages", "memories", "forgotten_messages"):
                r = conn.execute(
                    _text(f"DELETE FROM {parent} WHERE bot_id=:s"),
                    {"s": sanitized},
                )
                result["deleted_rows"][parent] = r.rowcount

            # 2. runtime_settings
            r = conn.execute(
                _text("DELETE FROM runtime_settings WHERE scope_type='bot' AND scope_id=:s"),
                {"s": normalized},
            )
            result["deleted_rows"]["runtime_settings"] = r.rowcount

            # 3. entity_profile_attributes (before entity_profiles — no FK cascade)
            r = conn.execute(
                _text("DELETE FROM entity_profile_attributes WHERE entity_type::text='bot' AND entity_id=:s"),
                {"s": normalized},
            )
            result["deleted_rows"]["entity_profile_attributes"] = r.rowcount

            # 4. entity_profiles
            r = conn.execute(
                _text("DELETE FROM entity_profiles WHERE entity_type::text='bot' AND entity_id=:s"),
                {"s": normalized},
            )
            result["deleted_rows"]["entity_profiles"] = r.rowcount

            # 5–6. Prompt companions precede the job_runs/scheduled_jobs parents.
            result["deleted_rows"].update(prompt_purge.delete_jobs(normalized))

            # 7. prompt_template_versions
            r = conn.execute(
                _text("DELETE FROM prompt_template_versions WHERE scope_type='bot' AND scope_id=:s"),
                {"s": normalized},
            )
            result["deleted_rows"]["prompt_template_versions"] = r.rowcount

            # 8. prompt_templates
            r = conn.execute(
                _text("DELETE FROM prompt_templates WHERE scope_type='bot' AND scope_id=:s"),
                {"s": normalized},
            )
            result["deleted_rows"]["prompt_templates"] = r.rowcount

            # 9. tool_call_records
            r = conn.execute(
                _text("DELETE FROM tool_call_records WHERE bot_id=:s"),
                {"s": normalized},
            )
            result["deleted_rows"]["tool_call_records"] = r.rowcount

            # 10. turn_logs
            r = conn.execute(
                _text("DELETE FROM turn_logs WHERE bot_id=:s"),
                {"s": normalized},
            )
            result["deleted_rows"]["turn_logs"] = r.rowcount

            conn.commit()
    finally:
        engine.dispose()

    # Evict cached SQLAlchemy table objects so they aren't revived on next access
    try:
        from .memory.postgresql import (
            _memory_table_cache,
            _message_table_cache,
            _forgotten_table_cache,
        )
        # Cache keys are the partition table names (``messages_p_<bot>`` etc.,
        # TASK-571), not the legacy ``<bot>_messages`` form.
        _evict_keys = {
            f"messages_p_{sanitized}",
            f"memories_p_{sanitized}",
            f"forgotten_messages_p_{sanitized}",
        }
        for cache in (_memory_table_cache, _message_table_cache, _forgotten_table_cache):
            for key in list(cache.keys()):
                if key in _evict_keys:
                    del cache[key]
    except Exception:
        pass

    return result


def purge_orphaned_shared_rows(config: Config, dry_run: bool = False) -> dict[str, Any]:
    """Separate shared-row purge; never called by the partition cleanup route.

    Shared-row deletion requires its own review and authorization.
    """
    from sqlalchemy import text as _text
    from .bot_identities import RESERVED_JOB_BOT_IDS
    from .orphan_partitions import PROTECTED_STORAGE_IDS
    from .utils.db import get_shared_engine

    if not has_database_credentials(config):
        return {"error": "No database credentials configured"}

    engine = get_shared_engine(config)
    protected = RESERVED_JOB_BOT_IDS | PROTECTED_STORAGE_IDS

    result: dict[str, Any] = {
        "dry_run": dry_run,
        "orphaned_bot_ids": [],
        "dropped_tables": [],
        "deleted_rows": {},
    }

    with engine.connect() as conn:
        # Known slugs from bot_profiles
        rows = conn.execute(_text("SELECT slug FROM bot_profiles")).fetchall()
        known_slugs: set[str] = {r[0].strip().lower() for r in rows}
        known_slugs |= protected

        # --- Shared tables only ---
        table_rows = conn.execute(
            _text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public' AND table_type='BASE TABLE'"
            )
        ).fetchall()
        all_tables = {r[0] for r in table_rows}
        from .prompt_purge import PromptPurgeGuard
        prompt_purge = PromptPurgeGuard(conn, all_tables)
        if not dry_run:
            # Guard shared-row deletion and stabilize affected producers before
            # discovering orphan identities.
            prompt_purge.guard(set())
            candidates = set()
            for table, column in (("scheduled_jobs", "bot_id"),
                                  ("inter_bot_deliveries", "target_bot_id")):
                if table in all_tables:
                    candidates.update(row[0] for row in conn.execute(
                        _text(f"SELECT DISTINCT {column} FROM {table}")
                    ) if row[0] and row[0] != "*" and row[0].strip().lower() not in known_slugs)
            prompt_purge.guard(candidates)

        # No partition or legacy-shard DDL belongs in shared-row purge.
        # Partition cleanup has its own bounds-based planner and executor.
        # --- Shared tables ---
        # bot_id column tables
        bot_id_tables = [
            "runtime_settings",  # scope_type='bot', scope_id
            "turn_logs",
            "tool_call_records",
            "scheduled_jobs",
            "entity_profiles",
            "entity_profile_attributes",
            "prompt_templates",
            "prompt_template_versions",
        ]

        def _find_orphan_ids(table: str, id_col: str, extra_where: str = "") -> set[str]:
            where = f"WHERE {extra_where}" if extra_where else ""
            q = f"SELECT DISTINCT {id_col} FROM {table} {where}"
            rs = conn.execute(_text(q)).fetchall()
            return {r[0] for r in rs if r[0] and r[0].strip().lower() not in known_slugs}

        def _delete_orphans(table: str, id_col: str, ids: set[str], extra_where: str = "") -> int:
            if not ids:
                return 0
            and_clause = f" AND {extra_where}" if extra_where else ""
            total = 0
            for oid in ids:
                r = conn.execute(
                    _text(f"DELETE FROM {table} WHERE {id_col}=:v{and_clause}"),
                    {"v": oid},
                )
                total += r.rowcount
            return total

        # Accumulate all orphaned bot IDs across shared tables
        orphan_ids: set[str] = set()

        for table, id_col, extra in [
            ("turn_logs", "bot_id", ""),
            ("tool_call_records", "bot_id", ""),
            ("scheduled_jobs", "bot_id", "bot_id != '*'"),
            ("entity_profiles", "entity_id", "entity_type::text = 'bot'"),
            ("entity_profile_attributes", "entity_id", "entity_type::text = 'bot'"),
            ("prompt_templates", "scope_id", "scope_type='bot'"),
            ("prompt_template_versions", "scope_id", "scope_type='bot'"),
        ]:
            if table not in all_tables:
                continue
            orphans = _find_orphan_ids(table, id_col, extra)
            orphan_ids |= orphans
            if not dry_run and orphans:
                if table == "scheduled_jobs":
                    for oid in orphans:
                        for child, count in prompt_purge.delete_jobs(oid).items():
                            result["deleted_rows"][child] = result["deleted_rows"].get(child, 0) + count
                    continue
                cnt = _delete_orphans(table, id_col, orphans, extra)
                result["deleted_rows"][table] = result["deleted_rows"].get(table, 0) + cnt
            elif dry_run:
                result["deleted_rows"][table] = len(orphans)

        # runtime_settings uses scope_id not bot_id
        if "runtime_settings" in all_tables:
            rs = conn.execute(
                _text("SELECT DISTINCT scope_id FROM runtime_settings WHERE scope_type='bot'")
            ).fetchall()
            orphan_rs = {r[0] for r in rs if r[0] and r[0].strip().lower() not in known_slugs}
            orphan_ids |= orphan_rs
            if not dry_run and orphan_rs:
                cnt = _delete_orphans("runtime_settings", "scope_id", orphan_rs, "scope_type='bot'")
                result["deleted_rows"]["runtime_settings"] = cnt
            elif dry_run:
                result["deleted_rows"]["runtime_settings"] = len(orphan_rs)

        result["orphaned_bot_ids"] = sorted(orphan_ids)

        if not dry_run:
            conn.commit()

    return result


def cleanup_orphaned_bot_data(config: Config, dry_run: bool = True,
                              manifest_digest: str | None = None) -> dict[str, Any]:
    """Preview or execute only orphan partitions; never purge shared rows."""
    from .orphan_partitions import OrphanPartitionPlanner, PartitionManifestExecutor
    from .utils.db import get_shared_engine
    from .utils.paths import resolve_log_dir

    if not has_database_credentials(config):
        return {"error": "No database credentials configured"}
    engine = get_shared_engine(config)
    plan = OrphanPartitionPlanner(engine).plan()
    manifest = plan.manifest()
    if dry_run:
        return {"dry_run": True, "manifest": manifest}
    if not manifest_digest or manifest_digest != plan.content_digest:
        raise ValueError("Manifest digest mismatch; preview again")
    # One exclusive journal per digest. Never automatically replay a partial run.
    journal = resolve_log_dir() / "orphan-partitions" / f"{plan.content_digest}.jsonl"
    outcomes = PartitionManifestExecutor(engine).execute(
        plan, manifest_digest, journal_path=journal,
    )
    return {"dry_run": False, "manifest": manifest, "outcomes": outcomes,
            "status": "completed", "committed_count": len(outcomes),
            "journal_incomplete": False, "journal_error": None,
            "journal": str(journal) if outcomes else None}


class RuntimeSettingsStore:
    """Low-level DB access for runtime settings."""

    _schema_guard = SchemaBootstrapGuard()

    def __init__(self, config: Config):
        self.config = config
        self.engine = None
        if not has_database_credentials(config):
            return

        try:
            host = getattr(config, "POSTGRES_HOST", "localhost")
            port = int(getattr(config, "POSTGRES_PORT", 5432))
            user = getattr(config, "POSTGRES_USER", "llm_bawt")
            password = getattr(config, "POSTGRES_PASSWORD", "")
            database = getattr(config, "POSTGRES_DATABASE", "llm_bawt")
            encoded_password = quote_plus(password)
            from .utils.db import get_shared_engine
            self.engine = get_shared_engine(config)  # TASK-202: shared pool
            if self.engine is None:
                return
            self._ensure_tables_exist()
        except Exception as e:
            self.engine = None
            logger.warning("Runtime settings DB unavailable: %s", e)

    def _ensure_tables_exist(self) -> None:
        if self.engine is None:
            return
        def bootstrap(conn) -> None:
            SQLModel.metadata.create_all(
                bind=conn,
                tables=[RuntimeSetting.__table__, BotProfile.__table__],
            )
            conn.execute(
                text(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_runtime_settings_scope_key
                    ON runtime_settings (scope_type, scope_id, key)
                    """
                )
            )

        self._schema_guard.run(self.engine, "runtime-settings-store", bootstrap)

    def get_scope_settings(self, scope_type: str, scope_id: str) -> dict[str, Any]:
        """Return all settings for a scope."""
        if self.engine is None:
            return {}
        with Session(self.engine) as session:
            rows = session.exec(
                select(RuntimeSetting).where(
                    RuntimeSetting.scope_type == scope_type,
                    RuntimeSetting.scope_id == scope_id,
                )
            ).all()
        out: dict[str, Any] = {}
        for row in rows:
            try:
                out[row.key] = json.loads(row.value_json)
            except Exception:
                logger.warning(
                    "Invalid runtime setting JSON for %s/%s key=%s",
                    scope_type,
                    scope_id,
                    row.key,
                )
        return out

    def set_value(self, scope_type: str, scope_id: str, key: str, value: Any) -> None:
        """Upsert setting value."""
        if self.engine is None:
            raise RuntimeError("Runtime settings DB unavailable")
        value_json = json.dumps(value, ensure_ascii=False)
        with Session(self.engine) as session:
            row = session.exec(
                select(RuntimeSetting).where(
                    RuntimeSetting.scope_type == scope_type,
                    RuntimeSetting.scope_id == scope_id,
                    RuntimeSetting.key == key,
                )
            ).first()
            if row:
                row.value_json = value_json
                row.updated_at = datetime.now(timezone.utc)
            else:
                row = RuntimeSetting(
                    scope_type=scope_type,
                    scope_id=scope_id,
                    key=key,
                    value_json=value_json,
                )
            session.add(row)
            session.commit()

    def delete_value(self, scope_type: str, scope_id: str, key: str) -> bool:
        """Delete one setting value. Returns True if deleted."""
        if self.engine is None:
            raise RuntimeError("Runtime settings DB unavailable")
        with Session(self.engine) as session:
            row = session.exec(
                select(RuntimeSetting).where(
                    RuntimeSetting.scope_type == scope_type,
                    RuntimeSetting.scope_id == scope_id,
                    RuntimeSetting.key == key,
                )
            ).first()
            if not row:
                return False
            session.delete(row)
            session.commit()
            return True


class RuntimeSettingsResolver:
    """Resolve runtime settings with layered precedence and short-lived cache."""

    def __init__(
        self,
        config: Config,
        bot: Bot | None = None,
        bot_id: str | None = None,
        cache_ttl_seconds: float = 5.0,
    ):
        self.config = config
        self.bot = bot
        self.bot_id = (bot_id or getattr(bot, "slug", "") or "").strip().lower() or None
        self.cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))
        self.store = RuntimeSettingsStore(config)
        self._bot_cache: dict[str, Any] = {}
        self._global_cache: dict[str, Any] = {}
        self._cache_loaded_at: float = 0.0

    def _refresh_cache_if_needed(self) -> None:
        if self.store.engine is None:
            return
        now = time.time()
        if self._cache_loaded_at and (now - self._cache_loaded_at) < self.cache_ttl_seconds:
            return
        self._global_cache = self.store.get_scope_settings("global", "*")
        self._bot_cache = (
            self.store.get_scope_settings("bot", self.bot_id)
            if self.bot_id
            else {}
        )
        self._cache_loaded_at = now

    def resolve(
        self,
        key: str,
        fallback: Any = None,
        request_overrides: dict[str, Any] | None = None,
    ) -> Any:
        """Resolve a setting value using precedence chain."""
        return self.resolve_with_source(
            key, fallback=fallback, request_overrides=request_overrides
        )[0]

    def resolve_with_source(
        self,
        key: str,
        fallback: Any = None,
        request_overrides: dict[str, Any] | None = None,
    ) -> tuple[Any, str]:
        """Resolve a setting AND report which layer supplied it (TASK-487).

        Returns ``(value, source)`` where source is one of:
        ``request_override`` | ``bot_override`` | ``global_override`` |
        ``code_default`` | ``unset``. Precedence matches ``resolve``; this is
        the provenance-carrying sibling the effective-config inspector reads
        through so "why is this value what it is" is answerable without
        re-deriving the precedence chain by hand.
        """
        if request_overrides and key in request_overrides:
            return request_overrides[key], "request_override"

        self._refresh_cache_if_needed()
        if key in self._bot_cache:
            return self._bot_cache[key], "bot_override"
        if key in self._global_cache:
            return self._global_cache[key], "global_override"

        if fallback is not None:
            return fallback, "code_default"
        return None, "unset"

    def resolve_from_config_attr(
        self,
        key: str,
        config_attr: str,
        request_overrides: dict[str, Any] | None = None,
    ) -> Any:
        """Resolve setting using config attribute as final fallback."""
        fallback = getattr(self.config, config_attr, None)
        return self.resolve(key=key, fallback=fallback, request_overrides=request_overrides)


def resolve_job_model(config: Config, key: str) -> str | None:
    """Resolve a background-job model alias from the GLOBAL runtime setting.

    Background-job models (history summarization, memory extraction, profile
    maintenance) are system-wide, not per-bot (TASK-522), so this resolves at
    **global scope only** — a fresh ``RuntimeSettingsResolver`` with no bot, so
    a stray bot-scoped row can never shadow the global value. Falls back to the
    key's declared ``SETTING_DEFINITIONS`` default when no global row exists.

    These keys retired the old env/config vars (SUMMARIZATION_MODEL /
    EXTRACTION_MODEL / MAINTENANCE_MODEL / PROFILE_MAINTENANCE_MODEL); call sites
    keep their cascade by chaining ``resolve_job_model(config, k) or ...``.

    Returns the alias string, or ``None`` (empty string normalized to None) so
    an unset key falls through a caller's ``or`` cascade.
    """
    from .setting_definitions import SETTING_DEFINITIONS

    definition = SETTING_DEFINITIONS.get(key)
    default = definition.default if definition is not None else None
    resolver = RuntimeSettingsResolver(config=config, bot=None)
    value = resolver.resolve(key, fallback=default)
    return value or None
