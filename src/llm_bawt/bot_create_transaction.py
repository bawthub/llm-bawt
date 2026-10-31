"""Single commit boundary for a complete new bot profile, tags, and overrides."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException
from sqlalchemy import inspect, text
from sqlmodel import Session

from .runtime_settings import BotProfile, BotProfileStore, ProfileWriteGuard, RuntimeSetting
from .setting_definitions import HISTORY_SCOPES, SETTING_DEFINITIONS, STORAGE_RUNTIME_SETTING
from .utils.bot_partition import lock_bot_storage, sanitize_table_name


class BotCreateTransaction:
    """Persist a complete create under the profile's existing admission/tag lock.

    Runtime settings and bot tags share the profile transaction. A table lock is
    deliberately brief and create-only: legacy settings writers do not take the
    per-bot lock, and a deleted profile may leave bot-scoped settings behind.
    Refuse that pre-existing identity rather than silently importing its options.
    """

    def __init__(self, store: BotProfileStore):
        if store.engine is None:
            raise HTTPException(status_code=503, detail="Bot profiles DB unavailable")
        self.store = store

    @staticmethod
    def _refuse_retained_identity(conn, slug: str, storage_id: str) -> None:
        """A create may not inherit a deleted bot's audit, settings or stored turns.

        Partition rows use a lossy sanitized ID; profile, tag, setting and audit
        slugs can also identify a *different* bot that maps to that same ID.
        Check both forms under the create/purge storage lock before any write.
        """
        tables = set(inspect(conn).get_table_names())
        identities = (
            ("bot_profiles", "slug", None),
            ("bot_profile_tags", "bot_slug", None),
            ("bot_profile_changes", "bot_slug", None),
            ("runtime_settings", "scope_id", "scope_type='bot'"),
        )
        for table, column, qualifier in identities:
            if table not in tables:
                continue
            where = (f"{qualifier} AND " if qualifier else "")
            # Keep this SQL mapping identical to sanitize_table_name() above.
            collision = conn.execute(text(
                f"SELECT EXISTS (SELECT 1 FROM {table} WHERE {where}"
                f"COALESCE(NULLIF(regexp_replace(lower({column}), '[^a-z0-9_]', '', 'g'), ''),"
                "'default')=:storage)"
            ), {"storage": storage_id}).scalar_one()
            if collision:
                raise HTTPException(status_code=409, detail="Bot slug has an existing or retained storage identity")

        rows = (
            ("messages", "bot_id", storage_id),
            ("memories", "bot_id", storage_id),
            ("forgotten_messages", "bot_id", storage_id),
            ("sessions", "bot_id", slug),
            ("turn_logs", "bot_id", slug),
            ("tool_call_records", "bot_id", slug),
            ("working_memory_namespaces", "bot_id", slug),
            ("inter_bot_deliveries", "target_bot_id", slug),
            ("inter_bot_deliveries", "sender_bot_id", slug),
            ("scheduled_jobs", "bot_id", slug),
        )
        for table, column, value in rows:
            if table not in tables:
                continue
            occupied = conn.execute(text(
                f"SELECT EXISTS (SELECT 1 FROM {table} WHERE {column}=:value)"
            ), {"value": value}).scalar_one()
            if occupied:
                raise HTTPException(status_code=409, detail="Bot slug has retained data; choose a fresh slug")

    def create(
        self,
        payload: dict[str, object],
        overrides: dict[str, Any],
        guard: ProfileWriteGuard,
        validate_writes: Callable[[Any, list[tuple[str, str, str, object]]], None],
    ) -> BotProfile:
        with Session(self.store.engine, expire_on_commit=False) as session:
            conn = session.connection()
            slug = str(payload["slug"]).strip().lower()
            storage_id = sanitize_table_name(slug)
            lock_bot_storage(conn, storage_id)
            self._refuse_retained_identity(conn, slug, storage_id)
            profile = self.store.upsert(payload, guard=guard, session=session, create_only=True)
            slug = profile.slug
            if conn.dialect.name == "postgresql":
                conn.execute(text("LOCK TABLE runtime_settings IN SHARE ROW EXCLUSIVE MODE"))
            orphaned = conn.execute(text(
                "SELECT EXISTS (SELECT 1 FROM runtime_settings "
                "WHERE scope_type='bot' AND scope_id=:slug)"
            ), {"slug": slug}).scalar_one()
            if orphaned:
                raise HTTPException(status_code=409, detail="Existing bot settings require cleanup before this slug can be reused")

            persona_id = profile.prompt_override_id
            if persona_id is not None:
                persona = conn.execute(text(
                    "SELECT 1 FROM prompt_templates WHERE id=:id AND category='persona' "
                    "AND scope_type='global' FOR KEY SHARE"
                ), {"id": persona_id}).first()
                if persona is None:
                    raise HTTPException(status_code=422, detail="Selected persona is not available")

            overrides = dict(overrides)
            scope = overrides.get("history_scope")
            continuity = overrides.get("session_memory_continuity")
            if scope is not None:
                mirrored = scope != "none"
                if continuity is not None and continuity != mirrored:
                    raise HTTPException(status_code=422, detail="session_memory_continuity must mirror history_scope")
                overrides["session_memory_continuity"] = mirrored

            writes = []
            for key, value in overrides.items():
                definition = SETTING_DEFINITIONS.get(key)
                if (definition is None or definition.storage != STORAGE_RUNTIME_SETTING
                        or not definition.applies_to):
                    raise HTTPException(status_code=422, detail=f"Not an editable bot setting: {key}")
                if value is None:  # new bot: null/omission means inherit, not a stored JSON null
                    continue
                if profile.bot_type not in definition.applies_to:
                    raise HTTPException(status_code=422, detail=f"Not an editable bot setting: {key}")
                valid_types = {
                    "bool": lambda v: isinstance(v, bool),
                    "int": lambda v: type(v) is int,
                    "float": lambda v: type(v) in (int, float),
                    "str": lambda v: isinstance(v, str),
                    "string_list": lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v),
                    "json": lambda v: isinstance(v, (dict, list)),
                }
                checker = valid_types.get(definition.type)
                if checker is None or not checker(value):
                    raise HTTPException(status_code=422, detail=f"Invalid value for bot setting: {key}")
                if key == "history_scope" and value not in HISTORY_SCOPES:
                    raise HTTPException(status_code=422, detail="Invalid history_scope")
                try:
                    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
                except (TypeError, ValueError) as exc:
                    raise HTTPException(status_code=422, detail=f"Invalid JSON for bot setting: {key}") from exc
                writes.append(("bot", slug, key, value, encoded))

            validate_writes(conn, [(scope, bot, key, value) for scope, bot, key, value, _ in writes])
            for scope, bot, key, _value, encoded in writes:
                session.add(RuntimeSetting(scope_type=scope, scope_id=bot, key=key, value_json=encoded))
            session.commit()
            return profile
