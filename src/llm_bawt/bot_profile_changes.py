"""Bot profile write preconditions and change audit (TASK-1521).

Everything here runs inside :meth:`BotProfileStore.upsert`'s transaction, after
the write guard has locked the bot and the row has been read, so a refusal
writes nothing and an audit row commits only with the change it records.

Two preconditions, both optional and both carried in the upsert payload:

- ``expected_fields``: the values a caller read for the fields it is changing.
  Any difference means another writer got there first: :class:`ProfileFieldsChanged`.
- ``_expected_row_version``: the ``updated_at`` of the row a PATCH merged onto.
  The PATCH route sends the whole merged row, so a stale merge would silently
  revert a concurrent edit to a field this caller never touched. A mismatch
  raises :class:`ProfileRowChanged` and the route re-merges on the fresh row.

``bot_profile_changes`` keeps one row per write that changed anything, with
the before/after of each changed field (tags included) and who made it.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql.expression import bindparam

EXPECTED_FIELDS = "expected_fields"
ROW_VERSION = "_expected_row_version"
CHANGED_BY = "_changed_by"
CHANGE_NOTE = "change_note"
CHANGE_TABLE = "bot_profile_changes"

# What a change record covers: the profile's editable columns plus its tags.
# ``avatar_render`` is derived from ``avatar``; timestamps are bookkeeping.
AUDITED_FIELDS: tuple[str, ...] = (
    "name", "description", "system_prompt", "prompt_override_id", "requires_memory",
    "voice_optimized", "tts_mode", "include_summaries", "include_in_global_search",
    "uses_tools", "uses_search", "uses_home_assistant", "default_model", "harness",
    "endpoint_id", "color", "avatar", "default_voice", "nextcloud_config", "bot_type",
    "agent_backend", "agent_backend_config",
)
TAGS = "tags"

_DDL = f"""
CREATE TABLE IF NOT EXISTS {CHANGE_TABLE} (
  id          BIGSERIAL    PRIMARY KEY,
  bot_slug    VARCHAR(128) NOT NULL,
  changed_by  VARCHAR(128) NOT NULL,
  change_note TEXT         NULL,
  changes     JSONB        NOT NULL CHECK (jsonb_typeof(changes) = 'object'),
  created_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
)
"""
_INDEX_DDL = (f"CREATE INDEX IF NOT EXISTS {CHANGE_TABLE}_bot_idx "
              f"ON {CHANGE_TABLE} (bot_slug, created_at DESC)")

_TAGS_SQL = text("SELECT tag_slug FROM bot_profile_tags WHERE bot_slug = :bot ORDER BY tag_slug")
_INSERT_SQL = text(f"""
    INSERT INTO {CHANGE_TABLE} (bot_slug, changed_by, change_note, changes)
    VALUES (:bot, :by, :note, :changes)
""").bindparams(bindparam("changes", type_=JSONB))


class ProfileWriteConflict(Exception):
    """A precondition failed under the bot lock; nothing was written."""

    status_code = 409
    code = "profile_changed"

    def __init__(self, slug: str, current: Mapping[str, Any], message: str):
        super().__init__(message)
        self.slug = slug
        self.current = dict(current)


class ProfileFieldsChanged(ProfileWriteConflict):
    code = "fields_changed"


class ProfileRowChanged(ProfileWriteConflict):
    code = "row_changed"


def ensure_schema(conn) -> None:
    conn.execute(text(_DDL))
    conn.execute(text(_INDEX_DDL))


def check_preconditions(row, payload: Mapping[str, object]) -> None:
    """Refuse a write whose caller read a different row than the locked one."""
    slug = str(payload.get("slug") or "")
    version = payload.get(ROW_VERSION)
    if version is not None and (row is None or row.updated_at != version):
        raise ProfileRowChanged(slug, {}, f"{slug} changed since it was read; nothing was saved")
    expected = payload.get(EXPECTED_FIELDS)
    if not expected:
        return
    current = {field: getattr(row, field, None) if row is not None else None for field in expected}
    if any(current[field] != value for field, value in expected.items()):
        raise ProfileFieldsChanged(
            slug, current, f"{slug}'s {', '.join(sorted(expected))} changed since read; nothing was saved")


def snapshot(conn, row) -> dict[str, Any]:
    """The audited state of one profile row, tags read in the same transaction."""
    if row is None:
        return {}
    state = {field: getattr(row, field, None) for field in AUDITED_FIELDS}
    state[TAGS] = list(conn.execute(_TAGS_SQL, {"bot": row.slug}).scalars())
    return state


def record_change(conn, slug: str, before: Mapping[str, Any], after: Mapping[str, Any],
                  payload: Mapping[str, object], default_by: str) -> dict[str, Any]:
    """Insert the change row when anything changed; return the per-field diff."""
    changes = {
        field: {"before": before.get(field), "after": after.get(field)}
        for field in (*AUDITED_FIELDS, TAGS)
        if before.get(field) != after.get(field)
    }
    if changes:
        conn.execute(_INSERT_SQL, {
            "bot": slug,
            "by": str(payload.get(CHANGED_BY) or default_by),
            "note": payload.get(CHANGE_NOTE) or None,
            "changes": changes,
        })
    return changes
