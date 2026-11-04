"""Retired bot slugs that still point at a live bot (TASK-1900).

An alias reserves an old slug for the bot that used to carry it, so an old
link resolves to that bot and a new bot can never inherit it. Keys are the
lossy storage identity (``sanitize_table_name``), the same form create uses
to refuse retained identities, so cleaned-name clashes are reserved too.

``bot_slug`` references ``bot_profiles.slug`` with ON DELETE RESTRICT and
ON UPDATE CASCADE: a later rename of the target carries its aliases, and a
delete of a bot that still has aliases is refused instead of orphaning them.

Rows are operator data; product code only reads them. The insert trigger
takes the same advisory lock as bot create and refuses an alias whose key a
live profile already occupies, so a create and an alias insert serialize.

Operator procedure: insert or remove alias rows only while no delete or purge
of the target bot is running. The delete route checks aliases before purging
and again inside the profile delete, but those are separate transactions; an
alias inserted between them leaves the bot's data purged and its profile
refused (never orphaned). Alias writes are rare manual steps, so this is a
procedural exclusion, not online rename machinery.
"""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import text

ALIAS_TABLE = "bot_slug_aliases"

# Keep identical to sanitize_table_name() and BotCreateTransaction's mapping.
_SQL_STORAGE_ID = (
    "COALESCE(NULLIF(regexp_replace(lower({col}), '[^a-z0-9_]', '', 'g'), ''), 'default')"
)

_DDL = f"""
CREATE TABLE IF NOT EXISTS {ALIAS_TABLE} (
  old_slug    VARCHAR(128) PRIMARY KEY
              CHECK (old_slug = {_SQL_STORAGE_ID.format(col='old_slug')}),
  bot_slug    VARCHAR(128) NOT NULL
              REFERENCES bot_profiles (slug) ON UPDATE CASCADE ON DELETE RESTRICT,
  reason      TEXT         NULL,
  created_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
)
"""
_INDEX_DDL = f"CREATE INDEX IF NOT EXISTS {ALIAS_TABLE}_bot_idx ON {ALIAS_TABLE} (bot_slug)"

_GUARD_FN = f"""
CREATE OR REPLACE FUNCTION {ALIAS_TABLE}_guard() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  PERFORM pg_advisory_xact_lock(hashtext('bot-storage'), hashtext(NEW.old_slug));
  IF EXISTS (SELECT 1 FROM bot_profiles
             WHERE {_SQL_STORAGE_ID.format(col='slug')} = NEW.old_slug) THEN
    RAISE EXCEPTION 'bot slug % is a live bot; an alias may only reserve a retired slug', NEW.old_slug
      USING ERRCODE = 'unique_violation';
  END IF;
  RETURN NEW;
END $$
"""
_GUARD_TRIGGER = (
    f"DROP TRIGGER IF EXISTS {ALIAS_TABLE}_guard ON {ALIAS_TABLE}; "
    f"CREATE TRIGGER {ALIAS_TABLE}_guard BEFORE INSERT OR UPDATE OF old_slug ON {ALIAS_TABLE} "
    f"FOR EACH ROW EXECUTE FUNCTION {ALIAS_TABLE}_guard()"
)


class BotHasAliases(Exception):
    """A profile delete was refused because retired slugs still point at it."""

    status_code = 409

    def __init__(self, slug: str, aliases: Iterable[str]):
        self.slug = slug
        self.aliases = sorted(aliases)
        super().__init__(
            f"{slug} still has retired slug aliases ({', '.join(self.aliases)}); remove them first")


def ensure_schema(conn) -> None:
    conn.execute(text(_DDL))
    conn.execute(text(_INDEX_DDL))
    if conn.dialect.name == "postgresql":
        conn.execute(text(_GUARD_FN))
        conn.execute(text(_GUARD_TRIGGER))


def _exists(conn) -> bool:
    from sqlalchemy import inspect

    return ALIAS_TABLE in set(inspect(conn).get_table_names())


def aliases_by_bot(conn) -> dict[str, list[str]]:
    """Every alias grouped by target bot slug, sorted. Raises on DB failure."""
    if not _exists(conn):
        return {}
    grouped: dict[str, list[str]] = {}
    for old, bot in conn.execute(text(
        f"SELECT old_slug, bot_slug FROM {ALIAS_TABLE} ORDER BY old_slug"
    )):
        grouped.setdefault(bot, []).append(old)
    return grouped


def aliases_for(conn, bot_slug: str) -> list[str]:
    if not _exists(conn):
        return []
    return list(conn.execute(text(
        f"SELECT old_slug FROM {ALIAS_TABLE} WHERE bot_slug=:bot ORDER BY old_slug"
    ), {"bot": bot_slug}).scalars())


def storage_id_reserved(conn, storage_id: str) -> bool:
    """True when a retired slug with this storage identity is reserved."""
    if not _exists(conn):
        return False
    return bool(conn.execute(text(
        f"SELECT EXISTS (SELECT 1 FROM {ALIAS_TABLE} WHERE old_slug=:id)"
    ), {"id": storage_id}).scalar_one())
