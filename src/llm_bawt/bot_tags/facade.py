"""One validated tag snapshot, a fail-closed sync facade and the SQL predicate.

Effective rules, everywhere (BOT_TAGS.md §3, §4.1, §5.2):
- private = ``private`` tag OR ``include_in_global_search = false``;
- private implies no_global_search;
- a bot with no profile row is private;
- a snapshot that could not be read treats every bot as private AND inactive;
- an opt-in tag (``external``) fails closed the other way: an unread snapshot,
  or a catalog not yet seeded with it, grants it to nobody (TASK-1362).
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from sqlalchemy import text

from .vocabulary import OPT_IN_BEHAVIOR_TAGS, TAG_SLUG, BehaviorTag, is_canonical_bot_slug

logger = logging.getLogger(__name__)

MAX_BOTS = 256

# One statement, one snapshot: profiles, their tags and the catalog slugs. Every
# row repeats the (small) catalog so recall's single-statement reader and the
# resolver validate against exactly the same view.
SNAPSHOT_SQL = """
    SELECT p.slug, p.include_in_global_search,
           COALESCE(array_agg(t.tag_slug ORDER BY t.tag_slug)
                    FILTER (WHERE t.tag_slug IS NOT NULL), '{}') AS tags,
           (SELECT COALESCE(array_agg(c.slug ORDER BY c.slug), '{}')
            FROM bot_tag_catalog c) AS catalog
    FROM bot_profiles p LEFT JOIN bot_profile_tags t ON t.bot_slug = p.slug
    GROUP BY p.slug, p.include_in_global_search ORDER BY p.slug LIMIT 257
"""

_PRIVATE = BehaviorTag.PRIVATE
_NO_GLOBAL = BehaviorTag.NO_GLOBAL_SEARCH
_INACTIVE = BehaviorTag.INACTIVE
_BEHAVIOR = frozenset(str(t) for t in BehaviorTag)
_OPT_IN = frozenset(str(t) for t in OPT_IN_BEHAVIOR_TAGS)
# The catalog must carry every restricting tag. An opt-in tag may be missing
# (code ahead of the seed, or a rollback window) without poisoning the snapshot.
_REQUIRED = _BEHAVIOR - _OPT_IN
# Qualified only: an unqualified column inside the correlated subqueries would
# bind to the subquery's own table (bot_profiles.slug) and admit every row.
_COLUMN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")
_PREFIX = re.compile(r"^[a-z][a-z0-9_]*$")


class TagPolicyUnavailable(RuntimeError):
    """The snapshot is missing, malformed or over the bound."""


@dataclass(frozen=True)
class TagSnapshot:
    """Immutable view of every profile's tags; answers fail closed."""

    tags: Mapping[str, frozenset[str]] = field(default_factory=lambda: MappingProxyType({}))
    column_private: frozenset[str] = frozenset()
    catalog: frozenset[str] = _BEHAVIOR
    available: bool = True

    @classmethod
    def unavailable(cls) -> TagSnapshot:
        return cls(available=False)

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping]) -> TagSnapshot:
        """Validate snapshot rows; anything unexpected raises TagPolicyUnavailable.

        Rows without ``tags``/``catalog`` keys (legacy fixtures) mean no tags
        and the behaviour vocabulary as catalog.
        """
        rows = list(rows)
        if len(rows) > MAX_BOTS:
            raise TagPolicyUnavailable("too many bots")
        catalog = _BEHAVIOR
        if rows and "catalog" in rows[0]:
            catalog = cls._slugs(rows[0]["catalog"])
            if not _REQUIRED <= catalog:
                raise TagPolicyUnavailable("behaviour tags missing from the catalog")
        tags: dict[str, frozenset[str]] = {}
        column_private: set[str] = set()
        for row in rows:
            bot, flag = row["slug"], row["include_in_global_search"]
            if not is_canonical_bot_slug(bot) or type(flag) is not bool or bot in tags:
                raise TagPolicyUnavailable("malformed profile row")
            if "catalog" in row and cls._slugs(row["catalog"]) != catalog:
                raise TagPolicyUnavailable("inconsistent catalog")
            bot_tags = cls._slugs(row.get("tags", ()))
            if not bot_tags <= catalog:
                raise TagPolicyUnavailable("unknown tag slug")
            tags[bot] = bot_tags
            if not flag:
                column_private.add(bot)
        return cls(MappingProxyType(tags), frozenset(column_private), catalog)

    @staticmethod
    def _slugs(values) -> frozenset[str]:
        if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple, set, frozenset)):
            raise TagPolicyUnavailable("malformed tag array")
        slugs = frozenset(values)
        if len(slugs) != len(values) or not all(
                isinstance(v, str) and TAG_SLUG.fullmatch(v) for v in slugs):
            raise TagPolicyUnavailable("malformed tag array")
        return slugs

    @property
    def known_bots(self) -> frozenset[str]:
        return frozenset(self.tags)

    @property
    def private_bots(self) -> frozenset[str]:
        """Known bots that are private (tag or legacy column fallback)."""
        if not self.available:
            return self.known_bots
        return frozenset(b for b, t in self.tags.items() if _PRIVATE in t) | self.column_private

    @property
    def searchable_bots(self) -> frozenset[str]:
        if not self.available:
            return frozenset()
        return frozenset(b for b, t in self.tags.items()
                         if _NO_GLOBAL not in t) - self.private_bots

    def has(self, bot: str, tag: str) -> bool:
        """Effective tag check. An unknown tag slug raises, never answers."""
        tag = str(tag)
        if tag in _OPT_IN and (not self.available or tag not in self.catalog):
            # An opt-in grant fails closed by granting nothing.
            return False
        if not self.available:
            # No catalog to validate against: behaviour tags fail closed and a
            # role answer would be a guess.
            if tag in _BEHAVIOR:
                return True
            raise TagPolicyUnavailable("bot tags unavailable")
        if tag not in self.catalog:
            raise ValueError(f"unknown bot tag {tag!r}")
        bot = bot.strip().lower() if isinstance(bot, str) else bot
        stored = self.tags.get(bot)
        if stored is None:
            # Unknown bot: private for every check, nothing else.
            return tag in (_PRIVATE, _NO_GLOBAL)
        private = _PRIVATE in stored or bot in self.column_private
        if tag == _PRIVATE:
            return private
        if tag == _NO_GLOBAL:
            return private or _NO_GLOBAL in stored
        return tag in stored

    def excluded_reason(self, bot: str) -> str | None:
        """Reason a private or inactive bot is excluded from a bot-facing job."""
        if self.has(bot, _PRIVATE):
            return "private bot"
        if self.has(bot, _INACTIVE):
            return "inactive bot"
        return None


def read_snapshot_rows(engine) -> list[dict]:
    """Bounded, read-only snapshot read on its own connection; raises on any failure."""
    with engine.connect() as connection:
        return read_snapshot_rows_on(connection)


def read_snapshot_rows_on(connection) -> list[dict]:
    """The snapshot read on a caller's connection, so a caller never holds two.

    Outside a transaction it runs in its own bounded read-only one. Inside the
    caller's transaction it runs in a savepoint: a failed read raises without
    aborting that transaction, and no ``SET LOCAL`` leaks into it.
    """
    if connection.in_transaction():
        with connection.begin_nested():
            return [dict(row) for row in connection.execute(text(SNAPSHOT_SQL)).mappings()]
    with connection.begin():
        connection.execute(text("SET TRANSACTION READ ONLY"))
        connection.execute(text("SET LOCAL statement_timeout = '2000ms'"))
        connection.execute(text("SET LOCAL lock_timeout = '500ms'"))
        return [dict(row) for row in connection.execute(text(SNAPSHOT_SQL)).mappings()]


class BotTags:
    """Sync facade for embedded callers that cannot raise to a user."""

    def __init__(self, engine=None):
        self._engine = engine

    def _resolve_engine(self):
        if self._engine is not None:
            return self._engine
        from llm_bawt.utils.config import Config
        from llm_bawt.utils.db import get_shared_engine
        return get_shared_engine(Config())

    def snapshot(self, conn=None) -> TagSnapshot:
        """One validated snapshot; ``conn`` reads it on the caller's connection."""
        try:
            if conn is not None:
                return TagSnapshot.from_rows(read_snapshot_rows_on(conn))
            engine = self._resolve_engine()
            if engine is None:
                raise TagPolicyUnavailable("no database engine")
            return TagSnapshot.from_rows(read_snapshot_rows(engine))
        except Exception as exc:
            logger.warning("Bot tag snapshot unavailable; treating every bot as "
                           "private and inactive: %s", type(exc).__name__)
            return TagSnapshot.unavailable()

    def has(self, bot: str, tag: str, conn=None) -> bool:
        return self.snapshot(conn).has(bot, tag)

    def excluded_reason(self, bot: str, conn=None) -> str | None:
        return self.snapshot(conn).excluded_reason(bot)


def sql_allowed(column: str, exclude: Iterable[str], *, prefix: str = "tagx") -> tuple[str, dict]:
    """SQL predicate admitting only bots that carry none of ``exclude``.

    Correlated lookups only, so callers need no profile join. ``column`` must
    be qualified with its table alias (``d.target_bot_id``). Excluding
    ``no_global_search`` also excludes ``private``; excluding ``private`` also
    requires a profile row whose legacy column is true (unknown bot = private,
    BOT_TAGS.md §5.3). A missing table errors, which fails closed.
    """
    if not _COLUMN.fullmatch(column) or not _PREFIX.fullmatch(prefix):
        raise ValueError("unsafe column or prefix for sql_allowed")
    excluded = {str(BehaviorTag(t)) for t in exclude}
    if _NO_GLOBAL in excluded:
        excluded.add(str(_PRIVATE))
    if not excluded:
        raise ValueError("sql_allowed needs at least one excluded tag")
    param = f"{prefix}_tags"
    clauses = [f"""NOT EXISTS (SELECT 1 FROM bot_profile_tags {prefix}_t
            WHERE {prefix}_t.bot_slug = lower({column})
              AND {prefix}_t.tag_slug = ANY(CAST(:{param} AS text[])))"""]
    if _PRIVATE in excluded:
        clauses.append(f"""EXISTS (SELECT 1 FROM bot_profiles {prefix}_p
            WHERE {prefix}_p.slug = lower({column}) AND {prefix}_p.include_in_global_search)""")
    return "(" + " AND ".join(clauses) + ")", {param: sorted(excluded)}
