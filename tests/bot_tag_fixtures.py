"""Test helpers for TASK-1329 bot tags: fixture bots only, never a real slug.

``PRIVATE_MODES`` are the three ways a bot is private (BOT_TAGS.md §10.11): a
``private`` tag, only the legacy column false, or a ``private`` assignment
whose profile row is gone.
"""
from __future__ import annotations

from sqlalchemy import text

from llm_bawt.bot_tags import BehaviorTag
from llm_bawt.bot_tags import catalog

PRIVATE_MODES = ("tag", "column", "profileless")


def ensure_tag_tables(conn) -> None:
    """Give a minimal ``bot_profiles`` rig the legacy column, the catalog and assignments."""
    conn.execute(text(
        "ALTER TABLE bot_profiles ADD COLUMN IF NOT EXISTS"
        " include_in_global_search BOOLEAN NOT NULL DEFAULT TRUE"))
    catalog.ensure_schema(conn)
    catalog.seed(conn)


def tag(conn, slug: str, *tags: str) -> None:
    catalog.assign(conn, slug, tags, "test:TASK-1329")


def make_private(conn, slug: str, mode: str) -> None:
    """Make an existing fixture bot private one of the ``PRIVATE_MODES`` ways."""
    if mode == "tag":
        tag(conn, slug, BehaviorTag.PRIVATE)
    elif mode == "column":
        updated = conn.execute(text(
            "UPDATE bot_profiles SET include_in_global_search=false WHERE slug=:slug"
        ), {"slug": slug}).rowcount
        assert updated == 1, f"fixture bot {slug!r} has no profile row"
    elif mode == "profileless":
        tag(conn, slug, BehaviorTag.PRIVATE)
        conn.execute(text("DELETE FROM bot_profiles WHERE slug=:slug"), {"slug": slug})
    else:
        raise ValueError(mode)


def clear_tags(conn, *slugs: str) -> None:
    conn.execute(text("DELETE FROM bot_profile_tags WHERE bot_slug = ANY(:slugs)"),
                 {"slugs": list(slugs)})


def static_tags(private=(), inactive=()):
    """A ``BotTags`` stand-in for unit tests with no database.

    Every bot is public unless listed; ``unavailable`` set on the returned
    class makes every bot private, like a failed snapshot.
    """
    class StaticTags:
        excluded: dict[str, str] = {
            **{slug: "inactive bot" for slug in inactive},
            **{slug: "private bot" for slug in private},
        }
        unavailable = False
        reads = 0

        def __init__(self, engine=None):
            pass

        def snapshot(self, conn=None):
            type(self).reads += 1
            return self

        def excluded_reason(self, bot):
            if self.unavailable:
                return "private bot"
            return self.excluded.get(bot.strip().lower())

    return StaticTags
