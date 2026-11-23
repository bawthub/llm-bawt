"""Read-only scope shared by manual preset export and product seed sync."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import text
from sqlalchemy.engine import Connection

from ..bot_tags.vocabulary import LEGACY_PRIVATE_TAGS
from .bot_presets import PresetError

# Never read a private/protected profile and filter its contents in Python.
def no_private_assignment(slug_sql: str, private_sql: str) -> str:
    # Both arguments are static SQL fragments, never operator input.
    return f"""NOT EXISTS (SELECT 1 FROM bot_profile_tags t
                      WHERE t.bot_slug = {slug_sql} AND ({private_sql}))"""


GLOBAL_SEARCH_PROFILE = "p.include_in_global_search"
EXPORTABLE_PROFILES = f"""
    FROM bot_profiles p
   WHERE {GLOBAL_SEARCH_PROFILE}
     AND {no_private_assignment('lower(btrim(p.slug))', 't.tag_slug = ANY(:private)')}
"""

PORTABLE_PROFILE_FIELDS = (
    "SELECT lower(btrim(p.slug)) AS slug, p.name, p.description, p.system_prompt, p.avatar, p.color"
)

ELIGIBLE_SLUGS = text("SELECT lower(btrim(p.slug)) AS slug" + EXPORTABLE_PROFILES + " ORDER BY 1")
ELIGIBLE_PROFILES = text(PORTABLE_PROFILE_FIELDS + EXPORTABLE_PROFILES + " ORDER BY 1")
SELECTED_PROFILES = text(
    PORTABLE_PROFILE_FIELDS + EXPORTABLE_PROFILES + " AND lower(btrim(p.slug)) = ANY(:slugs) ORDER BY 1"
)
ROLE_ASSIGNMENTS = text("""
    SELECT t.bot_slug, c.slug, c.name, c.description, c.color, c.prompt_key
      FROM bot_profile_tags t
      JOIN bot_tag_catalog c ON c.slug = t.tag_slug AND c.kind = 'role'
     WHERE t.bot_slug = ANY(:slugs)
     ORDER BY t.bot_slug, c.slug
""")


def eligibility_params() -> dict[str, list[str]]:
    return {"private": sorted(str(tag) for tag in LEGACY_PRIVATE_TAGS)}


@contextmanager
def read_preset_source(config) -> Iterator[Connection]:
    """One repeatable, read-only transaction; no store or schema bootstrap."""
    from ..utils.db import build_engine_from_config

    engine = build_engine_from_config(config, application_name="llm-bawt-preset-export",
                                      pool_size=1, max_overflow=0)
    if engine is None:
        raise PresetError("the bot database is not configured")
    try:
        with engine.connect().execution_options(isolation_level="REPEATABLE READ",
                                                postgresql_readonly=True) as conn:
            with conn.begin():
                yield conn
    finally:
        engine.dispose()
