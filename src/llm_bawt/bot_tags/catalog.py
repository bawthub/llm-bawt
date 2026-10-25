"""Bot tag schema, seed, one-shot legacy migration and assignment writes.

Every function takes an open connection and runs inside the caller's
transaction; nothing here commits. Existence checks are pinned to
``current_schema()`` through ``pg_class``, never search_path resolution
(``to_regclass`` would see ``public.x`` from a ``tenant,public`` path).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

from sqlalchemy import text

from .errors import TagSetChanged, TagValidationError
from .vocabulary import (
    API_ASSIGNER, BEHAVIOR_TAGS, LEGACY_FLAG_ASSIGNER, LEGACY_PRIVATE_TAGS, MIGRATION_ASSIGNER,
    TAG_SLUG, BehaviorTag,
)

CATALOG_TABLE = "bot_tag_catalog"
ASSIGNMENT_TABLE = "bot_profile_tags"
ASSIGNMENT_INDEX = "bot_profile_tags_tag_idx"

_OWNED = text("""
    SELECT EXISTS (
      SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
      WHERE n.nspname = current_schema() AND c.relname = :rel
        AND c.relkind = ANY(CAST(:kinds AS "char"[])))
""")

_CATALOG_DDL = """
CREATE TABLE IF NOT EXISTS bot_tag_catalog (
  slug         VARCHAR(64)  PRIMARY KEY CHECK (slug ~ '^[a-z][a-z0-9_]{0,63}$'),
  name         VARCHAR(128) NOT NULL,
  description  TEXT         NOT NULL DEFAULT '',
  kind         VARCHAR(16)  NOT NULL CHECK (kind IN ('behavior','role')),
  prompt_key   VARCHAR(160) NULL
               CHECK (prompt_key IS NULL OR (kind = 'role' AND prompt_key LIKE 'role.%')),
  capabilities JSONB        NOT NULL DEFAULT '[]'::jsonb
               CHECK (jsonb_typeof(capabilities) = 'array'),
  system       BOOLEAN      NOT NULL DEFAULT FALSE,
  created_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
  updated_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
  CHECK (NOT system OR kind = 'behavior')
)
"""

# Deliberately no FK to bot_profiles: assignments outlive a deleted profile, and
# a surviving ``private`` row is the storage-protection list (BOT_TAGS.md §6).
_ASSIGNMENT_DDL = """
CREATE TABLE IF NOT EXISTS bot_profile_tags (
  bot_slug    VARCHAR(128) NOT NULL,
  tag_slug    VARCHAR(64)  NOT NULL
              REFERENCES bot_tag_catalog(slug) ON DELETE RESTRICT ON UPDATE RESTRICT,
  assigned_at TIMESTAMPTZ  NOT NULL DEFAULT now(),
  assigned_by VARCHAR(128) NULL,
  PRIMARY KEY (bot_slug, tag_slug),
  CHECK (bot_slug = lower(btrim(bot_slug)) AND bot_slug <> '')
)
"""

_INDEX_DDL = "CREATE INDEX IF NOT EXISTS bot_profile_tags_tag_idx ON bot_profile_tags (tag_slug)"


def owned(conn, rel: str, kinds: str = "rp") -> bool:
    """Whether ``current_schema()`` itself holds relation ``rel``."""
    return bool(conn.execute(_OWNED, {"rel": rel, "kinds": list(kinds)}).scalar_one())


def require_owned(conn, rel: str) -> None:
    if not owned(conn, rel):
        raise RuntimeError(
            f"{rel} is not owned by the current schema; refusing to bootstrap bot tags "
            "against a table from another search_path entry")


def ensure_schema(conn) -> None:
    """Create the catalog, assignments and index only when absent.

    Even a no-op ``CREATE INDEX IF NOT EXISTS`` takes a SHARE lock on its table,
    so steady-state starts check the catalog first and issue no DDL at all.
    """
    if not owned(conn, CATALOG_TABLE):
        conn.execute(text(_CATALOG_DDL))
    if not owned(conn, ASSIGNMENT_TABLE):
        conn.execute(text(_ASSIGNMENT_DDL))
    if not owned(conn, ASSIGNMENT_INDEX, "i"):
        conn.execute(text(_INDEX_DDL))


def seed(conn) -> None:
    """Upsert behaviour rows from code. Roles are operator data: none are
    seeded and existing role rows are never touched. No assignments."""
    for tag in BEHAVIOR_TAGS:
        conn.execute(text("""
            INSERT INTO bot_tag_catalog (slug, name, description, kind, system)
            VALUES (:slug, :name, :description, 'behavior', TRUE)
            ON CONFLICT (slug) DO UPDATE
               SET name = EXCLUDED.name, description = EXCLUDED.description,
                   kind = 'behavior', system = TRUE, prompt_key = NULL,
                   capabilities = '[]'::jsonb, updated_at = now()
             WHERE (bot_tag_catalog.name, bot_tag_catalog.description, bot_tag_catalog.kind,
                    bot_tag_catalog.system, bot_tag_catalog.prompt_key,
                    bot_tag_catalog.capabilities)
                   IS DISTINCT FROM
                   (EXCLUDED.name, EXCLUDED.description, 'behavior', TRUE, NULL::varchar,
                    '[]'::jsonb)
        """), {"slug": str(tag.slug), "name": tag.name, "description": tag.description})


def migrate_legacy_flag(conn) -> int:
    """One-shot copy of ``include_in_global_search = false`` into tags.

    The caller runs this only in the transaction that created
    ``bot_profile_tags``, after proving the current schema owns ``bot_profiles``.
    """
    result = conn.execute(text("""
        INSERT INTO bot_profile_tags (bot_slug, tag_slug, assigned_by)
        SELECT lower(btrim(p.slug)), t.tag, :by
        FROM bot_profiles p CROSS JOIN unnest(CAST(:tags AS text[])) AS t(tag)
        WHERE p.include_in_global_search = false AND btrim(p.slug) <> ''
        ON CONFLICT DO NOTHING
    """), {"by": MIGRATION_ASSIGNER, "tags": [str(t) for t in LEGACY_PRIVATE_TAGS]})
    return int(result.rowcount or 0)


def assign(conn, bot_slug: str, tags: Iterable[str], by: str | None) -> None:
    for tag in tags:
        conn.execute(text("""
            INSERT INTO bot_profile_tags (bot_slug, tag_slug, assigned_by)
            VALUES (:bot, :tag, :by) ON CONFLICT DO NOTHING
        """), {"bot": bot_slug, "tag": str(tag), "by": by})


def unassign(conn, bot_slug: str, tags: Iterable[str]) -> None:
    conn.execute(text("""
        DELETE FROM bot_profile_tags
        WHERE bot_slug = :bot AND tag_slug = ANY(CAST(:tags AS text[]))
    """), {"bot": bot_slug, "tags": [str(t) for t in tags]})


def has_assignment(conn, bot_slug: str, tag: BehaviorTag) -> bool:
    return bool(conn.execute(text("""
        SELECT EXISTS (SELECT 1 FROM bot_profile_tags WHERE bot_slug = :bot AND tag_slug = :tag)
    """), {"bot": bot_slug, "tag": str(tag)}).scalar_one())


def legacy_flag_write(conn, bot_slug: str, *, previous: bool | None, requested: bool,
                      by: str | None) -> bool:
    """Translate a legacy ``include_in_global_search`` write; return the column value.

    The column stays a mirror of ``not private`` (BOT_TAGS.md §4.1). ``previous``
    is the stored column, or None for a create. Runs in the profile write's own
    transaction, so tags and column commit together.
    """
    if previous is not None and requested == previous:
        # An unchanged value (a whole-profile re-save) never touches tags.
        return previous
    if requested:
        if previous is not None:
            unassign(conn, bot_slug, LEGACY_PRIVATE_TAGS)
    else:
        assign(conn, bot_slug, LEGACY_PRIVATE_TAGS, by)
    # A re-created slug keeps inherited tags; a private identity stays private.
    return not has_assignment(conn, bot_slug, BehaviorTag.PRIVATE)


def lock_bot(conn, bot_slug: str) -> None:
    """Serialize every tag write for one bot until the caller's transaction ends.

    Without it two concurrent full-set replaces each miss the other's
    uncommitted inserts and commit the union, a set neither asked for.
    """
    conn.execute(text("SELECT pg_advisory_xact_lock(hashtext('bot-tags'), hashtext(:bot))"),
                 {"bot": bot_slug})


def validate_assignment_set(conn, tags: object) -> frozenset[str]:
    """Exact catalog slugs, deduplicated; anything else raises TagValidationError.

    No normalization: ``" Private"`` is refused, never quietly stored as
    ``private``. The catalog rows are KEY SHARE locked, so a concurrent catalog
    delete waits for this transaction and then meets the FK.
    """
    if isinstance(tags, (str, bytes)) or not isinstance(tags, (list, tuple, set, frozenset)):
        raise TagValidationError("tags must be a list of tag slugs")
    bad = sorted(repr(t) for t in tags if not isinstance(t, str) or not TAG_SLUG.fullmatch(t))
    if bad:
        raise TagValidationError(f"malformed tag slug(s): {', '.join(bad)}")
    wanted = frozenset(tags)
    known = set(conn.execute(text("""
        SELECT slug FROM bot_tag_catalog WHERE slug = ANY(CAST(:tags AS text[]))
        ORDER BY slug FOR KEY SHARE
    """), {"tags": sorted(wanted)}).scalars())
    unknown = sorted(wanted - known)
    if unknown:
        raise TagValidationError(f"unknown bot tag(s): {', '.join(unknown)}")
    return wanted


def replace_assignments(conn, bot_slug: str, tags: object, by: str | None) -> bool:
    """Full-set replace of one bot's tags; return the legacy column value.

    Kept assignments keep their ``assigned_at``/``assigned_by``. The column is
    written as ``not private`` from the new set (BOT_TAGS.md §4.1). The caller
    holds :func:`lock_bot`.
    """
    wanted = validate_assignment_set(conn, tags)
    conn.execute(text("""
        DELETE FROM bot_profile_tags
        WHERE bot_slug = :bot AND NOT (tag_slug = ANY(CAST(:tags AS text[])))
    """), {"bot": bot_slug, "tags": sorted(wanted)})
    assign(conn, bot_slug, sorted(wanted), by)
    return str(BehaviorTag.PRIVATE) not in wanted


def check_expected_assignments(conn, bot_slug: str, expected: object) -> None:
    """Compare-and-set guard for a full-set replace (TASK-1398).

    ``expected`` is the set the caller read before computing its replace. It is
    compared with the stored set under the caller's :func:`lock_bot`, so two
    writers that read the same set can't both commit; the second gets
    :class:`TagSetChanged` carrying the current set, and its transaction rolls back.
    """
    if isinstance(expected, (str, bytes)) or not isinstance(expected, (list, tuple, set, frozenset)):
        raise TagValidationError("expected_tags must be a list of tag slugs")
    if any(not isinstance(t, str) for t in expected):
        raise TagValidationError("expected_tags must be a list of tag slugs")
    current = sorted(conn.execute(text(
        "SELECT tag_slug FROM bot_profile_tags WHERE bot_slug = :bot"
    ), {"bot": bot_slug}).scalars())
    if frozenset(expected) != frozenset(current):
        raise TagSetChanged(bot_slug, current)


def profile_tag_write(conn, bot_slug: str, *, previous: bool | None,
                      payload: Mapping[str, object]) -> bool:
    """The one tag write a profile upsert makes; return the legacy column value.

    A payload carrying ``tags`` is a full-set replace by ``tags_assigned_by``;
    otherwise the legacy ``include_in_global_search`` flag is translated
    (BOT_TAGS.md §4.1). ``previous`` is the stored column, or None for a create.
    """
    lock_bot(conn, bot_slug)
    if "expected_tags" in payload:
        check_expected_assignments(conn, bot_slug, payload["expected_tags"])
    if "tags" in payload:
        by = payload.get("tags_assigned_by") or API_ASSIGNER
        return replace_assignments(conn, bot_slug, payload["tags"], str(by))
    default = True if previous is None else previous
    return legacy_flag_write(
        conn, bot_slug, previous=previous,
        requested=bool(payload.get("include_in_global_search", default)), by=LEGACY_FLAG_ASSIGNER)
