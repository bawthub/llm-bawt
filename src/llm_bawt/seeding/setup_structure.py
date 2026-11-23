"""Closed structural projection for Setup, shared by the app and pgq readers."""
from __future__ import annotations

import json
import subprocess
from collections.abc import Collection, Mapping
from dataclasses import dataclass

from sqlalchemy import text

from ..bot_tags.vocabulary import BEHAVIOR_TAGS
from .bot_presets import PresetError
from .preset_source import GLOBAL_SEARCH_PROFILE, eligibility_params, no_private_assignment, read_preset_source
from .setup_portable_input import is_digest, is_slug


@dataclass(frozen=True)
class SetupSourceStructure:
    profiles: Mapping[str, str]
    roles: Mapping[str, str]
    memberships: Mapping[str, Collection[str]]


ROLE_HOLD = "Setup source has an unavailable required role"
READ_FAILURE = "Setup source returned an incomplete or ambiguous roster"


def _timestamp(value: str) -> str:
    # Match Python's ISO format including optional six-place fraction and,
    # only for zoned columns, the UTC offset. Both readers set the zone locally.
    return f"""regexp_replace(to_char({value}, 'YYYY-MM-DD"T"HH24:MI:SS.US'), '[.]000000$', '')
        || CASE WHEN pg_typeof({value}) = 'timestamptz'::regtype THEN '+00:00' ELSE '' END"""


def _revision(kind: str, slug: str, identity: str, alias: str) -> str:
    # jsonb's array encoding matches the previous Python JSON encoding for
    # canonical ASCII slugs, integer or null identity, and ISO timestamps.
    return f"""CASE WHEN isfinite({alias}.created_at) AND isfinite({alias}.updated_at) THEN
        encode(sha256(convert_to(jsonb_build_array('{kind}', {slug}, {identity},
            {_timestamp(f'{alias}.created_at')}, {_timestamp(f'{alias}.updated_at')})::text, 'UTF8')), 'hex') END"""


# Structurally unknown assignments hold, including nonstandard behavior tags.
# Never query the catalog to classify an unknown or denied assignment. First
# materialize eligible membership and approved, non-denied role requests, then
# resolve only those requests. No role data is returned at all on a role hold.
_SQL = f"""
WITH policy AS (SELECT CAST(:scope AS jsonb) AS rules),
in_scope AS MATERIALIZED (
    SELECT lower(btrim(p.slug)) AS slug, p.id, p.created_at, p.updated_at
      FROM bot_profiles p
     WHERE NOT ((SELECT rules->'excluded' FROM policy) ? lower(btrim(p.slug)))
       AND {GLOBAL_SEARCH_PROFILE}
), eligible AS MATERIALIZED (
    SELECT p.slug, {_revision('profile', 'p.slug', 'p.id', 'p')} AS revision
      FROM in_scope p
     WHERE {no_private_assignment('p.slug', "(SELECT rules->'private' FROM policy) ? t.tag_slug")}
), assignments AS MATERIALIZED (
    SELECT e.slug AS bot, t.tag_slug AS tag
      FROM eligible e JOIN bot_profile_tags t ON t.bot_slug = e.slug
), role_requests AS MATERIALIZED (
    SELECT DISTINCT a.tag FROM assignments a CROSS JOIN policy
     WHERE rules->'allowed' ? a.tag AND NOT (rules->'denied' ? a.tag) AND NOT (rules->'behavior' ? a.tag)
), roles AS MATERIALIZED (
    SELECT r.tag AS slug, {_revision('role', 'r.tag', 'NULL::integer', 'c')} AS revision
      FROM role_requests r CROSS JOIN LATERAL (
          SELECT created_at, updated_at FROM bot_tag_catalog WHERE slug = r.tag AND kind = 'role' OFFSET 0
      ) c
), behavior_requests AS MATERIALIZED (
    SELECT DISTINCT a.tag FROM assignments a CROSS JOIN policy
     WHERE rules->'behavior' ? a.tag AND NOT (rules->'denied' ? a.tag)
), behavior AS MATERIALIZED (
    SELECT b.tag FROM behavior_requests b JOIN bot_tag_catalog c ON c.slug = b.tag AND c.kind = 'behavior'
)
SELECT (CASE WHEN current_setting('transaction_read_only') <> 'on'
               OR current_setting('transaction_isolation') <> 'repeatable read'
  THEN jsonb_build_object('status', 'incomplete')
  WHEN EXISTS (
    SELECT 1 FROM assignments a
     WHERE NOT EXISTS (SELECT 1 FROM behavior b WHERE b.tag = a.tag)
       AND NOT EXISTS (SELECT 1 FROM roles r WHERE r.slug = a.tag AND r.revision IS NOT NULL)
) THEN jsonb_build_object('status', 'role_hold')
  WHEN EXISTS (SELECT 1 FROM eligible WHERE revision IS NULL OR slug IS NULL
                 OR slug !~ '^[a-z0-9][a-z0-9_-]{{0,127}}$')
       OR EXISTS (SELECT 1 FROM eligible GROUP BY slug HAVING count(*) <> 1)
  THEN jsonb_build_object('status', 'incomplete')
  ELSE jsonb_build_object('status', 'complete',
    'profiles', COALESCE((SELECT jsonb_agg(jsonb_build_array(e.slug, e.revision,
        ARRAY(SELECT r.slug FROM assignments a JOIN roles r ON a.tag = r.slug
              WHERE a.bot = e.slug ORDER BY r.slug)) ORDER BY e.slug) FROM eligible e), '[]'::jsonb),
    'roles', COALESCE((SELECT jsonb_agg(jsonb_build_array(slug, revision) ORDER BY slug) FROM roles), '[]'::jsonb)
  ) END)::text
"""
_SETUP_SQL = ("SET LOCAL TIME ZONE 'UTC'", "SET LOCAL statement_timeout = '10s'")
_PGQ_COMMAND = ("pgq", "llm-bawt", "-X", "-qAtw", "-v", "ON_ERROR_STOP=1", "-f", "-")


def _scope(*, excluded: Collection[str], allowed_roles: Collection[str], denied_roles: Collection[str]) -> str:
    def normalized(slugs: Collection[str]) -> list[str]:
        if isinstance(slugs, str) or not all(isinstance(slug, str) for slug in slugs):
            raise PresetError("Setup source scope is invalid")
        values = {slug.strip().lower() for slug in slugs}
        if not all(is_slug(slug) for slug in values):
            raise PresetError("Setup source scope is invalid")
        return sorted(values)

    return json.dumps({**eligibility_params(), "excluded": normalized(excluded),
                       "allowed": normalized(allowed_roles), "denied": normalized(denied_roles),
                       "behavior": sorted(str(tag.slug) for tag in BEHAVIOR_TAGS)})


def _parse_structure(payload: str) -> SetupSourceStructure:
    profiles: dict[str, str] = {}
    roles: dict[str, str] = {}
    memberships: dict[str, Collection[str]] = {}
    try:
        value = json.loads(payload)
    except (TypeError, ValueError, RecursionError):
        raise PresetError(READ_FAILURE) from None
    if value == {"status": "role_hold"}:
        raise PresetError(ROLE_HOLD)
    try:
        if (not isinstance(value, dict) or set(value) != {"status", "profiles", "roles"}
                or value["status"] != "complete" or not isinstance(value["profiles"], list)
                or not isinstance(value["roles"], list)):
            raise ValueError
        for row in value["roles"]:
            if (not isinstance(row, list) or len(row) != 2 or not is_slug(row[0])
                    or not is_digest(row[1]) or row[0] in roles):
                raise ValueError
            roles[row[0]] = row[1]
        required = set()
        for row in value["profiles"]:
            if (not isinstance(row, list) or len(row) != 3 or not is_slug(row[0])
                    or not is_digest(row[1]) or row[0] in profiles or not isinstance(row[2], list)
                    or not all(is_slug(tag) for tag in row[2]) or len(set(row[2])) != len(row[2])):
                raise ValueError
            profiles[row[0]], memberships[row[0]] = row[1], frozenset(row[2])
            required.update(row[2])
        if required != roles.keys():
            raise ValueError
    except (TypeError, ValueError, RecursionError):
        raise PresetError(READ_FAILURE) from None
    return SetupSourceStructure(profiles, roles, memberships)


def read_setup_structure(config, *, excluded: Collection[str], allowed_roles: Collection[str],
                         denied_roles: Collection[str]) -> SetupSourceStructure:
    scope = _scope(excluded=excluded, allowed_roles=allowed_roles, denied_roles=denied_roles)
    try:
        with read_preset_source(config) as conn:
            for command in _SETUP_SQL:
                conn.exec_driver_sql(command)
            payload = conn.execute(text(_SQL), {"scope": scope}).scalar_one()
    except Exception:
        raise PresetError(READ_FAILURE) from None
    return _parse_structure(payload)


def read_setup_structure_pgq(*, excluded: Collection[str], allowed_roles: Collection[str],
                             denied_roles: Collection[str]) -> SetupSourceStructure:
    """One-shot operator reader; scope is private over stdin, never argv."""
    scope = _scope(excluded=excluded, allowed_roles=allowed_roles, denied_roles=denied_roles)
    # JSON values come only from strict ASCII slugs, never psql syntax. The
    # psql interpolation quotes the entire value as a SQL literal.
    script = (f"\\set scope '{scope}'\nBEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;\n"
              + ";\n".join(_SETUP_SQL) + ";\n" + _SQL.replace(":scope", ":'scope'") + ";\nCOMMIT;\n")
    try:
        result = subprocess.run(_PGQ_COMMAND, input=script, capture_output=True, text=True, timeout=20, check=True)
    except (OSError, subprocess.SubprocessError, UnicodeError):
        raise PresetError(READ_FAILURE) from None
    return _parse_structure(result.stdout)
