"""Read the live bot lineup for the generated Setup catalog (TASK-2516).

One read-only, repeatable-read query returns every eligible bot's travelling
fields and the role definitions those bots use. Eligible means visible in
global search, not carrying a private tag and not explicitly excluded by the
operator. Private profiles are filtered in SQL, never read and dropped later.
Only allowlisted columns are selected: no endpoint, model, house, session,
memory, history, credential or capability data enters the result.
"""
from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from sqlalchemy import text

from .bot_presets import PresetError
from .preset_source import GLOBAL_SEARCH_PROFILE, eligibility_params, no_private_assignment, read_preset_source

_SLUG = re.compile(r"[a-z0-9][a-z0-9_-]{0,127}")
READ_FAILURE = "Setup source returned an incomplete or ambiguous roster"


def is_slug(value: object) -> bool:
    return isinstance(value, str) and _SLUG.fullmatch(value) is not None


@dataclass(frozen=True)
class LiveSetupSource:
    """Eligible profiles (with role tag slugs) and the role rows they reference."""

    profiles: Sequence[Mapping[str, object]]
    roles: Mapping[str, Mapping[str, object]]


_SQL = f"""
WITH scope AS (SELECT CAST(:scope AS jsonb) AS rules),
eligible AS MATERIALIZED (
    SELECT lower(btrim(p.slug)) AS slug, p.name, p.description, p.system_prompt, p.avatar, p.color
      FROM bot_profiles p
     WHERE {GLOBAL_SEARCH_PROFILE}
       AND NOT ((SELECT rules->'excluded' FROM scope) ? lower(btrim(p.slug)))
       AND {no_private_assignment('lower(btrim(p.slug))', "(SELECT rules->'private' FROM scope) ? t.tag_slug")}
), memberships AS MATERIALIZED (
    SELECT e.slug AS bot, c.slug AS role
      FROM eligible e
      JOIN bot_profile_tags t ON t.bot_slug = e.slug
      JOIN bot_tag_catalog c ON c.slug = t.tag_slug AND c.kind = 'role'
)
SELECT (CASE WHEN current_setting('transaction_read_only') <> 'on'
               OR current_setting('transaction_isolation') <> 'repeatable read'
  THEN jsonb_build_object('status', 'incomplete')
  ELSE jsonb_build_object('status', 'complete',
    'profiles', COALESCE((SELECT jsonb_agg(jsonb_build_object(
        'slug', e.slug, 'name', e.name, 'description', e.description,
        'system_prompt', e.system_prompt, 'avatar', e.avatar, 'color', e.color,
        'roles', ARRAY(SELECT DISTINCT m.role FROM memberships m WHERE m.bot = e.slug ORDER BY 1))
      ORDER BY e.slug) FROM eligible e), '[]'::jsonb),
    'roles', COALESCE((SELECT jsonb_agg(jsonb_build_object(
        'slug', c.slug, 'name', c.name, 'description', c.description,
        'color', c.color, 'prompt_key', c.prompt_key) ORDER BY c.slug)
      FROM bot_tag_catalog c WHERE c.kind = 'role'
       AND c.slug IN (SELECT role FROM memberships)), '[]'::jsonb)
  ) END)::text
"""
_SETUP_SQL = ("SET LOCAL TIME ZONE 'UTC'", "SET LOCAL statement_timeout = '10s'")
_PGQ_COMMAND = ("pgq", "llm-bawt", "-X", "-qAtw", "-v", "ON_ERROR_STOP=1", "-f", "-")


def _scope(excluded: Collection[str]) -> str:
    if isinstance(excluded, str) or not all(isinstance(slug, str) for slug in excluded):
        raise PresetError("Setup source scope is invalid")
    values = {slug.strip().lower() for slug in excluded}
    if not all(is_slug(slug) for slug in values):
        raise PresetError("Setup source scope is invalid")
    return json.dumps({**eligibility_params(), "excluded": sorted(values)})


def _optional_text(value: object) -> bool:
    return value is None or isinstance(value, str)


def _parse(payload: str) -> LiveSetupSource:
    try:
        value = json.loads(payload)
        if (not isinstance(value, dict) or set(value) != {"status", "profiles", "roles"}
                or value["status"] != "complete" or not isinstance(value["profiles"], list)
                or not isinstance(value["roles"], list)):
            raise ValueError
        roles: dict[str, dict] = {}
        for row in value["roles"]:
            if (not isinstance(row, dict) or not is_slug(row.get("slug")) or row["slug"] in roles
                    or not all(_optional_text(row.get(k)) for k in ("name", "description", "color", "prompt_key"))):
                raise ValueError
            roles[row["slug"]] = row
        profiles, seen = [], set()
        for row in value["profiles"]:
            if (not isinstance(row, dict) or not is_slug(row.get("slug")) or row["slug"] in seen
                    or not all(_optional_text(row.get(k))
                               for k in ("name", "description", "system_prompt", "avatar", "color"))
                    or not isinstance(row.get("roles"), list)
                    or not all(isinstance(r, str) and r in roles for r in row["roles"])):
                raise ValueError
            seen.add(row["slug"])
            profiles.append(row)
        if {r for row in profiles for r in row["roles"]} != roles.keys():
            raise ValueError
    except (TypeError, ValueError, RecursionError):
        raise PresetError(READ_FAILURE) from None
    return LiveSetupSource(profiles, roles)


def read_live_setup_source(config, *, excluded: Collection[str]) -> LiveSetupSource:
    scope = _scope(excluded)
    try:
        with read_preset_source(config) as conn:
            for command in _SETUP_SQL:
                conn.exec_driver_sql(command)
            payload = conn.execute(text(_SQL), {"scope": scope}).scalar_one()
    except Exception:
        raise PresetError(READ_FAILURE) from None
    return _parse(payload)


def read_live_setup_source_pgq(*, excluded: Collection[str]) -> LiveSetupSource:
    """One-shot operator reader; the scope travels over stdin, never argv."""
    scope = _scope(excluded)
    # The scope holds only strict ASCII slugs; psql quotes it as one literal.
    script = (f"\\set scope '{scope}'\nBEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;\n"
              + ";\n".join(_SETUP_SQL) + ";\n" + _SQL.replace(":scope", ":'scope'") + ";\nCOMMIT;\n")
    try:
        result = subprocess.run(_PGQ_COMMAND, input=script, capture_output=True, text=True, timeout=20, check=True)
    except (OSError, subprocess.SubprocessError, UnicodeError):
        raise PresetError(READ_FAILURE) from None
    return _parse(result.stdout)
