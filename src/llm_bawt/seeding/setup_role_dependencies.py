"""Portable role dependencies of generated product Setup presets (TASK-2516).

Prompt content stays in the product prompt catalog. This projection never carries
source capabilities and does not read or write any installation state.
"""
from __future__ import annotations

import json
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from ..bot_tags.catalog import STARTER_ROLES
from ..bot_tags.store import RoleFields, TagValidationError
from ..prompt_promotion import validate_promotion_item
from ..prompt_sync import install_terms_in
from ..prompt_synced_defaults import body_path, load_manifest
from .bot_presets import BotPreset, PresetError, nonportable_content


def portable_role_prompt_revisions(root: Path, *, install_terms: Collection[str] = ()) -> dict[str, str]:
    """Hash both the registered public metadata and the exact portable body."""
    revisions = {}
    for key, entry in load_manifest(root).items():
        if not key.startswith("role.") or key == "role.":
            continue
        try:
            body = body_path(key, root).read_bytes().decode("utf-8")
        except (ValueError, FileNotFoundError):
            continue
        item = {**entry, "key": key, "body": body, "scope_type": "global",
                "scope_id": "*", "sync_class": "product"}
        if validate_promotion_item(item):
            continue
        public_text = "\n".join(item[field] for field in ("key", "title", "category", "body"))
        if not install_terms_in(public_text, install_terms):
            canonical = json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            revisions[key] = sha256(canonical.encode("utf-8")).hexdigest()
    return revisions


def portable_role_prompt_keys(root: Path, *, install_terms: Collection[str] = ()) -> set[str]:
    return set(portable_role_prompt_revisions(root, install_terms=install_terms))


@dataclass(frozen=True)
class SetupRole:
    slug: str
    name: str
    description: str
    color: str | None
    prompt_key: str | None


def required_roles(
    bots: Iterable[BotPreset],
    source_roles: Mapping[str, Mapping[str, object]],
    portable_prompt_keys: Collection[str],
    *, install_terms: Collection[str] = (),
) -> tuple[SetupRole, ...]:
    """Return the sorted, closed role set, or hold the candidate on any gap.

    The caller supplies already-eligible bot presets and the keys in the matching
    portable prompt candidate, not all keys found in a tenant registry. Unknown
    source fields (including capability grants) are deliberately not copied.
    """
    slugs = {slug for bot in bots for slug in bot.tags}
    roles = []
    names: set[str] = set()
    keys: set[str] = set()
    for index, slug in enumerate(sorted(slugs), 1):
        role_label = f"Setup role {index}"
        source = source_roles.get(slug)
        if source is None:
            raise PresetError(f"{role_label} is missing from the portable role catalog")
        try:
            role = SetupRole(
                RoleFields.slug(source.get("slug")),
                RoleFields.name(source.get("name")),
                RoleFields.description(source.get("description", "")),
                RoleFields.color(source.get("color")),
                RoleFields.prompt_key(source.get("prompt_key")),
            )
        except TagValidationError as exc:
            raise PresetError(f"{role_label} has invalid portable fields: {exc}") from None
        if role.slug != slug:
            raise PresetError(f"{role_label} does not match its reference")
        fields = {
            "slug": role.slug, "name": role.name, "description": role.description,
            "color": role.color or "", "prompt key": role.prompt_key or "",
        }
        hits = nonportable_content(fields)
        hits.extend(f"{label} contains an installation-specific term" for label, value in fields.items()
                    if install_terms_in(value, install_terms))
        if hits:
            raise PresetError(f"{role_label} is held: {'; '.join(hits)}")
        folded = role.name.casefold()
        if folded in names:
            raise PresetError(f"{role_label} has a duplicate role name")
        names.add(folded)
        if role.prompt_key is not None:
            if role.prompt_key not in portable_prompt_keys:
                raise PresetError(f"{role_label} has no released or exportable role prompt")
            if role.prompt_key in keys:
                raise PresetError(f"{role_label} has a duplicate role prompt key")
            keys.add(role.prompt_key)
        roles.append(role)
    return tuple(roles)


class SetupRoleConflict(PresetError):
    """Local role or prompt is not safe to attach to a generated bot."""


# Only the exact shipped role definition may carry the shipped starter grants.
# The portable catalog still never imports capabilities into a tenant.
_STARTER_GRANTS = {
    SetupRole(slug, name, description, None, f"role.{slug}"): list(grants)
    for slug, name, description, grants in STARTER_ROLES
}


def install_missing_roles(conn, roles: tuple[SetupRole, ...]) -> None:
    """Create privilege-free roles or reuse exact local matches, including stock starters."""
    if not roles:
        return
    from sqlalchemy import inspect, text
    from sqlalchemy.exc import DBAPIError

    from ..bot_policy_events import policy_changed
    from ..prompt_synced_defaults import synced_body

    try:
        # Role names aren't unique in operator data. Serialize the collision
        # check and insert against catalog writers, without waiting on them.
        conn.execute(text("LOCK TABLE bot_tag_catalog IN SHARE ROW EXCLUSIVE MODE NOWAIT"))
    except DBAPIError as exc:
        state = getattr(exc.orig, "sqlstate", None) or getattr(exc.orig, "pgcode", None)
        if state == "55P03":
            raise SetupRoleConflict("The role catalog is busy. Try again.") from None
        raise
    local = list(conn.execute(text("""
        SELECT slug, name, description, kind, prompt_key, capabilities, color, system
        FROM bot_tag_catalog
    """)).mappings())
    prompts_exist = any(role.prompt_key for role in roles) and inspect(conn).has_table("prompt_templates")
    created = False
    for role in roles:
        matches = [row for row in local if (row["slug"] == role.slug
                   or row["name"].casefold() == role.name.casefold()
                   or (role.prompt_key is not None and row["prompt_key"] == role.prompt_key))]
        fields = {"slug": role.slug, "name": role.name, "description": role.description,
                  "color": role.color, "prompt_key": role.prompt_key}
        if matches:
            if (len(matches) != 1 or matches[0]["kind"] != "role" or matches[0]["system"]
                    or matches[0]["capabilities"] not in ([], _STARTER_GRANTS.get(role, []))
                    or any(matches[0][key] != value for key, value in fields.items())):
                raise SetupRoleConflict(f"Required role '{role.name}' conflicts with an existing local tag.")
        if role.prompt_key is not None and prompts_exist:
            local_body = conn.execute(text("""
                SELECT body FROM prompt_templates
                 WHERE key = :key AND scope_type = 'global' AND scope_id = '*' FOR SHARE
            """), {"key": role.prompt_key}).scalar_one_or_none()
            if local_body is not None and local_body != synced_body(role.prompt_key):
                raise SetupRoleConflict(f"Required role '{role.name}' has different local instructions.")
        if not matches:
            conn.execute(text("""
                INSERT INTO bot_tag_catalog (slug, name, description, kind, color, prompt_key)
                VALUES (:slug, :name, :description, 'role', :color, :prompt_key)
            """), fields)
            created = True
    if created:
        policy_changed(conn)
