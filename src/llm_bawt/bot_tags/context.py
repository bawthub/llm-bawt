"""Data for the ``agents.tag_context`` prompt section (TASK-1249).

The section is a registry template (operator data). This module only fills its
variables with facts:

- ``own_tag_instructions``: the instructions of each role tag this bot carries,
  or the tag's description when it has none;
- ``bot_roster``: every visible agent bot with its tags, endpoint ref and
  provider;
- ``tag_catalog``: each role tag's slug, name, description and capabilities.

Nothing here interprets a tag or capability, and nothing volatile (quota, live
state) is included, so the rendered text stays byte-stable between edits to
the catalog. A tag whose ``prompt_key`` names a missing prompt renders its
description and reports ``missing_instructions:<slug>``; the build never fails
on data.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from sqlalchemy import text

logger = logging.getLogger(__name__)

TEMPLATE_KEY = "agents.tag_context"
NONE = "none"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class RosterBot:
    slug: str
    tags: tuple[str, ...]
    endpoint_ref: str | None
    provider: str | None


@dataclass(frozen=True)
class RoleTag:
    slug: str
    name: str
    description: str
    capabilities: tuple[str, ...]
    prompt_key: str | None


@dataclass(frozen=True)
class TagContext:
    own_tag_instructions: str
    bot_roster: str
    tag_catalog: str
    flags: tuple[str, ...] = ()

    def variables(self) -> dict[str, str]:
        return {
            "own_tag_instructions": self.own_tag_instructions,
            "bot_roster": self.bot_roster,
            "tag_catalog": self.tag_catalog,
        }


def render_tag_context(
    bots: Iterable[RosterBot],
    roles: Iterable[RoleTag],
    instructions: Callable[[str], str | None],
    own_tags: Iterable[str],
) -> TagContext:
    """Pure render, sorted throughout: equal facts always give equal bytes."""
    roles_by_slug = {role.slug: role for role in roles}
    role_slugs = sorted(roles_by_slug)
    flags: list[str] = []
    own_blocks = []
    for slug in sorted(set(own_tags) & set(roles_by_slug)):
        role = roles_by_slug[slug]
        body = instructions(role.prompt_key) if role.prompt_key else None
        if role.prompt_key and not (body and body.strip()):
            flags.append(f"missing_instructions:{slug}")
        shown = body.strip() if body and body.strip() else role.description.strip()
        own_blocks.append(f"### {role.name} (`{slug}`)\n{shown}".rstrip())
    roster = [
        f"- {bot.slug}: tags {', '.join(sorted(bot.tags)) or NONE}; "
        f"endpoint {bot.endpoint_ref or UNKNOWN}; provider {bot.provider or UNKNOWN}"
        for bot in sorted(bots, key=lambda b: b.slug)
    ]
    catalog = []
    for slug in role_slugs:
        role = roles_by_slug[slug]
        line = f"- `{slug}` ({role.name}): capabilities {', '.join(sorted(role.capabilities)) or NONE}"
        if role.description.strip():
            line += f". {role.description.strip()}"
        catalog.append(line)
    return TagContext(
        own_tag_instructions="\n\n".join(own_blocks) or NONE,
        bot_roster="\n".join(roster) or NONE,
        tag_catalog="\n".join(catalog) or NONE,
        flags=tuple(flags),
    )


def load_tag_context(
    engine,
    config,
    bot_id: str,
    instructions: Callable[[str], str | None],
    profiles: Iterable | None = None,
) -> TagContext:
    """Read the facts and render them. Raises on an unreadable tag policy.

    Visibility is the ``bots_list_available`` filter: private and inactive
    bots are excluded. Only agent bots are listed; an unresolvable endpoint is
    reported as unknown rather than dropping the bot.
    """
    from ..bot_types import normalize_bot_type
    from ..runtime_settings import BotProfileStore
    from .facade import TagSnapshot, read_snapshot_rows

    snapshot = TagSnapshot.from_rows(read_snapshot_rows(engine))
    roles = _role_tags(engine)
    if profiles is None:
        profiles = BotProfileStore(config).list_all()
    model_catalog = _model_catalog(config)
    bots = [
        RosterBot(
            slug=row.slug,
            tags=tuple(sorted(snapshot.tags.get(row.slug, ()))),
            **_endpoint_facts(model_catalog, row.endpoint_id),
        )
        for row in profiles
        if normalize_bot_type(row.bot_type, row.agent_backend) == "agent"
        and row.slug in snapshot.known_bots
        and snapshot.excluded_reason(row.slug) is None
    ]
    return render_tag_context(bots, roles, instructions, snapshot.tags.get(bot_id, ()))


def _role_tags(engine) -> list[RoleTag]:
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT slug, name, description, capabilities, prompt_key
            FROM bot_tag_catalog WHERE kind = 'role' ORDER BY slug
        """)).mappings()
        return [RoleTag(row["slug"], row["name"], row["description"] or "",
                        tuple(row["capabilities"] or ()), row["prompt_key"]) for row in rows]


def _model_catalog(config):
    try:
        return config.ensure_model_catalog()
    except Exception:
        logger.warning("tag context: model catalog unavailable", exc_info=True)
        return None


def _endpoint_facts(model_catalog, endpoint_id: int | None) -> Mapping[str, str | None]:
    if model_catalog is None or endpoint_id is None:
        return {"endpoint_ref": None, "provider": None}
    try:
        endpoint = model_catalog.resolve_endpoint(int(endpoint_id))
    except Exception:
        return {"endpoint_ref": None, "provider": None}
    return {"endpoint_ref": endpoint.ref, "provider": endpoint.model.vendor}
