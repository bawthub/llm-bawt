"""Bot tag vocabulary (TASK-1329): the only place code names a behaviour.

Code checks these tag names, never bot slugs. Behaviour rows are seeded from
here on every bootstrap, so code is the source of truth for their names and
descriptions; role rows are operator data and only the first two are seeded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class BehaviorTag(StrEnum):
    PRIVATE = "private"
    NO_GLOBAL_SEARCH = "no_global_search"
    INACTIVE = "inactive"


@dataclass(frozen=True)
class TagDefinition:
    slug: str
    name: str
    description: str
    kind: str


BEHAVIOR_TAGS: tuple[TagDefinition, ...] = (
    TagDefinition(
        BehaviorTag.PRIVATE, "Private",
        "Invisible on every bot-facing surface; other bots cannot read its content. "
        "Excluded from idle jobs, initiative, working memory and integrations. "
        "Implies no_global_search.",
        "behavior",
    ),
    TagDefinition(
        BehaviorTag.NO_GLOBAL_SEARCH, "No global search",
        "Excluded from aggregate search and cross-bot recall, including explicit "
        "aggregate filters by other bots. Still listed and readable by targeted tools.",
        "behavior",
    ),
    TagDefinition(
        BehaviorTag.INACTIVE, "Inactive",
        "Hidden from bot lists, discovery and routing; excluded from dispatch, idle "
        "jobs and initiative. New bot messages to it are refused. Data stays intact.",
        "behavior",
    ),
)

SEEDED_ROLES: tuple[TagDefinition, ...] = (
    TagDefinition("coder", "Coder", "Takes implementation work.", "role"),
    TagDefinition("reviewer", "Reviewer", "Reviews other bots' work.", "role"),
)

# What ``include_in_global_search = false`` has meant since TASK-1222.
LEGACY_PRIVATE_TAGS: tuple[BehaviorTag, ...] = (BehaviorTag.PRIVATE, BehaviorTag.NO_GLOBAL_SEARCH)

MIGRATION_ASSIGNER = "migration:TASK-1329"
LEGACY_FLAG_ASSIGNER = "legacy:include_in_global_search"

TAG_SLUG = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def is_canonical_bot_slug(value: object) -> bool:
    """The profile store's normalization (stripped, lower-case, bounded)."""
    return (isinstance(value, str) and bool(value) and len(value) <= 128
            and value == value.strip() and "\x00" not in value and value == value.lower())
