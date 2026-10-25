"""Bot tag vocabulary (TASK-1329): the only place code names a behaviour.

Code checks these tag names, never bot slugs. Behaviour rows are seeded from
here on every bootstrap, so code is the source of truth for their names and
descriptions. Role rows are operator data: code seeds none and never names
one (TASK-1249, guarded by tests/test_no_custom_tag_literals.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class BehaviorTag(StrEnum):
    PRIVATE = "private"
    NO_GLOBAL_SEARCH = "no_global_search"
    INACTIVE = "inactive"
    EXTERNAL = "external"


# Behaviour tags that GRANT something instead of restricting it (TASK-1362).
# Restricting tags fail closed by applying to every bot; an opt-in grant fails
# closed by applying to none. A catalog that has not been seeded with one yet
# means nobody has opted in, not that the whole snapshot is untrustworthy.
OPT_IN_BEHAVIOR_TAGS: frozenset[BehaviorTag] = frozenset({BehaviorTag.EXTERNAL})


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
    TagDefinition(
        BehaviorTag.EXTERNAL, "External",
        "Stands in for agents outside the family. Must also be inactive. Opt-in: "
        "today only the bots_send_message default sender honours it.",
        "behavior",
    ),
)

# What ``include_in_global_search = false`` has meant since TASK-1222.
LEGACY_PRIVATE_TAGS: tuple[BehaviorTag, ...] = (BehaviorTag.PRIVATE, BehaviorTag.NO_GLOBAL_SEARCH)

MIGRATION_ASSIGNER = "migration:TASK-1329"
LEGACY_FLAG_ASSIGNER = "legacy:include_in_global_search"
# A REST tag write whose request names no ``updated_by``.
API_ASSIGNER = "api"

TAG_SLUG = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def is_canonical_bot_slug(value: object) -> bool:
    """The profile store's normalization (stripped, lower-case, bounded)."""
    return (isinstance(value, str) and bool(value) and len(value) <= 128
            and value == value.strip() and "\x00" not in value and value == value.lower())
