"""Grant tags (TASK-1609): opt-in behaviour tags that authorize an action.

A grant tag (``GRANT_TAGS``) is written only by :func:`write_grant`, which
records who changed it and why. Every other tag writer refuses to add or remove
one: the profile tag write (:func:`check_grant_membership`) and the
single-assignment delete (:func:`refuse_grant_tag`), so a bot can't ordain
itself or a peer through ``bots_profile_patch``.

Who may write a grant is decided here too (:class:`GrantActor`):

- a person, signed in by BawtHub (the route checks the service bearer);
- a bot holding a role tag whose capabilities include the global runtime
  setting ``ops_grant_manager_capability``. Unset means persons only; the
  capability name is install data, never a code literal. A bot can't change its
  own grants, and private bots are neither managers nor targets for one.

Functions taking ``conn`` run in the caller's transaction.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import text

from .. import bot_profile_changes as profile_changes
from . import catalog
from .errors import GrantLocked, GrantRefused, TagNotFound, TagValidationError
from .facade import BotTags
from .vocabulary import GRANT_TAGS, BehaviorTag, is_canonical_bot_slug

#: Global runtime setting naming the role capability whose holders manage grants.
MANAGER_CAPABILITY_SETTING = "ops_grant_manager_capability"
#: How a bot actor is recorded on the assignment and the audit row.
BOT_ACTOR_PREFIX = "bot:"
REASON_MAX = 1000
ACTOR_MAX = 128

_GRANTS = frozenset(str(tag) for tag in GRANT_TAGS)


@dataclass(frozen=True)
class GrantRecord:
    tag: str
    bot: str
    granted: bool
    assigned_by: str | None
    assigned_at: datetime | None
    changed: bool = False


@dataclass(frozen=True)
class GrantActor:
    """Who asks for a grant write: a signed-in person or a signed bot turn."""

    person: str | None = None
    bot: str | None = None

    @classmethod
    def for_person(cls, actor: object) -> GrantActor:
        """A person actor as BawtHub signed it in; the bot prefix is reserved."""
        value = actor.strip() if isinstance(actor, str) else ""
        if not value or len(value) > ACTOR_MAX or value.startswith(BOT_ACTOR_PREFIX):
            raise TagValidationError(f"actor must be the signed-in person (1-{ACTOR_MAX} "
                                     f"characters, not starting with {BOT_ACTOR_PREFIX!r})")
        return cls(person=value)

    @classmethod
    def for_bot(cls, bot_slug: object) -> GrantActor:
        """A bot actor from its signed turn capability."""
        value = bot_slug.strip().lower() if isinstance(bot_slug, str) else ""
        if not is_canonical_bot_slug(value):
            raise TagValidationError("the signed caller is not a bot slug")
        return cls(bot=value)


def grant_tag(tag: object) -> BehaviorTag:
    """The grant tag named ``tag``; anything else is not found."""
    if not isinstance(tag, str) or tag not in _GRANTS:
        raise TagNotFound(f"{tag!r} is not a grant tag")
    return BehaviorTag(tag)


def check_grant_membership(conn, bot_slug: str, wanted: Iterable[str]) -> None:
    """Refuse a full-set replace that would add or remove a grant tag.

    A replace that keeps grant membership as stored passes, so a client that
    re-sends the full set it read is unaffected. The caller holds the bot lock.
    """
    current = set(conn.execute(text("""
        SELECT tag_slug FROM bot_profile_tags
        WHERE bot_slug = :bot AND tag_slug = ANY(CAST(:grants AS text[]))
    """), {"bot": bot_slug, "grants": sorted(_GRANTS)}).scalars())
    changed = sorted(current ^ (set(wanted) & _GRANTS))
    if changed:
        raise GrantLocked(bot_slug, changed)


def refuse_grant_tag(bot_slug: str, tag: str) -> None:
    """Refuse a single-assignment write of a grant tag outside :func:`write_grant`."""
    if tag in _GRANTS:
        raise GrantLocked(bot_slug, [tag])


def holds_grant(conn, bot_slug: str, tag: BehaviorTag) -> bool:
    """Live check that a non-private bot holds ``tag``; any doubt answers no.

    The snapshot fails closed: unreadable, or a catalog not yet seeded with the
    tag, grants nothing.
    """
    grant_tag(str(tag))
    snapshot = BotTags().snapshot(conn)
    return (snapshot.available and not snapshot.has(bot_slug, BehaviorTag.PRIVATE)
            and snapshot.has(bot_slug, tag))


def manager_capability(global_settings: Mapping[str, object]) -> str | None:
    """The configured manager capability from the global runtime settings, or None."""
    value = global_settings.get(MANAGER_CAPABILITY_SETTING)
    return value.strip() if isinstance(value, str) and value.strip() else None


def manager_roles(conn, capability: str | None, *, lock: bool = False) -> frozenset[str]:
    """Role tags whose capabilities include ``capability`` (none when unset).

    ``lock`` holds the matching catalog rows ``FOR SHARE`` until the caller's
    transaction ends. Every role edit (update, rename, delete) takes its row
    ``FOR UPDATE`` first, so none can take the capability away between this
    check and the caller's commit. A row whose edit committed while this waited
    is re-read, and no longer matches if the capability is gone.
    """
    if not isinstance(capability, str) or not capability.strip():
        return frozenset()
    return frozenset(conn.execute(text(f"""
        SELECT slug FROM bot_tag_catalog
        WHERE kind = 'role' AND capabilities @> jsonb_build_array(CAST(:capability AS text))
        {"FOR SHARE" if lock else ""}
    """), {"capability": capability.strip()}).scalars())


def is_grant_manager(conn, bot_slug: str, capability: str | None, *, lock: bool = False) -> bool:
    """Whether a non-private bot holds a role carrying the manager capability.

    With ``lock`` the answer holds until the caller commits (see :func:`manager_roles`);
    the caller holds the bot's tag lock, which serializes its role assignments.
    """
    return holds_capability(conn, bot_slug, capability, lock=lock)


def holds_capability(conn, bot_slug: str, capability: str | None, *, lock: bool = False) -> bool:
    """Whether a non-private bot holds a role carrying ``capability``; any doubt answers no."""
    roles = manager_roles(conn, capability, lock=lock)
    if not roles:
        return False
    snapshot = BotTags().snapshot(conn)
    if not snapshot.available or snapshot.has(bot_slug, BehaviorTag.PRIVATE):
        return False
    return any(snapshot.has(bot_slug, role) for role in roles if role in snapshot.catalog)


def bot_actor(bot_slug: str) -> str:
    return f"{BOT_ACTOR_PREFIX}{bot_slug}"


def authorize_bot_actor(conn, caller: str, target: str, capability: str | None) -> str:
    """The audit actor for a bot's grant write, or :class:`GrantRefused`."""
    if caller == target:
        raise GrantRefused("self_grant_refused", "a bot can't change its own grants")
    if not is_grant_manager(conn, caller, capability, lock=True):
        raise GrantRefused("not_grant_manager", "this bot can't manage grants")
    snapshot = BotTags().snapshot(conn)
    if (not snapshot.available or target not in snapshot.known_bots
            or snapshot.has(target, BehaviorTag.PRIVATE)):
        # Unknown and private are intentionally indistinguishable.
        raise GrantRefused("access_denied", "no such grantable bot")
    return bot_actor(caller)


def _reason(reason: object) -> str:
    value = reason.strip() if isinstance(reason, str) else ""
    if not value or len(value) > REASON_MAX:
        raise TagValidationError(f"reason is required (1-{REASON_MAX} characters)")
    return value


def _read(conn, tag: str, bot: str) -> GrantRecord:
    row = conn.execute(text("""
        SELECT assigned_by, assigned_at FROM bot_profile_tags
        WHERE bot_slug = :bot AND tag_slug = :tag
    """), {"bot": bot, "tag": tag}).mappings().first()
    if row is None:
        return GrantRecord(tag, bot, False, None, None)
    return GrantRecord(tag, bot, True, row["assigned_by"], row["assigned_at"])


def write_grant(conn, tag: object, bot: object, *, granted: bool, reason: object,
                actor: GrantActor, capability: str | None) -> GrantRecord:
    """Authorize ``actor`` and grant or revoke one grant tag, in one transaction.

    A bot actor's manager role and the target's privacy are read under both
    bots' tag locks (taken in slug order), and the manager role rows are held
    ``FOR SHARE``, so none of them can change between the check and the commit.
    """
    slug = str(grant_tag(tag))
    if not is_canonical_bot_slug(bot):
        raise TagValidationError("bot must be a canonical bot slug (lower-case, trimmed)")
    note = _reason(reason)
    if actor.bot is not None:
        for locked in sorted({actor.bot, bot}):
            catalog.lock_bot(conn, locked)
        by = authorize_bot_actor(conn, actor.bot, bot, capability)
    elif actor.person is not None:
        by = actor.person
    else:
        raise TagValidationError("a grant write needs an actor")
    return _set_grant(conn, slug, bot, granted=granted, by=by, note=note)


def _set_grant(conn, slug: str, bot: str, *, granted: bool, by: str, note: str) -> GrantRecord:
    """Grant or revoke with an audit row; idempotent. Only bots with a profile."""
    catalog.lock_bot(conn, bot)
    if not conn.execute(text("SELECT EXISTS (SELECT 1 FROM bot_profiles WHERE slug = :bot)"),
                        {"bot": bot}).scalar_one():
        raise TagNotFound(f"bot {bot!r} not found")
    before = _read(conn, slug, bot)
    if before.granted == bool(granted):
        return before
    tags_before = list(conn.execute(text(
        "SELECT tag_slug FROM bot_profile_tags WHERE bot_slug = :bot ORDER BY tag_slug"
    ), {"bot": bot}).scalars())
    if granted:
        catalog.assign(conn, bot, [slug], by)
    else:
        catalog.unassign(conn, bot, [slug])
    tags_after = sorted({*tags_before, slug} if granted else set(tags_before) - {slug})
    profile_changes.record_change(
        conn, bot, {profile_changes.TAGS: tags_before}, {profile_changes.TAGS: tags_after},
        {profile_changes.CHANGED_BY: by, profile_changes.CHANGE_NOTE: note}, by)
    after = _read(conn, slug, bot)
    return GrantRecord(after.tag, bot, after.granted, after.assigned_by, after.assigned_at,
                       changed=True)


def list_grants(conn) -> list[GrantRecord]:
    """Every current grant with who made it and when."""
    rows = conn.execute(text("""
        SELECT tag_slug, bot_slug, assigned_by, assigned_at FROM bot_profile_tags
        WHERE tag_slug = ANY(CAST(:grants AS text[])) ORDER BY tag_slug, bot_slug
    """), {"grants": sorted(_GRANTS)}).mappings()
    return [GrantRecord(r["tag_slug"], r["bot_slug"], True, r["assigned_by"], r["assigned_at"])
            for r in rows]
