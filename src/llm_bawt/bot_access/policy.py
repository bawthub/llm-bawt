"""Live positive bot policy; independent of MCP, retrieval, and storage."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Literal, Mapping

from llm_bawt.bot_tags import BehaviorTag, TagPolicyUnavailable, TagSnapshot
from llm_bawt.bot_tags.vocabulary import is_canonical_bot_slug
from llm_bawt.utils.bot_partition import sanitize_table_name
from .snapshot import PolicySnapshot, SnapshotUnavailable, policy_snapshot


class BotAccessError(RuntimeError):
    """Sanitized failure, never a successful empty or partial search."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class TrustedCaller:
    """Only transport verification or trusted in-process service code creates this."""

    bot_id: str
    user_id: str | None
    kind: Literal["turn", "approved", "service"]
    expires_at: float | None = None

    def validate(self) -> None:
        if (not _canonical(self.bot_id) or self.kind not in {"turn", "approved", "service"}
                or (self.user_id is not None and not _identity(self.user_id))
                or (self.user_id is None and self.kind != "service")
                or (self.expires_at is not None and not self.expires_at > time.time())):
            raise BotAccessError("caller_unavailable")


def _identity(value) -> bool:
    return (isinstance(value, str) and bool(value) and len(value) <= 128
            and value == value.strip() and "\x00" not in value)


def _canonical(value) -> bool:
    # Match the profile store's normalization, not a new slug regex.
    return is_canonical_bot_slug(value)


@dataclass(frozen=True)
class PartitionScope:
    bot_id: str
    partition: str
    authority: Literal["global", "caller"]


@dataclass(frozen=True)
class AggregateScope:
    partitions: tuple[PartitionScope, ...]
    public_authors: frozenset[str]
    user_id: str | None


@dataclass(frozen=True)
class BotAccess:
    """One caller's view of the live bot policy (TASK-1222, tags from TASK-1329).

    ``readable_bots`` (known and not private) gates targeted per-bot tools;
    ``searchable_bots`` (not private and not no_global_search) is the only set
    aggregate search, recall and receipts may span.
    """

    caller: TrustedCaller | None
    partitions: Mapping[str, str]
    tags: TagSnapshot
    _current: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def assert_current(self) -> None:
        if self._current is not None:
            try:
                self._current()
            except SnapshotUnavailable:
                raise BotAccessError("policy_unavailable") from None

    @property
    def private_bots(self) -> frozenset[str]:
        return self.tags.private_bots

    @property
    def searchable_bots(self) -> frozenset[str]:
        return self.tags.searchable_bots

    @property
    def readable_bots(self) -> frozenset[str]:
        return frozenset(self.partitions) - self.private_bots

    def has_tag(self, bot_id: str, tag: str | BehaviorTag) -> bool:
        return self.tags.has(bot_id, tag)

    def _is_caller(self, bot: str) -> bool:
        return self.caller is not None and bot == self.caller.bot_id

    def require_target(self, bot_id: str) -> str:
        """Targeted per-bot tools: readable bots, or the caller itself."""
        self.assert_current()
        if self.caller is not None:
            self.caller.validate()
        if not isinstance(bot_id, str):
            raise BotAccessError("access_denied")
        target = bot_id.strip().lower()
        if target not in self.partitions or (
            target not in self.readable_bots and not self._is_caller(target)
        ):
            # Unknown and private-other are intentionally indistinguishable.
            raise BotAccessError("access_denied")
        return target

    def aggregate_scope(
        self, bot_id: str | None = None, *, require_user: bool = False,
    ) -> AggregateScope:
        """Aggregate search spans searchable bots plus the caller.

        An explicit filter only narrows that set: a bot outside it (private,
        no_global_search or unknown) is the same indistinguishable denial.
        """
        self.assert_current()
        if self.caller is not None:
            self.caller.validate()
        user = self.caller.user_id if self.caller is not None else None
        if require_user and not user:
            raise BotAccessError("caller_unavailable")
        searchable = self.searchable_bots
        allowed = set(searchable)
        if self.caller is not None:
            allowed.add(self.caller.bot_id)
        if bot_id is not None:
            target = bot_id.strip().lower() if isinstance(bot_id, str) else None
            if target is None or target not in self.partitions or target not in allowed:
                raise BotAccessError("access_denied")
            allowed = {target}
        return AggregateScope(
            tuple(PartitionScope(bot, self.partitions[bot],
                                 "global" if bot in searchable else "caller")
                  for bot in sorted(allowed)),
            searchable, user,
        )


class BotPolicyResolver:
    """Memory-only caller views; lifecycle refresh owns all policy DB work."""

    def __init__(self, engine):
        self.engine = engine
        self.snapshot = policy_snapshot(engine) if engine is not None else PolicySnapshot()

    async def resolve(self, caller: TrustedCaller | None = None) -> BotAccess:
        return self.resolve_sync(caller)

    def resolve_sync(self, caller: TrustedCaller | None = None) -> BotAccess:
        if caller is not None:
            caller.validate()
        try:
            policy = self.snapshot.get()
        except SnapshotUnavailable:
            raise BotAccessError("policy_unavailable") from None
        if caller is not None and caller.bot_id not in policy.partitions:
            raise BotAccessError("caller_unavailable")
        return replace(policy, caller=caller,
                       _current=lambda: self.snapshot.require_current(policy))

    @staticmethod
    def _compile(rows, caller: TrustedCaller | None) -> BotAccess:
        rows = list(rows)
        if not rows:
            raise BotAccessError("policy_unavailable")
        try:
            tags = TagSnapshot.from_rows(rows)
        except TagPolicyUnavailable:
            raise BotAccessError("policy_unavailable") from None
        partitions: dict[str, str] = {}
        for bot in tags.tags:
            partition = sanitize_table_name(bot)
            # PostgreSQL truncates identifiers at 63 bytes. Reject aliases there
            # too, rather than trusting a distinct untruncated Python mapping.
            if (partition in partitions.values()
                    or len("forgotten_messages_p_" + partition) > 63):
                raise BotAccessError("policy_unavailable")
            partitions[bot] = partition
        if caller is not None and caller.bot_id not in partitions:
            raise BotAccessError("caller_unavailable")
        return BotAccess(caller, MappingProxyType(partitions), tags)
