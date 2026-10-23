"""Live positive bot policy; independent of MCP, retrieval, and storage."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

from sqlalchemy import text

from llm_bawt.utils.bot_partition import sanitize_table_name
from llm_bawt.utils.bounded_worker import BoundedWorker


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
    return _identity(value) and value == value.lower()


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
    caller: TrustedCaller | None
    partitions: Mapping[str, str]
    public_bots: frozenset[str]
    protected_bots: frozenset[str]

    def require_target(self, bot_id: str) -> str:
        if self.caller is not None:
            self.caller.validate()
        if not isinstance(bot_id, str):
            raise BotAccessError("access_denied")
        target = bot_id.strip().lower()
        if target not in self.partitions or (
            target not in self.public_bots
            and (self.caller is None or target != self.caller.bot_id)
        ):
            # Unknown and private-other are intentionally indistinguishable.
            raise BotAccessError("access_denied")
        return target

    def aggregate_scope(
        self, bot_id: str | None = None, *, require_user: bool = False,
    ) -> AggregateScope:
        if self.caller is not None:
            self.caller.validate()
        user = self.caller.user_id if self.caller is not None else None
        if require_user and not user:
            raise BotAccessError("caller_unavailable")
        allowed = set(self.public_bots)
        if self.caller is not None:
            allowed.add(self.caller.bot_id)
        if bot_id is not None:
            allowed = {self.require_target(bot_id)}
        return AggregateScope(
            tuple(PartitionScope(bot, self.partitions[bot],
                                 "global" if bot in self.public_bots else "caller")
                  for bot in sorted(allowed)),
            self.public_bots, user,
        )


_POLICY_WORKER = BoundedWorker(4, "bot-policy")


class BotPolicyResolver:
    """Fresh read per invocation, with bounded work and no cached fallback."""

    def __init__(self, engine):
        self.engine = engine

    async def resolve(self, caller: TrustedCaller | None = None) -> BotAccess:
        try:
            async with asyncio.timeout(3):
                return await _POLICY_WORKER.run(self.resolve_sync, caller)
        except BotAccessError:
            raise
        except Exception:
            raise BotAccessError("policy_unavailable") from None

    def resolve_sync(self, caller: TrustedCaller | None = None) -> BotAccess:
        """For synchronous embedded service callers, never an async MCP handler."""
        if caller is not None:
            caller.validate()
        try:
            with self.engine.connect() as connection, connection.begin():
                connection.execute(text("SET TRANSACTION READ ONLY"))
                connection.execute(text("SET LOCAL statement_timeout = '2000ms'"))
                connection.execute(text("SET LOCAL lock_timeout = '500ms'"))
                rows = list(connection.execute(text(
                    "SELECT slug, include_in_global_search FROM bot_profiles "
                    "ORDER BY slug LIMIT 257"
                )).mappings())
            return self._compile(rows, caller)
        except BotAccessError:
            raise
        except Exception:
            raise BotAccessError("policy_unavailable") from None

    @staticmethod
    def _compile(rows, caller: TrustedCaller | None) -> BotAccess:
        if not rows or len(rows) > 256:
            raise BotAccessError("policy_unavailable")
        partitions: dict[str, str] = {}
        public: set[str] = set()
        for row in rows:
            bot, policy = row["slug"], row["include_in_global_search"]
            if not _canonical(bot) or type(policy) is not bool:
                raise BotAccessError("policy_unavailable")
            partition = sanitize_table_name(bot)
            # PostgreSQL truncates identifiers at 63 bytes. Reject aliases there
            # too, rather than trusting a distinct untruncated Python mapping.
            if (bot in partitions or partition in partitions.values()
                    or len("forgotten_messages_p_" + partition) > 63):
                raise BotAccessError("policy_unavailable")
            partitions[bot] = partition
            if policy:
                public.add(bot)
        if caller is not None and caller.bot_id not in partitions:
            raise BotAccessError("caller_unavailable")
        return BotAccess(caller, MappingProxyType(partitions), frozenset(public),
                         frozenset(partitions.keys() - public))
