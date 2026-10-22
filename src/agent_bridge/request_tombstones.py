"""Stopped request ids a bridge must refuse to run (TASK-1232 P2).

Every ``chat.abort`` / ``chat.cancel`` writes a tombstone for its
``(session_key, request_id)``, whether or not that request is running. A
``chat.send`` for a tombstoned id is refused at receipt, before any preflight
side effect (session reset, seed). A send never clears its tombstone.

A tombstone lives as long as the command's dedupe key
(:func:`~agent_bridge.command_publisher.command_dedupe_ttl_s`): within that
window the app can't publish the same request id again, so once it lapses
there is nothing left to refuse. Tombstones are in memory only. Losing them on
a bridge restart is safe, because bridges read the command stream with ``">"``
only, so a command published before the restart is never delivered to the new
process.

The store is capped, oldest first; an eviction is logged because it means a
Stop could be forgotten before its TTL.

:class:`AcceptedRequests` is the same store for the requests this process has
accepted (TASK-1233, Snark's ruling on the refused marker). A refused send is
exit evidence only when this process never accepted it
(:meth:`AcceptedRequests.never_accepted`), so that store fails safe: after an
eviction it proves nothing until every evicted entry's TTL has passed.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Callable

from .command_publisher import command_dedupe_ttl_s

logger = logging.getLogger(__name__)

DEFAULT_TOMBSTONE_CAP = 10_000

_Key = tuple[str, str]


class _RequestKeys:
    """TTL-bound, size-capped set of ``(session_key, request_id)``."""

    _noun = "request key"

    def __init__(
        self,
        *,
        cap: int = DEFAULT_TOMBSTONE_CAP,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if cap <= 0:
            raise ValueError("cap must be positive")
        self._cap = cap
        self._clock = clock
        # key -> expiry (clock seconds); insertion order == write order.
        self._expiry: OrderedDict[_Key, float] = OrderedDict()
        # The latest expiry of an entry evicted before its TTL.
        self._evicted_until = float("-inf")

    def add(self, session_key: str, request_id: str) -> None:
        """Record this request; a repeat write renews its TTL."""
        key = (session_key, request_id)
        self._expiry.pop(key, None)
        self._expiry[key] = self._clock() + command_dedupe_ttl_s(request_id)
        self._prune()
        while len(self._expiry) > self._cap:
            evicted, expiry = self._expiry.popitem(last=False)
            self._evicted_until = max(self._evicted_until, expiry)
            logger.warning(
                "%s evicted before its TTL (cap=%d): session=%s request=%s",
                self._noun, self._cap, evicted[0], evicted[1],
            )

    def contains(self, session_key: str, request_id: str) -> bool:
        """True while this request's entry is in force."""
        key = (session_key, request_id)
        expiry = self._expiry.get(key)
        if expiry is None:
            return False
        if expiry <= self._clock():
            del self._expiry[key]
            return False
        return True

    def __len__(self) -> int:
        self._prune()
        return len(self._expiry)

    def _prune(self) -> None:
        # Expiries differ by request kind (24h vs 7d), so they aren't ordered;
        # a full scan is fine at this size and only runs on write.
        now = self._clock()
        for key in [k for k, expiry in self._expiry.items() if expiry <= now]:
            del self._expiry[key]


class RequestTombstones(_RequestKeys):
    """Stopped ``(session_key, request_id)``: a send for one is refused."""

    _noun = "Request tombstone"


class AcceptedRequests(_RequestKeys):
    """Requests this bridge process accepted (in memory, so boot-scoped)."""

    _noun = "Accepted request"

    def never_accepted(self, session_key: str, request_id: str) -> bool:
        """True only when this process provably never accepted the request.

        An eviction before TTL loses that knowledge, so until the latest
        evicted entry would have expired this answers False for everything.
        """
        return self._clock() >= self._evicted_until and not self.contains(session_key, request_id)


__all__ = ["DEFAULT_TOMBSTONE_CAP", "AcceptedRequests", "RequestTombstones"]
