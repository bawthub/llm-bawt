"""Owner-bound, expiring handles with a typed payload (TASK-1323 §5.3).

The ownership/TTL/capacity pattern of ``recall/handles.py``, generalized to any
frozen payload. Payloads hold intent and served identities, never content.
The store is in-process: after a restart, and for another owner, every key
reads as expired, the same as an unknown key.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

T = TypeVar("T")

Owner = tuple[str, str]  # (bot_id, user_id) from the trusted caller


class HandleExpired(LookupError):
    """Unknown, expired, evicted or another owner's handle."""


@dataclass(frozen=True)
class _Entry(Generic[T]):
    owner: Owner
    payload: T
    expires_at: float


class HandleStore(Generic[T]):
    def __init__(self, *, ttl: float = 900, capacity: int = 256, per_owner: int = 8,
                 clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time):
        self.ttl = ttl
        self.capacity = capacity
        self.per_owner = per_owner
        self.clock = clock
        self.wall = wall
        self._entries: OrderedDict[str, _Entry[T]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def new_key() -> str:
        """A fresh key, issued before the response that carries it is measured."""
        return secrets.token_urlsafe(32)

    def _prune(self) -> None:
        now = self.clock()
        for key in [k for k, e in self._entries.items() if e.expires_at <= now]:
            del self._entries[key]

    def get(self, key: str | None, owner: Owner) -> T:
        with self._lock:
            self._prune()
            entry = self._entries.get(key) if isinstance(key, str) else None
            if entry is None or entry.owner != owner:
                raise HandleExpired("handle_expired")
            return entry.payload

    def replace(self, key: str, owner: Owner, expected: T, payload: T) -> bool:
        """Swap the payload only if it is still ``expected`` (by identity).

        Two concurrent pages of one handle would otherwise both number their
        cards ``c<n>.*``. The loser gets ``False`` and serves nothing. The
        expiry never moves.
        """
        with self._lock:
            self._prune()
            entry = self._entries.get(key)
            if entry is None or entry.owner != owner or entry.payload is not expected:
                return False
            self._entries[key] = _Entry(owner, payload, entry.expires_at)
            return True

    def put(self, key: str, owner: Owner, payload: T, *,
            capability_expires_at: float | None = None) -> None:
        """Store ``payload`` under ``key``; a page update keeps the key, a new
        find evicts the owner's oldest handle past ``per_owner``."""
        remaining = self.ttl
        if capability_expires_at is not None:
            remaining = min(remaining, capability_expires_at - self.wall())
        if remaining <= 0:
            raise HandleExpired("handle_expired")
        with self._lock:
            self._prune()
            existing = self._entries.get(key)
            if existing is not None and existing.owner != owner:
                raise HandleExpired("handle_expired")
            if existing is None:
                owned = [k for k, e in self._entries.items() if e.owner == owner]
                while len(owned) >= self.per_owner:
                    del self._entries[owned.pop(0)]
                while len(self._entries) >= self.capacity:
                    self._entries.popitem(last=False)
                expires_at = self.clock() + remaining
            else:
                # A later page never extends the handle's life.
                expires_at = existing.expires_at
            self._entries[key] = _Entry(owner, payload, expires_at)
