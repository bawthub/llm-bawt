"""Opaque, session-owned continuations; retain intent, never retrieved text."""
from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

from .contracts import RecallError, RecallRequest, RecallScope, Source


@dataclass(frozen=True)
class Continuation:
    owner: tuple[str, str, str]
    sources: tuple[Source, ...]
    task_ids: tuple[str, ...]
    queries: tuple[str, ...]
    expires_at: float


class HandleStore:
    def __init__(self, *, ttl: float = 900, capacity: int = 256, per_owner: int = 8,
                 clock: Callable[[], float] = time.monotonic):
        self.ttl = ttl
        self.capacity = capacity
        self.per_owner = per_owner
        self.clock = clock
        self._entries: OrderedDict[str, Continuation] = OrderedDict()
        self._lock = threading.Lock()

    def _prune(self):
        now = self.clock()
        for key in list(self._entries):
            if self._entries[key].expires_at <= now:
                del self._entries[key]

    def resolve(self, request: RecallRequest, scope: RecallScope) -> tuple[str, ...]:
        if request.handle is None:
            return (request.query,)
        with self._lock:
            self._prune()
            item = self._entries.get(request.handle)
            if item is None or item.owner != scope.owner:
                raise RecallError("invalid_handle")
            if item.sources != request.sources or item.task_ids != request.task_ids:
                raise RecallError("handle_scope_mismatch")
            if len(item.queries) >= 4:
                raise RecallError("followup_limit")
            return (*item.queries, request.query)

    def put(self, request: RecallRequest, scope: RecallScope, queries: tuple[str, ...]) -> str:
        with self._lock:
            self._prune()
            owned = [key for key, item in self._entries.items() if item.owner == scope.owner]
            while len(owned) >= self.per_owner:
                self._entries.pop(owned.pop(0))
            while len(self._entries) >= self.capacity:
                self._entries.popitem(last=False)
            key = secrets.token_urlsafe(32)
            remaining = self.ttl
            if scope.capability_expires_at is not None:
                remaining = min(remaining, scope.capability_expires_at - time.time())
            if remaining <= 0:
                raise RecallError("unauthorized")
            self._entries[key] = Continuation(scope.owner, request.sources, request.task_ids,
                                               queries, self.clock() + remaining)
            return key
