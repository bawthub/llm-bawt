"""Exact reference-token accounting and cancellation-safe bounded workers."""
from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import tiktoken

from .contracts import RecallError


def compact_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


@lru_cache(maxsize=1)
def encoder():
    return tiktoken.get_encoding("o200k_base")


def tokens(text: str) -> int:
    # Evidence may literally contain model special-token strings.
    return len(encoder().encode(text, disallowed_special=()))


def clip(text: str, budget: int) -> str:
    encoded = encoder().encode(text, disallowed_special=())
    if len(encoded) <= budget:
        return text
    # UTF-8 safe prefix; no replacement characters invented in evidence quotes.
    return encoder().decode_bytes(encoded[:budget]).decode("utf-8", errors="ignore")


class BoundedWorker:
    """No queued jobs. Timed-out synchronous work retains its slot until done.

    Python cannot cancel an in-flight DB/API call. Shielding and releasing from
    the concurrent Future (not the cancelled asyncio waiter) prevents unbounded
    orphan workers. Provider client timeouts remain the underlying deadman.
    """

    def __init__(self, count: int, name: str):
        self._slots = threading.BoundedSemaphore(count)
        self._pool = ThreadPoolExecutor(max_workers=count, thread_name_prefix=name)

    async def run(self, function, *args):
        if not self._slots.acquire(blocking=False):
            raise RecallError("busy")
        try:
            future = self._pool.submit(function, *args)
        except BaseException:
            self._slots.release()
            raise
        future.add_done_callback(lambda _: self._slots.release())
        wrapped = asyncio.wrap_future(future)
        # Retrieve late exceptions even when the caller has timed out.
        wrapped.add_done_callback(lambda f: None if f.cancelled() else f.exception())
        return await asyncio.shield(wrapped)


DATABASE_WORKER = BoundedWorker(4, "recall-db")
