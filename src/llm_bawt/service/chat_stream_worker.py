"""Helpers for chat stream worker thread operations."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

_log = logging.getLogger(__name__)


def append_text_chunk(full_response_holder: list[str], chunk: Any) -> None:
    """Accumulate only text chunks into the response buffer."""
    if isinstance(chunk, str):
        full_response_holder[0] += chunk


def put_queue_item_threadsafe(loop: asyncio.AbstractEventLoop, chunk_queue: asyncio.Queue, item: Any) -> bool:
    """Best-effort queue delivery from worker thread to event loop.

    Returns False when the loop is already closed/disconnected.
    """
    try:
        loop.call_soon_threadsafe(chunk_queue.put_nowait, item)
        return True
    except RuntimeError:
        return False


def consume_stream_chunks(
    stream_iter,
    *,
    cancel_event: threading.Event,
    loop: asyncio.AbstractEventLoop,
    chunk_queue: asyncio.Queue,
    full_response_holder: list[str],
    stop_on_cancel: bool = False,
) -> bool:
    """Consume stream chunks without losing text when cancelled/disconnected.

    Returns True when cancellation was observed at least once.
    """
    cancelled = False
    if stop_on_cancel:
        stream_iter = _until_cancelled(stream_iter, cancel_event)
    for chunk in stream_iter:
        append_text_chunk(full_response_holder, chunk)
        delivered = put_queue_item_threadsafe(loop, chunk_queue, chunk)
        # DEBUG-292: trace approval chunks through the worker→queue hop
        if isinstance(chunk, dict) and chunk.get("event") == "approval_required":
            _log.info(
                "DEBUG-292 worker: approval_required chunk delivered=%s "
                "loop_closed=%s cancel=%s",
                delivered, loop.is_closed(), cancel_event.is_set(),
            )
        if cancel_event.is_set():
            cancelled = True
    return cancelled or cancel_event.is_set()


def _until_cancelled(stream_iter, cancel_event):
    """Close native generation on its owning thread, not the HTTP reader."""
    iterator = iter(stream_iter)
    try:
        while not cancel_event.is_set():
            try:
                chunk = next(iterator)
            except StopIteration:
                break
            if cancel_event.is_set():
                break
            yield chunk
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
