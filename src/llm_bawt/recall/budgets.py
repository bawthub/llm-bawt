"""Exact reference-token accounting and cancellation-safe bounded workers."""
from __future__ import annotations

import json
from functools import lru_cache

from llm_bawt.utils.bounded_worker import BoundedWorker as _BoundedWorker

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


class BoundedWorker(_BoundedWorker):
    """Recall-compatible errors over the shared cancellation-safe worker."""

    busy_error = RecallError


DATABASE_WORKER = BoundedWorker(4, "recall-db")
