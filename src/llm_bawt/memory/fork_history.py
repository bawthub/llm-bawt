"""History pool for a forked thread (TASK-259, TASK-2453).

A fork carries the parent thread up to and including the fork message, then
the fork's own turns. Nothing the parent said after the fork may reach it.

- Parent raw rows count through the fork point *whether or not they were later
  summarized*: a summary covering them may also cover post-fork turns, so the
  raw rows are the only leak-free representation.
- When the parent is itself a fork, its pool is in turn its own parent through
  its fork point plus its own rows, so the walk follows ``forked_from`` up the
  chain (bounded by ``MAX_FORK_DEPTH`` and stopped by a cycle).
- Rolling summaries count only when the walk reached the unforked root and
  they end before its first included row. Those describe history every thread
  in the chain shares. Any later summary, or any summary after a truncated
  walk, might describe a discarded branch (or duplicate rows already included).
- The fork's own rows count in full.
"""
from __future__ import annotations

import logging
from typing import Callable, Iterable

logger = logging.getLogger(__name__)

# Ancestors read beyond the direct parent's. Deeper history is dropped, not
# guessed at: the pool then starts at the deepest ancestor reached.
MAX_FORK_DEPTH = 8


def _prefix_through(rows: Iterable, at_message_id: str) -> list | None:
    """Non-summary rows up to and including ``at_message_id``, or None."""
    rows = [m for m in rows if m.role != "summary"]
    cut = next((m for m in rows if str(m.db_id) == str(at_message_id)), None)
    if cut is None:
        return None
    cut_ts = cut.timestamp or 0.0
    return [m for m in rows if (m.timestamp or 0.0) <= cut_ts]


def compose_fork_chain(
    fork_rows: Iterable, summaries: Iterable, *, session_id: str,
    fork_from: dict, read_rows: Callable[[str], Iterable],
    fork_of: Callable[[str], dict | None], max_depth: int = MAX_FORK_DEPTH,
) -> list | None:
    """Merge Message-like rows (``role``, ``timestamp``, ``db_id``) for a fork.

    ``read_rows(session_id)`` returns one thread's own rows (owner-scoped by
    the caller); ``fork_of(session_id)`` returns that thread's ``forked_from``
    or None. Returns None when the direct parent's fork message can't be found,
    so the caller can fall back instead of seeding a wrong cut. A missing cut
    further up, a cycle or the depth bound keeps the raw history gathered so
    far but drops shared summaries.
    """
    prefixes: list[list] = []
    reached_root = False
    seen = {str(session_id)}
    link = fork_from
    while isinstance(link, dict):
        parent_id = str(link.get("session_id") or "")
        at = str(link.get("at_message_id") or "")
        if not parent_id or not at or parent_id in seen:
            if parent_id in seen:
                logger.warning("Fork chain of %s loops at %s", session_id, parent_id)
            break
        if len(prefixes) > max_depth:
            logger.warning(
                "Fork chain of %s deeper than %d; older history dropped",
                session_id, max_depth,
            )
            break
        seen.add(parent_id)
        prefix = _prefix_through(read_rows(parent_id), at)
        if prefix is None:
            if not prefixes:
                return None
            logger.warning(
                "Fork chain of %s: %s has no message %s; older history dropped",
                session_id, parent_id, at,
            )
            break
        prefixes.append(prefix)
        link = fork_of(parent_id)
        reached_root = not isinstance(link, dict)

    inherited = [m for prefix in prefixes for m in prefix]
    # Only a walk that reached an unforked root proves which summaries are
    # shared. After a truncated walk, an older summary may cover an omitted
    # ancestor's discarded branch, so none are admitted.
    shared = []
    if reached_root:
        first_ts = min(((m.timestamp or 0.0) for m in inherited), default=0.0)
        shared = [
            m for m in summaries
            if m.role == "summary" and (m.timestamp or 0.0) < first_ts
        ]
    own = [m for m in fork_rows if m.role != "summary"]
    merged = shared + inherited + own
    merged.sort(key=lambda m: m.timestamp or 0.0)
    return merged


def compose_fork_history(
    parent_rows: Iterable, fork_rows: Iterable, summaries: Iterable,
    *, at_message_id: str,
) -> list | None:
    """A single-level fork: ``parent_rows`` through ``at_message_id``."""
    parent_rows = list(parent_rows)
    return compose_fork_chain(
        fork_rows, summaries, session_id="",
        fork_from={"session_id": "parent", "at_message_id": at_message_id},
        read_rows=lambda _sid: parent_rows, fork_of=lambda _sid: None,
    )


def forked_from_of(session: dict | None) -> dict | None:
    """A session row's ``forked_from`` link, or None."""
    meta = (session or {}).get("session_metadata") or {}
    link = meta.get("forked_from") if isinstance(meta, dict) else None
    return link if isinstance(link, dict) else None
