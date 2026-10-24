"""Live per-bot state for discovery readouts (TASK-1400).

``BotLiveStateReader.read`` answers "who is busy, queued, held or waiting"
for a set of already-visible bots. Every value comes from the subsystem
that owns it, never re-derived here:

* turn: ``turn_admission.classify_targets`` (the admission classifier);
* queued deliveries: ``inter_bot_delivery_query.in_flight_by_target``;
* gate: ``turn_gates.covering_gates`` (strictest live gate, as admission picks);
* pending approvals: ``approval_request_store.pending_counts_by_bot``;
* session: ``BotSessionState.resolve_many`` (the ``last_session_id`` pointer);
* last turn: the newest ended ``turn_logs`` row.

Each source is one query for all bots (no per-bot loop). The read is
unlocked and advisory, like ``peek_target``: it admits nothing. A source that
fails is reported in ``unavailable`` with its fields null, never as idle.
Task counts are not here: tasks live in BawtHub's database.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import bindparam, text

log = logging.getLogger(__name__)

# Newest ended turn per bot. LATERAL keeps it one statement that walks the
# ended_at index backwards per bot instead of sorting the whole table.
_LAST_TURNS = text("""
    SELECT b.bot AS bot_id, t.ended_at, t.status, t.end_reason
    FROM unnest(CAST(:bots AS text[])) AS b(bot)
    JOIN LATERAL (
        SELECT ended_at, status, end_reason FROM turn_logs
        WHERE bot_id = b.bot AND ended_at IS NOT NULL
        ORDER BY ended_at DESC LIMIT 1
    ) AS t ON true
""")
# Portable fallback (SQLite fixtures): same answer, no LATERAL.
_LAST_TURNS_PORTABLE = text("""
    SELECT t.bot_id, t.ended_at, t.status, t.end_reason FROM turn_logs AS t
    WHERE t.bot_id IN :bots AND t.id = (
        SELECT t2.id FROM turn_logs AS t2
        WHERE t2.bot_id = t.bot_id AND t2.ended_at IS NOT NULL
        ORDER BY t2.ended_at DESC, t2.id DESC LIMIT 1
    )
""").bindparams(bindparam("bots", expanding=True))


def _iso(value: Any) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else value


@dataclass(frozen=True)
class BotIdentity:
    """What the reader needs about one visible bot (from its profile row)."""

    slug: str
    bot_type: str | None
    agent_backend: str | None


class BotLiveStateReader:
    """Batched, read-only live state for a set of visible bots."""

    SOURCES = ("turn", "last_turn", "queued_deliveries", "gate", "pending_approvals", "session")

    def __init__(self, engine):
        self.engine = engine

    def read(self, bots: Iterable[BotIdentity]) -> dict[str, dict[str, Any]]:
        """``{slug: state}`` for every bot given; see the module docstring."""
        bots = list({bot.slug: bot for bot in bots}.values())
        if not bots:
            return {}
        if self.engine is None:
            return {bot.slug: self._assemble(bot.slug, {}, set(self.SOURCES)) for bot in bots}
        slugs = [bot.slug for bot in bots]
        loaders: dict[str, Callable[[Any], Mapping[str, Any]]] = {
            "turn": lambda conn: self._turns(conn, slugs),
            "last_turn": lambda conn: self._last_turns(conn, slugs),
            "queued_deliveries": lambda conn: self._queues(conn, slugs),
            "gate": lambda conn: self._gates(conn, bots),
            "pending_approvals": lambda conn: self._approvals(conn, slugs),
            "session": lambda conn: self._sessions(conn, slugs),
        }
        results: dict[str, Mapping[str, Any]] = {}
        failed: set[str] = set()
        with self.engine.connect() as conn:
            for source, load in loaders.items():
                try:
                    results[source] = load(conn)
                except Exception:
                    log.warning("bot live state: %s unavailable", source, exc_info=True)
                    failed.add(source)
                    conn.rollback()  # an aborted PG transaction must not poison the next source
        return {slug: self._assemble(slug, results, failed) for slug in slugs}

    # -- sources: each is one query for every bot -----------------------------

    @staticmethod
    def _turns(conn, slugs: list[str]) -> Mapping[str, Any]:
        from .service.turn_admission import classify_targets

        states = classify_targets(conn, slugs)
        turns = {}
        for slug, state in states.items():
            newest = state.rows[0] if state.rows else None
            turns[slug] = {
                "kind": state.kind,
                "turn_id": state.blocking_id(),
                "started_at": _iso(newest.get("created_at")) if newest else None,
            }
        return turns

    @staticmethod
    def _last_turns(conn, slugs: list[str]) -> Mapping[str, Any]:
        query = _LAST_TURNS if conn.dialect.name == "postgresql" else _LAST_TURNS_PORTABLE
        rows = conn.execute(query, {"bots": slugs}).mappings().all()
        return {
            row["bot_id"]: {
                "ended_at": _iso(row["ended_at"]),
                "status": row["status"],
                "end_reason": row["end_reason"],
            }
            for row in rows
        }

    @staticmethod
    def _queues(conn, slugs: list[str]) -> Mapping[str, Any]:
        from .inter_bot_delivery_query import in_flight_by_target

        queued = in_flight_by_target(conn, slugs)
        return {
            slug: {"count": queued[slug][0], "oldest_at": _iso(queued[slug][1])}
            if slug in queued else {"count": 0, "oldest_at": None}
            for slug in slugs
        }

    @staticmethod
    def _gates(conn, bots: list[BotIdentity]) -> Mapping[str, Any]:
        from .service.turn_gates import covering_gates

        covered = covering_gates(conn, {bot.slug: (bot.bot_type, bot.agent_backend) for bot in bots})
        return {
            slug: {
                "id": block.id, "mode": block.mode, "owner_kind": block.owner_kind,
                "reason": block.reason, "expires_at": _iso(block.expires_at),
            }
            for slug, block in covered.items()
        }

    @staticmethod
    def _approvals(conn, slugs: list[str]) -> Mapping[str, Any]:
        from .approval_request_store import pending_counts_by_bot

        counts = pending_counts_by_bot(conn, slugs)
        return {slug: counts.get(slug, 0) for slug in slugs}

    @staticmethod
    def _sessions(conn, slugs: list[str]) -> Mapping[str, Any]:
        from .bot_session_state import BotSessionState

        resolved = BotSessionState.resolve_many(conn, bot_ids=slugs)
        sessions: dict[str, Any] = {}
        for slug in slugs:
            if slug not in resolved:
                continue  # unknown bot or invalid pointer: unavailable, not "none"
            row = resolved[slug]
            sessions[slug] = None if row is None else {
                "id": row["id"], "started_at": _iso(row["started_at"]),
            }
        return sessions

    # -- assembly ------------------------------------------------------------

    @staticmethod
    def _assemble(slug: str, results: Mapping[str, Mapping[str, Any]], failed: set[str]) -> dict[str, Any]:
        unavailable = set(failed)

        def value(source: str, *, absent: Any = None, required: bool = False) -> Any:
            if source in failed:
                return None
            found = results.get(source, {})
            if slug not in found:
                if required:
                    unavailable.add(source)
                return absent
            return found[slug]

        gate = value("gate")
        state: dict[str, Any] = {
            "turn": value("turn", required=True),
            "last_turn": value("last_turn"),
            "queued_deliveries": value("queued_deliveries", required=True),
            "gated": None if "gate" in failed else gate is not None,
            "gate": gate,
            "pending_approvals": value("pending_approvals", required=True),
            "session": value("session", required=True),
        }
        if unavailable:
            state["unavailable"] = sorted(unavailable)
        return state
