"""Bounded retrieval counters, optionally shared by the MCP and app processes."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from prometheus_client import CollectorRegistry, Counter
from prometheus_client.core import CounterMetricFamily, Metric

if TYPE_CHECKING:
    from .context import WeakOpen
    from .find import FindRequest

PROMETHEUS_METRICS_ENV = "LLM_BAWT_PROMETHEUS_METRICS"
_COUNTER_DB_ENV = "LLM_BAWT_RETRIEVAL_METRICS_DB"
_FIND = "message_retrieval_find"
_OPEN = "message_retrieval_weak_opens"
_FIND_HELP = "Completed initial message searches by outcome."
_OPEN_HELP = "Successful context opens of a served card, by its original rung."
_FIND_OUTCOMES = Counter(_FIND, _FIND_HELP, ("outcome",), registry=None)
_CARD_OPENS = Counter(_OPEN, _OPEN_HELP, ("rung",), registry=None)
_RUNGS = frozenset({"browse", "exact_all", "exact_any", "parts", "parts_loose", "words", "words_loose"})
_OUTCOMES = ("ok", "no_match")
_MAX_LOCK_WAIT = 0.025
for outcome in _OUTCOMES:
    _FIND_OUTCOMES.labels(outcome=outcome)


@cache
def _warn_once(operation: str) -> None:
    logging.getLogger(__name__).warning("Retrieval metrics %s failed", operation)


def _write_count(path: str, family: str, label: str) -> None:
    with closing(sqlite3.connect(path, timeout=_MAX_LOCK_WAIT)) as db, db:
        db.execute("""CREATE TABLE IF NOT EXISTS counters (
            family TEXT NOT NULL, label TEXT NOT NULL, total INTEGER NOT NULL,
            PRIMARY KEY (family, label)) WITHOUT ROWID""")
        db.execute("""INSERT INTO counters VALUES (?, ?, 1)
            ON CONFLICT (family, label) DO UPDATE SET total = total + 1""", (family, label))


class _SharedCounters:
    def __init__(self, path: str) -> None:
        self.path = Path(path).absolute()

    def collect(self) -> Iterator[Metric]:
        values = {}
        try:
            if self.path.exists():
                with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True,
                                             timeout=_MAX_LOCK_WAIT)) as db:
                    values = {(family, label): count for family, label, count in
                              db.execute("SELECT family, label, total FROM counters")}
        except (OSError, sqlite3.Error):
            _warn_once("read")
            return  # A failed read is not a counter reset; leave these samples absent.
        finds = CounterMetricFamily(_FIND, _FIND_HELP, labels=("outcome",))
        for outcome in _OUTCOMES:
            finds.add_metric([outcome], values.get((_FIND, outcome), 0))
        opens = CounterMetricFamily(_OPEN, _OPEN_HELP, labels=("rung",))
        for rung in sorted(_RUNGS | {"other"}):
            if (_OPEN, rung) in values:
                opens.add_metric([rung], values[(_OPEN, rung)])
        yield finds
        yield opens


def register(registry: CollectorRegistry) -> None:
    if path := os.getenv(_COUNTER_DB_ENV):
        registry.register(_SharedCounters(path))
    else:
        registry.register(_FIND_OUTCOMES)
        registry.register(_CARD_OPENS)


def record_find(request: FindRequest, payload: str) -> None:
    if request.handle is not None or not (request.query or "").strip():
        return  # Exclude browsing and paging from the initial-search denominator.
    outcome = json.loads(payload)["status"]
    if outcome in _OUTCOMES:
        _record(_FIND, _FIND_OUTCOMES, outcome)


def record_open(label: WeakOpen) -> None:
    # Only consume the bounded served rung, never query, filter, rank or identity.
    rung = label.card.rung
    _record(_OPEN, _CARD_OPENS, rung if rung in _RUNGS else "other")


def _record(family: str, counter: Counter, label: str) -> None:
    if path := os.getenv(_COUNTER_DB_ENV):
        _write_count(path, family, label)
    else:
        counter.labels(label).inc()


async def best_effort(record: Callable[..., None], *args: object) -> None:
    """Keep the optional recorder off the MCP loop and out of tool failures."""
    if os.getenv(PROMETHEUS_METRICS_ENV) == "1":
        try:
            await asyncio.to_thread(record, *args)
        except Exception:
            _warn_once("write")  # Never include a payload, path or exception details.
