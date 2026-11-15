"""Explicit time hints narrow message recall (TASK-1558), no live DB."""
from datetime import UTC, datetime

import pytest

from llm_bawt.recall.contracts import Evidence, RecallRequest, WindowedEvidence
from llm_bawt.recall.postgres import UNBOUNDED_WINDOW, MessageSource
from llm_bawt.recall.service import RecallService
from llm_bawt.recall.temporal import PAD_SECONDS, time_window
from tests.test_isolated_recall import SCOPE, Model, Source
from tests.test_isolated_recall_sources import Database

# Monday 2026-10-26 00:30 UTC, the time of one replayed miss.
NOW = datetime(2026, 10, 26, 0, 30, tzinfo=UTC).timestamp()


def day(text: str) -> float:
    return datetime.fromisoformat(text).replace(tzinfo=UTC).timestamp()


@pytest.mark.parametrize("query,start,end,residual", [
    # The three replayed temporal misses.
    ("What happened when Nick ran openclaw auth login openai-codex in March, and what was the error?",
     "2026-03-01", "2026-04-01", "What happened when Nick ran openclaw auth login openai-codex , and what was the error?"),
    ("who split inter_bot_delivery.py around 00:19Z Oct 26",
     "2026-10-26", "2026-10-27", "who split inter_bot_delivery.py around 00:19Z"),
    ("Vex AltTab clone local path October 24 2026", "2026-10-24", "2026-10-25", "Vex AltTab clone local path"),
    ("deploy on 2026-03-04T10:00Z failed", "2026-03-04", "2026-03-05", "deploy on failed"),
    ("the 3rd of June gate", "2026-06-03", "2026-06-04", "the gate"),
    ("what did we decide last Tuesday", "2026-10-20", "2026-10-21", "what did we decide"),
    ("what broke yesterday", "2026-10-25", "2026-10-26", "what broke"),
    ("Dec 31 release", "2025-12-31", "2026-01-01", "release"),  # no year: latest past one
    ("decisions in May 2025", "2025-05-01", "2025-06-01", "decisions in"),
    ("since March what changed", "2026-03-01", "2026-10-27", "what changed"),
    # A range word governs every date form, not just a bare month (review B1).
    ("what changed since March 2025", "2025-03-01", "2026-10-27", "what changed"),
    ("deploys since 2025-03-01", "2025-03-01", "2026-10-27", "deploys"),
    ("deploys from the 3rd of June", "2026-06-03", "2026-10-27", "deploys"),
    ("what was planned until March", "1970-01-02", "2026-04-01", "what was planned"),
    ("gate runs before Oct 3", "1970-01-02", "2026-10-03", "gate runs"),
    ("gate runs after Oct 3", "2026-10-04", "2026-10-27", "gate runs"),
    ("throughout March 2025", "2025-03-01", "2025-04-01", "throughout"),
    # An explicit end bounds the range; one open date can't swallow it (review B2).
    ("deployment changes from Oct 3 to Oct 9", "2026-10-03", "2026-10-10", "deployment changes to"),
    ("gate runs after Oct 3 and before Oct 9", "2026-10-04", "2026-10-09", "gate runs and"),
    ("since Sep 1 until Oct 9", "2026-09-01", "2026-10-10", ""),
    ("from 2025-03-01 through March 2025", "2025-03-01", "2025-04-01", ""),
])
def test_explicit_time_phrases_set_a_padded_window(query, start, end, residual):
    window = time_window(query, NOW)
    assert window is not None
    assert window.since == day(start) - PAD_SECONDS
    # Never later than one pad past the moment of the question.
    assert window.until == min(day(end) + PAD_SECONDS, NOW + PAD_SECONDS)
    assert window.residual == residual


@pytest.mark.parametrize("query", [
    "What were the five blockers B1-B5 in Loopy's r9 review?",
    "TASK-1144 and TASK-1430 recreate at 8ee7d89",
    "it may 2 of them",            # verb, not a month
    "march 3 steps forward",       # verb, not a month
    "the gate ran 00:19 ok",       # a clock time alone is not a date
    "may we merge this",
])
def test_ordinary_questions_have_no_window(query):
    assert time_window(query, NOW) is None


def test_an_empty_range_is_no_window():
    assert time_window("anything after Oct 30 2026", NOW) is None
    assert time_window("anything after Oct 9 and before Oct 3", NOW) is None


def test_latest_question_line_with_a_hint_wins():
    window = time_window("what happened in March\nand what changed yesterday", NOW)
    assert window.since == day("2026-10-25") - PAD_SECONDS
    assert window.residual == "what happened in March\nand what changed"


def test_two_dates_cover_both():
    window = time_window("between Oct 3 and Oct 9", NOW)
    assert window.since == day("2026-10-03") - PAD_SECONDS
    assert window.until == day("2026-10-10") + PAD_SECONDS


class FrozenMessages(MessageSource):
    clock = staticmethod(lambda: NOW)


@pytest.mark.asyncio
async def test_undated_question_searches_unbounded_with_no_window_note():
    db = Database([{"id": "m1", "excerpt": "bounded evidence"}])
    rows = await FrozenMessages(db).search("relay background evidence", SCOPE, ())
    assert not isinstance(rows, WindowedEvidence)
    assert len(db.reads) == 1
    statement, params = db.reads[0]
    assert {key: params[key] for key in UNBOUNDED_WINDOW} == UNBOUNDED_WINDOW
    # Every direct-hit tier is windowed; recency is measured from rank_at.
    assert statement.count("m.timestamp >= :since AND m.timestamp < :until") == 6
    assert "LEAST(extract(epoch FROM now()), :rank_at)" in statement


@pytest.mark.asyncio
async def test_dated_question_searches_its_period_without_the_date_terms():
    db = Database([{"id": "m1", "excerpt": "cloned to /tmp/atc"}])
    rows = await FrozenMessages(db).search("Vex AltTab clone local path October 24 2026", SCOPE, ())
    assert [row.id for row in rows] == ["m1"]
    assert rows.time_window == {"from": "2026-10-23", "to": "2026-10-25", "applied": True}
    (_, params), = db.reads
    assert params["since"] == day("2026-10-23") and params["until"] == day("2026-10-26")
    assert params["rank_at"] == params["until"]
    assert not {"october", "24", "2026"} & set(params["terms"])


@pytest.mark.asyncio
async def test_empty_period_falls_back_to_the_whole_history_and_says_so():
    class EmptyFirst(Database):
        def read(self, sql, params):
            super().read(sql, params)
            return [] if len(self.reads) == 1 else self.rows

    db = EmptyFirst([{"id": "m9", "excerpt": "answer from April"}])
    rows = await FrozenMessages(db).search("what was the auth error in March", SCOPE, ())
    assert [row.id for row in rows] == ["m9"]
    assert rows.time_window == {"from": "2026-02-28", "to": "2026-04-01", "applied": False}
    assert len(db.reads) == 2
    assert db.reads[1][1]["since"] == 0.0 and "march" in db.reads[1][1]["terms"]


@pytest.mark.asyncio
async def test_coverage_reports_the_searched_period():
    rows = WindowedEvidence([Evidence("messages", "m1", "reviewer", "operator", "The relay background is cool glass.")],
                            {"from": "2026-10-23", "to": "2026-10-25", "applied": True})
    result = await RecallService([Source(rows)], Model()).recall(
        RecallRequest(query="relay background", sources=("messages",)), SCOPE)
    assert result["status"] == "ok"
    assert result["coverage"]["messages"]["time_window"] == {
        "from": "2026-10-23", "to": "2026-10-25", "applied": True}
