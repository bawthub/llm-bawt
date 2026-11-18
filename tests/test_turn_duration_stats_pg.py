"""Real PostgreSQL percentiles over fixture-only, private-cluster turn logs."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import event, text

from llm_bawt.bot_tags.facade import TagPolicyUnavailable
from llm_bawt.runtime_settings import BotProfile, BotProfileStore
from llm_bawt.service.turn_duration_stats import load_turn_duration_stats
from llm_bawt.service.turn_logs import TurnLog, TurnLogStore
from tests.bot_tag_fixtures import tag

pytestmark = pytest.mark.integration
NOW = datetime(2026, 10, 20, 12, tzinfo=timezone.utc)
ROLE = "fixture_duration_role"
ALPHA, BETA = "fixture-duration-alpha", "fixture-duration-beta"


@pytest.fixture
def engine(private_app_pg):
    BotProfileStore(private_app_pg)
    engine = TurnLogStore(private_app_pg).engine
    assert engine is not None
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO bot_tag_catalog (slug, name, kind) VALUES (:s, :s, 'role') "
            "ON CONFLICT (slug) DO NOTHING"
        ), {"s": ROLE})
        for bot in (ALPHA, BETA, "fixture-duration-hidden", "fixture-duration-quiet",
                    "fixture-duration-column", "fixture-duration-empty"):
            conn.execute(BotProfile.__table__.insert().values(
                slug=bot, name=bot, description="", system_prompt="fixture",
                include_in_global_search=not bot.endswith("column"),
                created_at=NOW, updated_at=NOW,
            ))
            tag(conn, bot, ROLE)
        tag(conn, "fixture-duration-hidden", "private")
        tag(conn, "fixture-duration-quiet", "no_global_search")
    yield engine
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM turn_logs WHERE id LIKE 'fixture-duration-%'"))
        conn.execute(text("DELETE FROM bot_profile_tags WHERE bot_slug LIKE 'fixture-duration-%'"))
        conn.execute(text("DELETE FROM bot_profiles WHERE slug LIKE 'fixture-duration-%'"))


def add(engine, bot=ALPHA, latency=100, **overrides):
    values = {
        "id": f"fixture-duration-{uuid4().hex}", "bot_id": bot, "user_id": "fixture-owner",
        "created_at": NOW - timedelta(hours=1), "ended_at": NOW - timedelta(minutes=30),
        "latency_ms": latency, "status": "ok",
    }
    values.update(overrides)
    with engine.begin() as conn:
        conn.execute(TurnLog.__table__.insert().values(**values))


def load(engine, **kwargs):
    return load_turn_duration_stats(engine, now=NOW, **kwargs)


def test_continuous_percentiles_pool_samples_not_bot_percentiles(engine):
    for latency in (100, 200, 300):
        add(engine, latency=latency)
    add(engine, BETA.upper(), 1000)
    result = load(engine)
    assert result["by_bot"] == [
        {"bot_id": ALPHA, "tags": [ROLE], "count": 3,
         "p50_ms": 200, "p95_ms": pytest.approx(290), "max_ms": 300},
        {"bot_id": BETA, "tags": [ROLE], "count": 1,
         "p50_ms": 1000, "p95_ms": 1000, "max_ms": 1000},
    ]
    assert result["by_tag"] == [{
        "tag": ROLE, "count": 4, "p50_ms": 250, "p95_ms": pytest.approx(895), "max_ms": 1000,
    }]
    assert result["days"] == 7
    assert result["tag_membership"] == "current_snapshot"


def test_current_tag_and_bot_and_user_filters_intersect(engine):
    add(engine, latency=100)
    add(engine, latency=900, user_id="fixture-other")
    add(engine, BETA, 500)
    result = load(engine, tag=ROLE, bot_id=ALPHA.upper(), user_id="fixture-owner")
    assert result["by_bot"][0]["count"] == 1
    assert result["by_tag"][0]["p95_ms"] == 100
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM bot_profile_tags WHERE bot_slug=:b"), {"b": ALPHA})
    assert load(engine, tag=ROLE, bot_id=ALPHA)["by_bot"] == []


def test_multi_tag_membership_does_not_duplicate_bot_samples(engine):
    other_role = "fixture_duration_other"
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO bot_tag_catalog (slug, name, kind) VALUES (:s, :s, 'role') "
            "ON CONFLICT (slug) DO NOTHING"
        ), {"s": other_role})
        tag(conn, ALPHA, other_role)
    add(engine, latency=100)
    add(engine, BETA, 500)
    result = load(engine, tag=ROLE)
    assert [row["count"] for row in result["by_bot"]] == [1, 1]
    assert {row["tag"]: row["count"] for row in result["by_tag"]} == {
        ROLE: 2, other_role: 1,
    }


def test_private_quiet_column_and_unknown_bots_never_contribute(engine):
    for bot in ("fixture-duration-hidden", "fixture-duration-quiet",
                "fixture-duration-column", "fixture-duration-unknown"):
        add(engine, bot, 99999)
        assert load(engine, bot_id=bot)["by_bot"] == []
    add(engine)
    assert load(engine, tag=ROLE)["by_tag"][0]["count"] == 1


def test_window_is_start_inclusive_end_exclusive_and_finished_only(engine):
    add(engine, latency=0, created_at=NOW - timedelta(days=7))
    add(engine, latency=200, ended_at=NOW)
    add(engine, latency=900, created_at=NOW - timedelta(days=7, microseconds=1))
    add(engine, latency=900, created_at=NOW)
    add(engine, latency=900, ended_at=None)
    add(engine, latency=900, ended_at=NOW + timedelta(seconds=1))
    for latency in (None, -1, float("inf"), float("-inf"), float("nan")):
        add(engine, latency=latency)
    assert load(engine)["by_bot"][0]["count"] == 2
    assert load(engine)["by_bot"][0]["p50_ms"] == 100
    assert load(engine, days=1)["by_bot"][0]["count"] == 1


def test_empty_catalog_cohorts_return_no_invented_samples(engine):
    assert load(engine)["by_bot"] == []
    assert load(engine, bot_id="fixture-duration-missing")["by_tag"] == []
    with pytest.raises(ValueError, match="unknown bot tag"):
        load(engine, tag="fixture_missing_tag")


def test_snapshot_failure_fails_closed_without_raw_aggregate(engine, monkeypatch):
    from llm_bawt.service import turn_duration_stats as module
    from llm_bawt.bot_tags.facade import TagSnapshot

    add(engine)
    monkeypatch.setattr(module.BotTags, "snapshot", lambda *args: TagSnapshot.unavailable())
    with pytest.raises(TagPolicyUnavailable):
        load(engine)


def test_aggregate_is_read_only_bounded_and_does_not_load_turn_content(engine):
    add(engine, user_prompt="not-for-stats", request_json='{"secret":"not-for-stats"}')
    statements = []

    def capture(conn, cursor, statement, params, context, many):
        statements.append(statement.lower())

    event.listen(engine, "before_cursor_execute", capture)
    try:
        result = load(engine)
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert result["by_bot"][0]["count"] == 1
    assert "not-for-stats" not in str(result)
    assert any("repeatable read, read only" in sql for sql in statements)
    assert any("statement_timeout = '5000ms'" in sql for sql in statements)
    aggregates = [sql for sql in statements if "percentile_cont" in sql]
    assert len(aggregates) == 1
    assert not any(field in aggregates[0] for field in ("user_prompt", "request_json", "response_text"))
